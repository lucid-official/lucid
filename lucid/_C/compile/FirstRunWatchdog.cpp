// lucid/_C/compile/FirstRunWatchdog.cpp
//
// See FirstRunWatchdog.h for the state machine and why it exists.

#include "FirstRunWatchdog.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

#include <pthread.h>

namespace lucid::compile {

// One watched run, shared by the thread that started it, any thread that joined it,
// and the worker.  ``done`` and ``abandoned`` are final once set and exclusive: a run
// that finished is never given up on, and a run given up on is never reported as
// finished to a later caller.
struct FirstRunLedger::Job {
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    bool abandoned = false;
    std::exception_ptr error;
    std::chrono::steady_clock::time_point deadline;
};

namespace {

// The main thread's stack.  The run is on the caller's thread today, so a worker
// gets no less than the thread it replaces; the default for a secondary thread
// (512 KB) would be a regression for a first run that recurses.
constexpr std::size_t kWorkerStackBytes = 8u << 20;

constexpr const char* kBudgetEnv = "LUCID_COMPILE_FIRST_RUN_TIMEOUT_S";

// What ``exit()`` waits for the workers still running a first run (LCD-1019).  Their number is
// small and their runs end in seconds; a run that does not (LCD-847's hang, inside an MPSGraph
// pass for good) is left, as before, and costs this much at exit.
constexpr std::chrono::seconds kExitGrace{10};

// What the exit handler knows of the first-run workers (LCD-1019): how many there are, from the
// moment one is started (before its thread exists) to the end of its thread's function, and how
// many of those sit at the test seam's gate.  It is not the watchdog's own state but borrows its
// gate (mutex, condition variable, flag), which guards the counts, and holds plain integers on
// purpose: nothing of theirs is destroyed before the exit handler reads them.
//
// A fork copies the counts and not the threads.  A child that ends through ``exit()`` would wait
// out the whole grace for workers it does not have, and a gate mutex copied locked would hold it
// for good, so the forking thread owns the mutex across the fork (no other thread can have it
// locked) and the child starts from zero.
class WorkerCensus {
public:
    // Leaked: the exit handler reads it after every destructor has run.
    static WorkerCensus& get() {
        static WorkerCensus* const census = new WorkerCensus();
        return *census;
    }

    // Borrow the watchdog's gate and register the process hooks, once.  ``exit()`` runs its
    // handlers last-in first-out, so the exit hook has to be registered after the frameworks'
    // own: the first worker does, and by then MPSGraph has made its first log (the compile that
    // produced the executable).
    void attach(std::mutex& mu, std::condition_variable& cv, const bool& held) {
        std::call_once(attached_, [this, &mu, &cv, &held] {
            mu_ = &mu;
            cv_ = &cv;
            held_ = &held;
            [[maybe_unused]] const int exit_hook = std::atexit(&WorkerCensus::drain_at_exit);
            [[maybe_unused]] const int fork_hooks =
                pthread_atfork(&WorkerCensus::before_fork, &WorkerCensus::after_fork_parent,
                               &WorkerCensus::after_fork_child);
        });
    }

    // A worker is about to be started: counted before its thread exists, so the exit handler
    // never finds the gap between starting one and its first line.
    void enter() {
        const std::lock_guard<std::mutex> lock(*mu_);
        ++workers_;
    }

    // A worker's thread is gone, or was never made.
    void leave() {
        {
            const std::lock_guard<std::mutex> lock(*mu_);
            --workers_;
        }
        cv_->notify_all();
    }

    // A worker reached the gate / left it.  The caller holds the gate mutex.  Parking is news to
    // the exit handler, which counts a parked worker as quiet and may be waiting for this one.
    void park() {
        ++gated_;
        cv_->notify_all();
    }
    void unpark() { --gated_; }

private:
    static void drain_at_exit() { get().drain(); }
    static void before_fork() { get().mu_->lock(); }
    static void after_fork_parent() { get().mu_->unlock(); }
    static void after_fork_child() {
        // The condition variable is left as it is: the drain checks its condition before it
        // waits (no workers: it does not), and the gate waits only while the test seam holds it.
        WorkerCensus& census = get();
        census.workers_ = 0;
        census.gated_ = 0;
        census.mu_->unlock();
    }

    void drain() {
        std::unique_lock<std::mutex> lock(*mu_);
        // A worker held at the gate stays there for good, so it is not waited for; one that was
        // just let go is still counted, whether or not it has left the gate yet.  (More parked
        // than counted happens only when a thread that failed to start one runs the run itself.)
        cv_->wait_for(lock, kExitGrace,
                      [this] { return workers_ <= (*held_ ? gated_ : std::size_t{0}); });
    }

    std::once_flag attached_;
    std::mutex* mu_ = nullptr;
    std::condition_variable* cv_ = nullptr;
    const bool* held_ = nullptr;
    std::size_t workers_ = 0;
    std::size_t gated_ = 0;
};

// The longest wait taken at its word (about 31 years): a deadline in a clock's integer
// nanoseconds must not overflow on a budget meant as "forever".
constexpr double kLongestBudgetS = 1e9;

int rank_of(FirstRunState s) noexcept {
    switch (s) {
    case FirstRunState::Refused:
        return 3;
    case FirstRunState::Running:
        return 2;
    case FirstRunState::Warm:
        return 1;
    case FirstRunState::Cold:
        break;
    }
    return 0;
}

std::string seconds_text(double s) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%g", s);
    return buf;
}

// ``LUCID_COMPILE_FIRST_RUN_TIMEOUT_S`` as seconds.  An unusable value is said once on
// stderr and replaced by the default: a typo must not turn the guard off, and must not
// fail a run that has done nothing wrong.
double budget_from_environment() {
    const char* text = std::getenv(kBudgetEnv);
    if (text == nullptr || *text == '\0')
        return FirstRunWatchdog::kDefaultBudgetS;
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(value) || value < 0.0) {
        std::fprintf(stderr,
                     "lucid: ignoring %s=%s (want a non-negative number of seconds; 0 turns the "
                     "guard off); using %g\n",
                     kBudgetEnv, text, FirstRunWatchdog::kDefaultBudgetS);
        return FirstRunWatchdog::kDefaultBudgetS;
    }
    return value;
}

std::string timeout_message(const char* who, double budget) {
    return std::string(who) +
           ": MPSGraph failed: the first run of this graph did not finish within " +
           seconds_text(budget) +
           " s and was given up on. MPSGraph's own thread keeps running it (one core) until this "
           "process exits. Raise the budget with LUCID_COMPILE_FIRST_RUN_TIMEOUT_S, or set it to 0 "
           "to wait for the run as before.";
}

std::string refused_message(const char* who) {
    return std::string(who) +
           ": MPSGraph failed: this graph's first run was given up on earlier in this process "
           "(LUCID_COMPILE_FIRST_RUN_TIMEOUT_S) and is not tried again.";
}

std::string breaker_message(const char* who, std::size_t abandoned) {
    return std::string(who) + ": MPSGraph failed: " + std::to_string(abandoned) +
           " first runs have already been given up on in this process, each leaving a thread "
           "running, so no further first run is started. Set LUCID_COMPILE_FIRST_RUN_TIMEOUT_S=0 "
           "to run without the guard.";
}

// Runs ``undo`` when it goes out of scope unless dismissed: a state change that has to be
// taken back if what follows it throws.
template <typename Undo>
class ScopeExit {
public:
    explicit ScopeExit(Undo undo) : undo_(std::move(undo)) {}
    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;
    ~ScopeExit() {
        if (armed_)
            undo_();
    }
    void dismiss() noexcept { armed_ = false; }

private:
    Undo undo_;
    bool armed_ = true;
};

// What a worker thread runs, and the census it leaves when it is done.
struct Launch {
    std::function<void()> work;
    WorkerCensus* census;
};

void* worker_entry(void* arg) {
    const std::unique_ptr<Launch> launch(static_cast<Launch*>(arg));
    pthread_setname_np("lucid-first-run");
    launch->work();
    // What the run captured (the executable, its queue, the feeds) is let go here, before the
    // census is left: the last release of an Objective-C object runs framework code, and leaving
    // is the point the exit handler stops waiting for this thread.
    launch->work = nullptr;
    launch->census->leave();
    return nullptr;
}

// Start ``work`` on a detached thread with a stack of ``kWorkerStackBytes``, counted in
// ``census`` from before the thread exists until ``work`` and everything it captured are gone.
// False when the system has no thread to give (and nothing is counted then).
bool spawn_detached(std::function<void()> work, WorkerCensus& census) {
    auto boxed = std::make_unique<Launch>(Launch{std::move(work), &census});
    pthread_attr_t attr;
    if (pthread_attr_init(&attr) != 0)
        return false;
    pthread_attr_setstacksize(&attr, kWorkerStackBytes);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_t thread;
    census.enter();
    const int rc = pthread_create(&thread, &attr, &worker_entry, boxed.get());
    pthread_attr_destroy(&attr);
    if (rc != 0) {
        census.leave();
        return false;
    }
    boxed.release();  // the thread owns it now
    return true;
}

}  // namespace

const char* first_run_state_name(FirstRunState state) noexcept {
    switch (state) {
    case FirstRunState::Cold:
        return "cold";
    case FirstRunState::Running:
        return "running";
    case FirstRunState::Warm:
        return "warm";
    case FirstRunState::Refused:
        return "refused";
    }
    return "cold";
}

FirstRunLedger::FirstRunLedger() = default;
FirstRunLedger::~FirstRunLedger() = default;

FirstRunState FirstRunLedger::state(std::uint64_t signature) const {
    const std::lock_guard<std::mutex> lock(mu_);
    const auto it = slots_.find(signature);
    return it == slots_.end() ? FirstRunState::Cold : it->second.state;
}

FirstRunState FirstRunLedger::summary() const {
    const std::lock_guard<std::mutex> lock(mu_);
    FirstRunState best = FirstRunState::Cold;
    for (const auto& entry : slots_) {
        if (rank_of(entry.second.state) > rank_of(best))
            best = entry.second.state;
    }
    return best;
}

FirstRunWatchdog& FirstRunWatchdog::instance() {
    static FirstRunWatchdog* const watchdog = new FirstRunWatchdog();
    return *watchdog;
}

double FirstRunWatchdog::budget_s() {
    const double held = budget_.load(std::memory_order_acquire);
    if (held >= 0.0)
        return held;
    // Two threads may both read the environment; the first store wins and both then
    // agree on it.
    double unset = -1.0;
    budget_.compare_exchange_strong(unset, budget_from_environment(), std::memory_order_acq_rel);
    return budget_.load(std::memory_order_acquire);
}

void FirstRunWatchdog::set_budget_s(std::optional<double> seconds) {
    if (!seconds.has_value()) {
        budget_.store(-1.0, std::memory_order_release);
        return;
    }
    budget_.store(*seconds > 0.0 ? *seconds : 0.0, std::memory_order_release);
}

void FirstRunWatchdog::hold_workers(bool hold) {
    {
        const std::lock_guard<std::mutex> lock(gate_mu_);
        held_ = hold;
    }
    if (!hold)
        gate_cv_.notify_all();
}

void FirstRunWatchdog::wait_gate() {
    WorkerCensus& census = WorkerCensus::get();
    std::unique_lock<std::mutex> lock(gate_mu_);
    census.park();
    gate_cv_.wait(lock, [this] { return !held_; });
    census.unpark();
}

// Wait for ``job`` until it finishes, is given up on, or passes its deadline.  True
// when it finished.  ``poll`` runs between waits with no lock held.
bool FirstRunWatchdog::await(FirstRunLedger::Job& job, const FirstRunPoll& poll) {
    std::unique_lock<std::mutex> lock(job.mu);
    const auto settled = [&job] { return job.done || job.abandoned; };
    while (!settled()) {
        const auto now = Clock::now();
        if (now >= job.deadline)
            return false;
        if (!poll) {
            job.cv.wait_until(lock, job.deadline, settled);
            continue;
        }
        job.cv.wait_until(lock, std::min(job.deadline, now + kPollSlice), settled);
        if (settled())
            break;
        lock.unlock();
        poll();
        lock.lock();
    }
    return job.done;
}

// Give up on ``job``.  False when it finished first (nothing to give up on).  The first
// caller to give up counts it and refuses the signature; later callers find it done.
bool FirstRunWatchdog::abandon(FirstRunLedger& ledger,
                               std::uint64_t signature,
                               FirstRunLedger::Job& job) {
    bool first = false;
    {
        const std::lock_guard<std::mutex> lock(job.mu);
        if (job.done)
            return false;
        first = !job.abandoned;
        job.abandoned = true;
    }
    if (first) {
        abandoned_.fetch_add(1, std::memory_order_relaxed);
        const std::lock_guard<std::mutex> lock(ledger.mu_);
        FirstRunLedger::Slot& slot = ledger.slots_[signature];
        slot.state = FirstRunState::Refused;
        slot.job.reset();
    }
    return true;
}

void FirstRunWatchdog::worker_main(const std::shared_ptr<FirstRunLedger>& ledger,
                                   std::uint64_t signature,
                                   const std::shared_ptr<FirstRunLedger::Job>& job,
                                   const FirstRunBody& body) {
    wait_gate();
    std::exception_ptr error;
    try {
        body();
    } catch (...) {
        // Whatever the run threw (an NSException turned into std::runtime_error, an
        // allocation failure) is the waiter's to rethrow: it must not unwind off the
        // top of a thread.
        error = std::current_exception();
    }
    bool given_up = false;
    {
        const std::lock_guard<std::mutex> lock(job->mu);
        job->error = error;
        job->done = true;
        given_up = job->abandoned;
    }
    job->cv.notify_all();

    // A run given up on stays refused whatever became of it, and ``abandon`` owns that
    // slot: it may not have taken the ledger's lock yet, and the slot must not be seen
    // running without a job in between.
    if (given_up)
        return;
    const std::lock_guard<std::mutex> lock(ledger->mu_);
    FirstRunLedger::Slot& slot = ledger->slots_[signature];
    slot.job.reset();
    slot.state = error ? FirstRunState::Cold : FirstRunState::Warm;
}

void FirstRunWatchdog::run(const std::shared_ptr<FirstRunLedger>& ledger,
                           std::uint64_t signature,
                           const char* who,
                           FirstRunBody body,
                           const FirstRunPoll& poll) {
    FirstRunLedger& led = *ledger;
    enum class Plan { Inline, Join, Start };
    Plan plan = Plan::Inline;
    std::shared_ptr<FirstRunLedger::Job> job;
    const double budget = budget_s();
    {
        const std::lock_guard<std::mutex> lock(led.mu_);
        FirstRunLedger::Slot& slot = led.slots_[signature];
        switch (slot.state) {
        case FirstRunState::Warm:
            plan = Plan::Inline;
            break;
        case FirstRunState::Refused:
            throw FirstRunTimeout(refused_message(who));
        case FirstRunState::Running:
            job = slot.job;
            plan = Plan::Join;
            break;
        case FirstRunState::Cold:
            // Read here and not reserved: callers starting cold executables at the same
            // moment all see the breaker closed and each start a worker, so the threads
            // left behind are the limit plus the first runs in flight when it opens.
            if (breaker_open())
                throw FirstRunTimeout(breaker_message(who, abandoned()));
            job = std::make_shared<FirstRunLedger::Job>();
            job->deadline = Clock::now() +
                            std::chrono::duration_cast<Clock::duration>(
                                std::chrono::duration<double>(std::min(budget, kLongestBudgetS)));
            slot.state = FirstRunState::Running;
            slot.job = job;
            plan = Plan::Start;
            break;
        }
    }

    if (plan == Plan::Inline) {
        body();
        return;
    }

    if (plan == Plan::Start) {
        // A worker that is still inside MPSGraph when ``exit()`` runs the static destructors
        // meets a framework that has begun to come down: MPSGraph destroys the table its
        // ``MPSGraphOSLog`` reads there, and a worker that logs after that faults (SIGSEGV on
        // the way out of a process whose results were all delivered, LCD-1019).  A worker cannot
        // be stopped, so the exit waits for the running ones, up to ``kExitGrace``.  Handlers run
        // last-in first-out and MPSGraph registers its own with its first log, made by the
        // compile that produced this executable, so the census's, registered now, runs before it.
        WorkerCensus& census = WorkerCensus::get();
        census.attach(gate_mu_, gate_cv_, held_);
        // The slot says Running from here, with a job nobody completes until a thread
        // owns the run.  Anything that throws before that (the allocations below) must
        // give the slot back, and let whoever joined go on to run their own call.
        ScopeExit unstarted([&led, signature, &job] {
            {
                const std::lock_guard<std::mutex> lock(job->mu);
                job->done = true;
            }
            job->cv.notify_all();
            const std::lock_guard<std::mutex> lock(led.mu_);
            FirstRunLedger::Slot& slot = led.slots_[signature];
            slot.state = FirstRunState::Cold;
            slot.job.reset();
        });
        const auto shared_body = std::make_shared<FirstRunBody>(std::move(body));
        auto work = [this, ledger, signature, job, shared_body] {
            worker_main(ledger, signature, job, *shared_body);
        };
        const bool spawned = spawn_detached(std::move(work), census);
        unstarted.dismiss();
        if (!spawned) {
            // No thread to watch it on: this thread is the worker, as it was before the
            // guard existed.  The job completes after the run, so whoever joined waits
            // for it, up to the same deadline, like for any other worker.
            worker_main(ledger, signature, job, *shared_body);
        }
    }

    if (!await(*job, poll) && abandon(led, signature, *job))
        throw FirstRunTimeout(timeout_message(who, budget));
    // The run finished: ``done`` was seen under the job's lock, so ``error`` is final.
    if (job->error)
        std::rethrow_exception(job->error);
    // A starter's outputs are in; a joiner waited only for the executable to warm up,
    // and now runs its own call, which is an ordinary run.
    if (plan == Plan::Join)
        body();
}

}  // namespace lucid::compile
