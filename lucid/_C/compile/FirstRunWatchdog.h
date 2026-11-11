// lucid/_C/compile/FirstRunWatchdog.h
//
// A time budget for the first run of a compiled executable (LCD-847).
//
// MPSGraph specializes an executable inside its first ``runWithMTLCommandQueue``.
// One of the passes there (``MemrefDeallocationPass``) walks the optimized module
// path by path, and on some graphs -- a long chain of ``cat(head view, computed
// middle, tail view)`` -- it does not come back, and cannot be interrupted.  Lucid
// cannot predict it from the graph it holds, so the only bound is a clock.
//
// This file is pure C++: the MPSGraph call is handed in as a ``FirstRunBody``, so
// the state machine has no Objective-C in it and ``CompiledExecutable.mm`` is the
// only place that knows what a run is.
//
// What a first run becomes
// ------------------------
// Each (executable, feed-shape signature) is in one of four states, kept in a
// :class:`FirstRunLedger` that hangs off the ``MPSGraphExecutable`` itself, so every
// relabelled or cached copy of an executable shares one ledger:
//
//   Cold     -> never run.  The first caller starts a worker thread and waits.
//   Running  -> a worker holds the run.  Another caller joins: it waits for the
//               same absolute deadline, then runs its own call on its own thread.
//   Warm     -> the first run finished.  Later runs go straight to the caller's
//               thread, with no thread, no copy and no allocation.
//   Refused  -> the budget ran out.  Never tried again in this process.
//
// A refused run's thread is left alone: it is inside Apple's code and cannot be
// stopped safely.  It holds one core until the process exits, which is why
// :constant:`kAbandonLimit` such runs end the attempts (a circuit breaker): after
// that a cold first run is not started, only refused.  The count is read when a first
// run starts, so the threads left behind number :constant:`kAbandonLimit` plus the
// first runs already in flight when it was reached: eight threads starting eight cold
// executables at once, all before the first budget ran out, leave eight.

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>

#include "../api.h"

namespace lucid::compile {

// A first run that was given up on, or not started.  A ``RuntimeError`` in Python
// (``FirstRunTimeout``, in the engine's compile module).  The message begins "<who>: MPSGraph
// failed:" like every other failure of a first run, so a caller that matches the
// engine's text still recognises it.
class LUCID_API FirstRunTimeout : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class FirstRunState : std::uint8_t { Cold, Running, Warm, Refused };

// Lower-case name of ``state`` ("cold", "running", "warm", "refused").
LUCID_API const char* first_run_state_name(FirstRunState state) noexcept;

// What a caller waiting on a first run does between waits (every
// ``FirstRunWatchdog::kPollSlice``).  It may throw to stop waiting: the run itself is
// left going, and the next caller joins it.  The Python bindings use it to deliver
// Ctrl-C.
using FirstRunPoll = std::function<void()>;

// The run to watch: executes one MPSGraph first run to completion, or throws.
using FirstRunBody = std::function<void()>;

// The first-run record of one ``MPSGraphExecutable``, per feed-shape signature.
class LUCID_API FirstRunLedger {
public:
    // The signature of an executable whose feed shapes never vary.  A dynamic-batch
    // executable specializes once per batch size and uses a hash of the sizes
    // instead (never this value).
    static constexpr std::uint64_t kStaticSignature = 0;

    FirstRunLedger();
    ~FirstRunLedger();
    FirstRunLedger(const FirstRunLedger&) = delete;
    FirstRunLedger& operator=(const FirstRunLedger&) = delete;

    FirstRunState state(std::uint64_t signature) const;

    // The most telling state across every signature seen: Refused, else Running,
    // else Warm, else Cold.
    FirstRunState summary() const;

private:
    friend class FirstRunWatchdog;
    struct Job;
    struct Slot {
        FirstRunState state = FirstRunState::Cold;
        std::shared_ptr<Job> job;
    };

    mutable std::mutex mu_;
    std::unordered_map<std::uint64_t, Slot> slots_;
};

// The process-wide watchdog.  Its state is a budget and a count of abandoned runs.
class LUCID_API FirstRunWatchdog {
public:
    // Abandoned runs after which a cold first run is no longer started.  Not a hard cap
    // on the threads left behind: first runs already in flight when it is reached can
    // still be abandoned afterwards.
    static constexpr std::size_t kAbandonLimit = 4;
    // The default budget: well above the slowest legitimate first run measured
    // (a 16x16 Jacobi ``eigh``, 12 to 14 s on an M4 Max; three kernels, 93 s).
    static constexpr double kDefaultBudgetS = 120.0;
    // How often a waiting caller wakes to call its ``FirstRunPoll``.
    static constexpr std::chrono::milliseconds kPollSlice{100};

    // Leaked on purpose: a worker still running at process exit must never meet a
    // destroyed watchdog.
    static FirstRunWatchdog& instance();

    // The budget in seconds; zero (or less) means the guard is off and a first run is
    // an ordinary run.  Read from ``LUCID_COMPILE_FIRST_RUN_TIMEOUT_S`` at first use
    // (default :constant:`kDefaultBudgetS`) and then held, as ``LUCID_COMPILE_ASYNC``
    // is.
    double budget_s();

    // Replace the budget for this process; ``std::nullopt`` reads the environment
    // again.
    void set_budget_s(std::optional<double> seconds);

    bool engaged() { return budget_s() > 0.0; }

    // First runs given up on so far: each left a worker running to the end of the
    // process.  Never decreases: a worker that finishes late does not reopen the
    // breaker.
    std::size_t abandoned() const { return abandoned_.load(std::memory_order_relaxed); }
    bool breaker_open() const { return abandoned() >= kAbandonLimit; }

    // Run ``body`` as the first run of ``signature`` under the budget.
    //
    // A warm signature runs ``body`` on this thread.  A cold one runs it on a worker
    // and waits; a running one is joined.  Throws what ``body`` threw, or
    // :class:`FirstRunTimeout` when the budget ran out, when the signature was refused
    // before, or when the circuit breaker is open.  ``poll`` may throw to stop
    // waiting.
    //
    // A caller that joined, and then finds the run finished, runs ``body`` itself:
    // it waited for the executable to be warm, not for another caller's outputs.
    void run(const std::shared_ptr<FirstRunLedger>& ledger,
             std::uint64_t signature,
             const char* who,
             FirstRunBody body,
             const FirstRunPoll& poll = {});

    // Test seam: while held, a worker waits before it starts its run, so a test can
    // make a first run last as long as it needs.  Holding is refused by the bindings
    // unless ``LUCID_COMPILE_TEST_HOOKS=1``.  A held worker waits on a condition and
    // holds no core.
    void hold_workers(bool hold);

private:
    FirstRunWatchdog() = default;

    using Clock = std::chrono::steady_clock;

    void worker_main(const std::shared_ptr<FirstRunLedger>& ledger,
                     std::uint64_t signature,
                     const std::shared_ptr<FirstRunLedger::Job>& job,
                     const FirstRunBody& body);
    bool await(FirstRunLedger::Job& job, const FirstRunPoll& poll);
    bool abandon(FirstRunLedger& ledger, std::uint64_t signature, FirstRunLedger::Job& job);
    void wait_gate();

    // < 0: not read yet.
    std::atomic<double> budget_{-1.0};
    std::atomic<std::size_t> abandoned_{0};

    std::mutex gate_mu_;
    std::condition_variable gate_cv_;
    bool held_ = false;
};

}  // namespace lucid::compile
