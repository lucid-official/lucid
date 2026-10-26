// lucid/_C/core/Generator.cpp
//
// Philox-4x32-10 implementation and Generator methods.
//
// Philox constants (multipliers and Weyl-sequence increments) are taken from
// the original paper: "Random123: A Library of Counter-Based Random Number
// Generators" (Salmon et al., SC 2011).  The 10-round variant provides
// sufficient statistical quality for ML applications while remaining fast on
// scalar code paths.
//
// Algorithm sketch for philox_4x32_10:
//   1. Initialise a 4-word counter register from the 64-bit counter value
//      (words 0..1) and zeros (words 2..3).
//   2. Initialise a 2-word key register from the 64-bit seed.
//   3. For 10 rounds:
//        a. Perform two independent 32×32→64 multiply-high/low operations to
//           mix counter words with their respective multiplier constants.
//        b. XOR the high results with complementary counter words and the key
//           words to produce the new counter state.
//        c. Advance the key by the two Weyl constants (wrapping on overflow).
//   4. The final counter words are the four pseudo-random outputs.

#include "Generator.h"

#include <cstring>

namespace lucid {

namespace {

// Philox-4x32-10 constants from Salmon et al. 2011.
constexpr std::uint32_t PHILOX_M0 = 0xD2511F53u;  // Multiplier for words 0,1
constexpr std::uint32_t PHILOX_M1 = 0xCD9E8D57u;  // Multiplier for words 2,3
constexpr std::uint32_t PHILOX_W0 = 0x9E3779B9u;  // Weyl constant for key word 0
constexpr std::uint32_t PHILOX_W1 = 0xBB67AE85u;  // Weyl constant for key word 1

// Computes hi:lo = a * b as a 64-bit product split into two 32-bit halves.
inline void mulhilo32(std::uint32_t a, std::uint32_t b, std::uint32_t& hi, std::uint32_t& lo) {
    const std::uint64_t product = static_cast<std::uint64_t>(a) * b;
    hi = static_cast<std::uint32_t>(product >> 32);
    lo = static_cast<std::uint32_t>(product);
}

// Applies one Philox-4x32 round to ctr[4] using key[2].  The bijective
// function maps (ctr, key) → new_ctr in a way that passes the BigCrush suite.
//
// Each round performs two independent 32×32→64-bit multiply-high/low pairs
// and recombines the halves with XOR.  The cross-coupling (hi1 feeds new0
// via ctr[1], hi0 feeds new2 via ctr[3]) breaks simple linear patterns.
inline void philox_round(std::uint32_t ctr[4], std::uint32_t key[2]) {
    std::uint32_t hi0, lo0, hi1, lo1;
    mulhilo32(PHILOX_M0, ctr[0], hi0, lo0);
    mulhilo32(PHILOX_M1, ctr[2], hi1, lo1);
    const std::uint32_t new0 = hi1 ^ ctr[1] ^ key[0];
    const std::uint32_t new1 = lo1;
    const std::uint32_t new2 = hi0 ^ ctr[3] ^ key[1];
    const std::uint32_t new3 = lo0;
    ctr[0] = new0;
    ctr[1] = new1;
    ctr[2] = new2;
    ctr[3] = new3;
}

// Advances the key by one step of the Weyl sequence.  This "bumps" the key
// between rounds so that the 10 rounds are not degenerate permutations of the
// same key.
inline void philox_bumpkey(std::uint32_t key[2]) {
    key[0] += PHILOX_W0;
    key[1] += PHILOX_W1;
}

// Core Philox-4x32-10 function.  Maps (counter, seed) → 4 × uint32 outputs.
// The key is not bumped after the final (10th) round — only 9 key steps are
// applied interleaved with 10 mixing rounds.
void philox_4x32_10(std::uint64_t counter, std::uint64_t seed, std::uint32_t out[4]) {
    std::uint32_t ctr[4];
    // Pack the 64-bit counter into the lower two 32-bit words of the counter
    // register; the upper two words are zero (unused counter space).
    ctr[0] = static_cast<std::uint32_t>(counter);
    ctr[1] = static_cast<std::uint32_t>(counter >> 32);
    ctr[2] = 0;
    ctr[3] = 0;

    std::uint32_t key[2];
    key[0] = static_cast<std::uint32_t>(seed);
    key[1] = static_cast<std::uint32_t>(seed >> 32);

    for (int i = 0; i < 10; ++i) {
        philox_round(ctr, key);
        // Don't bump the key after the final round — only 9 bumps for 10 rounds.
        if (i != 9)
            philox_bumpkey(key);
    }

    out[0] = ctr[0];
    out[1] = ctr[1];
    out[2] = ctr[2];
    out[3] = ctr[3];
}

}  // namespace

Generator::Generator(std::uint64_t seed) : seed_(seed), counter_(0) {}

void Generator::set_seed(std::uint64_t seed) {
    seed_ = seed;
    counter_ = 0;
}

void Generator::next_uint32x4(std::uint32_t out[4]) {
    philox_4x32_10(counter_, seed_, out);
    // Post-increment: the counter value used for this call has already been
    // consumed; the next call gets a distinct (counter+1, seed) input pair.
    counter_ += 1;
}

float Generator::next_uniform_float() {
    std::uint32_t buf[4];
    next_uint32x4(buf);

    // Extract the top 24 bits of buf[0] and divide by 2^24 to produce a
    // float in [0, 1).  Using only the upper 24 bits avoids the precision
    // loss that would occur from dividing the full 32-bit value by 2^32 when
    // representing it in a 23-bit mantissa float.
    const std::uint32_t mantissa = buf[0] >> 8;
    return static_cast<float>(mantissa) * (1.0f / static_cast<float>(1u << 24));
}

// The default generator is a process-wide singleton initialised with seed 0.
// It is constructed on first use and lives for the duration of the process.
Generator& default_generator() {
    static Generator g{0};
    return g;
}

}  // namespace lucid
