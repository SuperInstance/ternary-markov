# ternary-markov

Markov chains on ternary state spaces — transition matrices, stationary distributions, mixing, and entropy.

## Why This Exists

When your system has three states — bullish/neutral/bearish, positive/neutral/negative, active/idle/failing — you need a Markov chain library that treats three-state spaces as first-class, not as a degenerate case of an N-state chain. This crate provides a focused, efficient implementation for 3×3 transition matrices: fixed-size arrays instead of heap-allocated matrices, specialized solvers for hitting times, and exact stationary distribution computation. The small state space means you get O(1) matrix operations and can run analyses that would be expensive for larger chains.

## Core Concepts

- **`TernaryMarkov`** — A Markov chain on states {0, 1, 2} with a 3×3 transition probability matrix. Each row must sum to 1.0.
- **Stationary distribution** — The probability vector π such that πP = π, computed via power iteration.
- **Mixing time** — The smallest n such that Pⁿ is within ε of the stationary distribution (total variation distance).
- **Entropy rate** — The asymptotic rate of information production: H = −Σᵢ π(i) Σⱼ P(i,j) log₂ P(i,j).

## Quick Start

```toml
# Cargo.toml
[dependencies]
ternary-markov = "0.1"
```

```rust
use ternary_markov::*;

fn main() {
    // Define a Markov chain with 3 states
    let chain = TernaryMarkov::new([
        [0.5, 0.5, 0.0], // from state 0: stay or go to 1
        [0.0, 0.5, 0.5], // from state 1: stay or go to 2
        [0.5, 0.0, 0.5], // from state 2: stay or go to 0
    ]);

    // Stationary distribution
    let pi = chain.stationary_distribution();
    println!("Stationary distribution: {:?}", pi);

    // N-step transition probabilities
    let p10 = chain.n_step(10);
    println!("P^10: {:?}", p10);

    // Mixing time (within ε=0.01)
    let t_mix = chain.mixing_time(0.01);
    println!("Mixing time: {}", t_mix);

    // Entropy rate
    let h = chain.entropy_rate();
    println!("Entropy rate: {:.4} bits/symbol", h);

    // Properties
    println!("Irreducible: {}", chain.is_irreducible());
    println!("Aperiodic: {}", chain.is_aperiodic());
    println!("Ergodic: {}", chain.is_ergodic());

    // Expected hitting time from state 0 to state 2
    let h02 = chain.expected_hitting_time(0, 2);
    println!("E[hitting 0→2] = {:.2}", h02);
}
```

## API Overview

### Chain Construction
- `TernaryMarkov::new(p)` — Create from a 3×3 transition matrix (rows must sum to 1.0)
- `TernaryMarkov::uniform()` — Uniform chain (all transitions = 1/3)

### Properties
- `chain.transition_prob(i, j)` / `chain.matrix()` — Access transition probabilities
- `chain.is_irreducible()` — Can every state reach every other state?
- `chain.is_aperiodic()` — Is every state's period 1?
- `chain.is_ergodic()` — Irreducible + aperiodic (guarantees unique stationary distribution)
- `chain.period(i)` — Period of state i
- `chain.absorbing_states()` — States with P(i,i) = 1.0

### Analysis
- `chain.n_step(n)` — Compute Pⁿ via repeated squaring
- `chain.stationary_distribution()` — Power iteration for π
- `chain.mixing_time(epsilon)` — Steps until Pⁿ converges within ε
- `chain.entropy_rate()` — Shannon entropy rate in bits/symbol
- `chain.expected_hitting_time(i, j)` — Expected steps to reach j from i

### Simulation
- `chain.simulate(start, n)` — Generate a sequence of n steps

### Estimation (free functions)
- `state_frequencies(seq)` — Empirical state frequencies
- `transition_counts(seq)` — Count transitions in a sequence
- `estimate_transition_matrix(seq)` — MLE of transition probabilities

## How It Works

**N-step transitions** use repeated squaring of the 3×3 matrix, giving O(log n) time complexity. Since the matrix is fixed at 3×3, all operations use stack-allocated arrays — no heap allocation.

**Stationary distribution** starts from the uniform distribution and repeatedly multiplies by P on the left (power iteration) until convergence (< 10⁻¹² change).

**Mixing time** iterates Pⁿ for increasing n and checks the maximum absolute difference between each row of Pⁿ and the stationary distribution π.

**Entropy rate** computes the Shannon entropy of each row of P, weighted by the stationary distribution: H = −Σᵢ π(i) Σⱼ P(i,j) log₂ P(i,j).

**Expected hitting time** solves the system h(j)=0, h(i)=1+Σₖ P(i,k)h(k) for i≠j. With 3 states, this reduces to a 2×2 linear system solved by Cramer's rule.

## Use Cases

1. **Market regime modeling** — Model transitions between bullish (0), neutral (1), and bearish (2) market states with empirically estimated transition probabilities.
2. **Sentiment dynamics** — Track how sentiment in text streams transitions between negative, neutral, and positive over time.
3. **Reliability modeling** — Model system health transitions (healthy/degraded/failed) to estimate mean time to failure.
4. **Text generation analysis** — Estimate the entropy rate of a ternary-encoded language model to measure information content.

## Ecosystem

- [`ternary-streaming`](https://github.com/user/ternary-streaming) — Streaming processing for ternary signals
- [`ternary-signals`](https://github.com/user/ternary-signals) — Signal processing (DFT, autocorrelation) for ternary data
- [`ternary-clustering`](https://github.com/user/ternary-clustering) — Clustering algorithms for ternary data

## License

MIT
