# Future Integration: ternary-markov

## Current State
Implements Markov chains on ternary state spaces {0, 1, 2}: `TernaryMarkov` with 3×3 transition matrix, n-step transitions via matrix powering, stationary distribution via power iteration, mixing time estimation, absorbing state detection, irreducibility checking, and entropy rate computation.

## Integration Opportunities

### With ternary-cell / room-as-codespace
Room transitions form a Markov chain. Each room is a state; `transition_prob(i, j)` is the probability an agent moves from room i to room j. `stationary_distribution()` reveals the long-term occupancy distribution — which rooms are most visited. `mixing_time()` tells how long until the room population reaches equilibrium. `absorbing_states()` identifies trap rooms that agents never leave.

### With ternary-bayesian
`TernaryMarkov` is a special case of `BayesianNetwork` (chain structure). The transition matrix IS the CPT. Use `belief_propagation` on a HMM (hidden Markov model) where the hidden state is the room's true condition and observations are sensor readings. The forward pass through the HMM is exactly `n_step()` with observation updates.

### With ternary-graph
The transition matrix defines a weighted graph. `ternary-graph::shortest_path()` on the Markov graph (with -log(p) edge weights) finds the most probable path between rooms. Community detection on the Markov graph identifies clusters of rooms that agents tend to move between — natural grouping for ensign assignment.

## Potential in Mature Systems
In construct-core, `TernaryMarkov` models the fleet's state evolution. Each construct type is a state; the transition matrix comes from historical data. `entropy_rate()` measures the fleet's unpredictability — high entropy means the system is exploring, low entropy means it has converged. `mixing_time()` estimates how long after a disruption until the fleet stabilizes. At Layer 0, the chain collapses to a deterministic state machine (absorbing transitions only).

## Cross-Pollination Ideas
**Music × Markov:** Chord progressions are Markov chains. Train `TernaryMarkov` on a corpus of ternary-encoded chord sequences. `stationary_distribution()` reveals the tonal center. `n_step(10)` generates a 10-chord progression. `entropy_rate()` measures harmonic predictability — low entropy = repetitive, high entropy = atonal. Connects to `ternary-music` and `agent-rhythm-rs`.

**Economics × Markov:** Market states (bear/flat/bull) form a ternary Markov chain. `mixing_time()` estimates how long until market sentiment stabilizes. `absorbing_states()` would indicate market lock-in.

## Dependencies for Next Steps
- Higher-order Markov chains (state depends on k previous states)
- Hidden Markov Models wrapping `TernaryMarkov` with observation model
- Online transition matrix estimation from streaming cell data
