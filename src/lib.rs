//! Markov chains on ternary state spaces.
//!
//! Provides transition matrices, stationary distribution computation,
//! mixing time estimation, n-step transitions, absorbing states,
//! irreducibility checking, and entropy rate computation.

#![forbid(unsafe_code)]

/// A Markov chain on states {0, 1, 2} (ternary state space).
#[derive(Clone, Debug)]
pub struct TernaryMarkov {
    /// Transition matrix: P[i][j] = probability of transitioning from state i to state j.
    /// Stored as [[f64; 3]; 3].
    p: [[f64; 3]; 3],
}

impl TernaryMarkov {
    /// Create a new Markov chain from a 3x3 transition matrix.
    /// Each row must sum to 1.0 (within tolerance).
    pub fn new(p: [[f64; 3]; 3]) -> Self {
        for i in 0..3 {
            let row_sum: f64 = p[i].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-9,
                "Row {} sums to {}, expected 1.0",
                i, row_sum
            );
        }
        Self { p }
    }

    /// Create a uniform Markov chain (all transitions equal probability 1/3).
    pub fn uniform() -> Self {
        Self {
            p: [[1.0 / 3.0; 3]; 3],
        }
    }

    /// Get transition probability from state i to state j.
    pub fn transition_prob(&self, i: usize, j: usize) -> f64 {
        self.p[i][j]
    }

    /// Get the full transition matrix.
    pub fn matrix(&self) -> [[f64; 3]; 3] {
        self.p
    }

    /// Compute the n-step transition matrix by matrix power.
    pub fn n_step(&self, n: usize) -> [[f64; 3]; 3] {
        if n == 0 {
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        }
        if n == 1 {
            return self.p;
        }
        // Repeated squaring
        let mut result = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut base = self.p;
        let mut exp = n;
        while exp > 0 {
            if exp % 2 == 1 {
                result = mat_mul(&result, &base);
            }
            base = mat_mul(&base, &base);
            exp /= 2;
        }
        result
    }

    /// Compute stationary distribution using power iteration.
    /// Returns [π(0), π(1), π(2)] such that π * P = π.
    pub fn stationary_distribution(&self) -> [f64; 3] {
        // Start with uniform distribution
        let mut pi = [1.0 / 3.0; 3];
        for _ in 0..10000 {
            let new_pi = left_mat_vec(&pi, &self.p);
            let diff: f64 = new_pi.iter().zip(pi.iter()).map(|(a, b)| (a - b).abs()).sum();
            pi = new_pi;
            if diff < 1e-12 {
                break;
            }
        }
        pi
    }

    /// Check if the chain is irreducible (every state can reach every other state).
    pub fn is_irreducible(&self) -> bool {
        // BFS from each state
        for start in 0..3 {
            let mut reachable = [false; 3];
            let mut stack = vec![start];
            reachable[start] = true;
            while let Some(s) = stack.pop() {
                for j in 0..3 {
                    if self.p[s][j] > 0.0 && !reachable[j] {
                        reachable[j] = true;
                        stack.push(j);
                    }
                }
            }
            if !reachable.iter().all(|&r| r) {
                return false;
            }
        }
        true
    }

    /// Find absorbing states (states that transition to themselves with probability 1).
    pub fn absorbing_states(&self) -> Vec<usize> {
        (0..3).filter(|&i| (self.p[i][i] - 1.0).abs() < 1e-9).collect()
    }

    /// Check if a state is absorbing.
    pub fn is_absorbing(&self, i: usize) -> bool {
        (self.p[i][i] - 1.0).abs() < 1e-9
    }

    /// Estimate mixing time: smallest n such that max |P^n(i, j) - π(j)| < epsilon
    /// for all i, j.
    pub fn mixing_time(&self, epsilon: f64) -> usize {
        let pi = self.stationary_distribution();
        for n in 1..10000 {
            let pn = self.n_step(n);
            let mut max_diff: f64 = 0.0;
            for i in 0..3 {
                for j in 0..3 {
                    let diff = (pn[i][j] - pi[j]).abs();
                    max_diff = max_diff.max(diff);
                }
            }
            if max_diff < epsilon {
                return n;
            }
        }
        10000
    }

    /// Compute entropy rate of the Markov chain.
    /// H = -Σ_i π(i) Σ_j P(i,j) log₂(P(i,j))
    pub fn entropy_rate(&self) -> f64 {
        let pi = self.stationary_distribution();
        let mut h = 0.0;
        for i in 0..3 {
            if pi[i] < 1e-15 {
                continue;
            }
            let mut row_entropy = 0.0;
            for j in 0..3 {
                if self.p[i][j] > 1e-15 {
                    row_entropy -= self.p[i][j] * self.p[i][j].log2();
                }
            }
            h += pi[i] * row_entropy;
        }
        h
    }

    /// Simulate n steps starting from state `start`.
    /// Returns the sequence of visited states.
    pub fn simulate(&self, start: usize, n: usize) -> Vec<usize> {
        assert!(start < 3);
        let mut states = vec![start];
        let mut current = start;
        for _ in 0..n {
            let r = pseudo_random();
            let mut cumulative = 0.0;
            let mut next = current;
            for j in 0..3 {
                cumulative += self.p[current][j];
                if r < cumulative {
                    next = j;
                    break;
                }
            }
            states.push(next);
            current = next;
        }
        states
    }

    /// Compute expected hitting time from state i to state j.
    /// Uses solving a system of equations.
    pub fn expected_hitting_time(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return 0.0;
        }
        // h(j) = 0
        // h(i) = 1 + Σ_k P(i,k) * h(k) for i ≠ j
        // This gives a system: h(i) - Σ_{k≠j} P(i,k)*h(k) = 1 + P(i,j)*0 = 1 for i ≠ j
        // For 3 states, solve by substitution
        // States other than j: let's call them a, b
        let others: Vec<usize> = (0..3).filter(|&s| s != j).collect();
        let n = others.len();
        // Build system: for each i in others, h(i) = 1 + Σ_{k≠j} P(i,k)*h(k)
        // => h(i) - Σ_{k∈others} P(i,k)*h(k) = 1
        // Using simple Gaussian elimination for 2x2
        if n == 0 {
            return 0.0;
        }
        let a = others[0];
        let row_a = [
            1.0 - self.p[a][a],
            if n > 1 { -self.p[a][others[1]] } else { 0.0 },
        ];
        let rhs_a = 1.0;

        if n == 1 {
            return rhs_a / row_a[0];
        }

        let b = others[1];
        let row_b = [
            -self.p[b][a],
            1.0 - self.p[b][b],
        ];
        let rhs_b = 1.0;

        // 2x2 system
        let det = row_a[0] * row_b[1] - row_a[1] * row_b[0];
        if det.abs() < 1e-15 {
            return f64::INFINITY;
        }
        let h_a = (rhs_a * row_b[1] - row_a[1] * rhs_b) / det;
        if i == a { h_a } else { (row_a[0] * rhs_b - rhs_a * row_b[0]) / det }
    }

    /// Check if the chain is aperiodic (period of every state is 1).
    pub fn is_aperiodic(&self) -> bool {
        for i in 0..3 {
            if self.period(i) > 1 {
                return false;
            }
        }
        true
    }

    /// Compute the period of state i.
    pub fn period(&self, i: usize) -> usize {
        let mut return_times = Vec::new();
        for step in 1..=100 {
            let ps = self.n_step(step);
            if ps[i][i] > 1e-10 {
                return_times.push(step);
            }
        }
        if return_times.is_empty() {
            return 0;
        }
        let mut g = return_times[0];
        for &t in &return_times[1..] {
            g = gcd(g, t);
        }
        g
    }

    /// Check if chain has a unique stationary distribution (irreducible + aperiodic).
    pub fn is_ergodic(&self) -> bool {
        self.is_irreducible() && self.is_aperiodic()
    }
}

fn gcd(a: usize, b: usize) -> usize {
    let (mut a, mut b) = (a, b);
    while b > 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn mat_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

fn left_mat_vec(v: &[f64; 3], m: &[[f64; 3]; 3]) -> [f64; 3] {
    let mut result = [0.0; 3];
    for j in 0..3 {
        for i in 0..3 {
            result[j] += v[i] * m[i][j];
        }
    }
    result
}

use std::cell::Cell;

thread_local! {
    static SEED: Cell<u64> = Cell::new(12345);
}

fn pseudo_random() -> f64 {
    SEED.with(|s| {
        let new_seed = s.get().wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s.set(new_seed);
        (new_seed >> 33) as f64 / (1u64 << 31) as f64
    })
}

/// Estimate state frequencies from a sequence of ternary states.
pub fn state_frequencies(seq: &[usize]) -> [f64; 3] {
    let mut counts = [0usize; 3];
    for &s in seq {
        if s < 3 {
            counts[s] += 1;
        }
    }
    let total = counts.iter().sum::<usize>() as f64;
    if total == 0.0 {
        return [0.0; 3];
    }
    [counts[0] as f64 / total, counts[1] as f64 / total, counts[2] as f64 / total]
}

/// Estimate transition counts from a sequence.
pub fn transition_counts(seq: &[usize]) -> [[usize; 3]; 3] {
    let mut counts = [[0usize; 3]; 3];
    for w in seq.windows(2) {
        if w[0] < 3 && w[1] < 3 {
            counts[w[0]][w[1]] += 1;
        }
    }
    counts
}

/// Estimate transition probabilities from a sequence using MLE.
pub fn estimate_transition_matrix(seq: &[usize]) -> [[f64; 3]; 3] {
    let counts = transition_counts(seq);
    let mut p = [[0.0; 3]; 3];
    for i in 0..3 {
        let row_sum: usize = counts[i].iter().sum();
        if row_sum > 0 {
            for j in 0..3 {
                p[i][j] = counts[i][j] as f64 / row_sum as f64;
            }
        } else {
            p[i] = [1.0 / 3.0; 3];
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn almost_equal(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn vec_almost_equal(a: &[f64; 3], b: &[f64; 3]) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-6)
    }

    #[test]
    fn test_identity_chain() {
        let chain = TernaryMarkov::new([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        assert_eq!(chain.absorbing_states(), vec![0, 1, 2]);
        // Each state only reaches itself, so NOT irreducible
        assert!(!chain.is_irreducible());
    }

    #[test]
    fn test_uniform_chain() {
        let chain = TernaryMarkov::uniform();
        assert!(chain.is_irreducible());
        assert!(chain.is_aperiodic());
        let pi = chain.stationary_distribution();
        assert!(vec_almost_equal(&pi, &[1.0 / 3.0; 3]));
    }

    #[test]
    fn test_cyclic_chain() {
        // 0->1->2->0 (deterministic cycle)
        let chain = TernaryMarkov::new([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]);
        assert!(chain.is_irreducible());
        assert!(!chain.is_aperiodic()); // Period 3
        assert_eq!(chain.period(0), 3);
    }

    #[test]
    fn test_n_step_identity() {
        let chain = TernaryMarkov::uniform();
        let p0 = chain.n_step(0);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(almost_equal(p0[i][j], expected));
            }
        }
    }

    #[test]
    fn test_n_step_cyclic() {
        let chain = TernaryMarkov::new([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]);
        let p3 = chain.n_step(3);
        // After 3 steps in a 3-cycle, should be back to identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(almost_equal(p3[i][j], expected), "p3[{}][{}] = {} != {}", i, j, p3[i][j], expected);
            }
        }
    }

    #[test]
    fn test_stationary_distribution() {
        let chain = TernaryMarkov::new([
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ]);
        let pi = chain.stationary_distribution();
        assert!(vec_almost_equal(&pi, &[1.0 / 3.0; 3]));
    }

    #[test]
    fn test_stationary_distribution_asymmetric() {
        let chain = TernaryMarkov::new([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]);
        let pi = chain.stationary_distribution();
        assert!(vec_almost_equal(&pi, &[1.0 / 3.0; 3]));
    }

    #[test]
    fn test_mixing_time() {
        let chain = TernaryMarkov::uniform();
        let t = chain.mixing_time(0.01);
        assert_eq!(t, 1); // Already mixed
    }

    #[test]
    fn test_entropy_rate_uniform() {
        let chain = TernaryMarkov::uniform();
        let h = chain.entropy_rate();
        // For uniform chain, each row has entropy log2(3) ≈ 1.585
        assert!((h - 3.0f64.log2()).abs() < 0.01);
    }

    #[test]
    fn test_entropy_rate_deterministic() {
        let chain = TernaryMarkov::new([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]);
        let h = chain.entropy_rate();
        assert!(h < 0.001); // Deterministic → 0 entropy
    }

    #[test]
    fn test_absorbing_states() {
        let chain = TernaryMarkov::new([
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
        ]);
        assert_eq!(chain.absorbing_states(), vec![0]);
        assert!(chain.is_absorbing(0));
        assert!(!chain.is_absorbing(1));
    }

    #[test]
    fn test_expected_hitting_time() {
        let chain = TernaryMarkov::new([
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ]);
        let h = chain.expected_hitting_time(0, 0);
        assert!(almost_equal(h, 0.0));
    }

    #[test]
    fn test_state_frequencies() {
        let seq = vec![0, 0, 1, 2, 2, 2];
        let freq = state_frequencies(&seq);
        assert!((freq[0] - 2.0 / 6.0).abs() < 1e-9);
        assert!((freq[1] - 1.0 / 6.0).abs() < 1e-9);
        assert!((freq[2] - 3.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_transition_counts() {
        let seq = vec![0, 1, 2, 0, 1];
        let counts = transition_counts(&seq);
        assert_eq!(counts[0][1], 2);
        assert_eq!(counts[1][2], 1);
        assert_eq!(counts[2][0], 1);
    }

    #[test]
    fn test_estimate_transition_matrix() {
        let seq = vec![0, 1, 1, 1, 2, 0];
        let p = estimate_transition_matrix(&seq);
        // From 0: 0->1, 0->... only one transition 0->1 at start
        // Actually: transitions are (0,1), (1,1), (1,1), (1,2), (2,0)
        // From 0: [0,1,0] → 0->1 once
        assert!((p[0][1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_ergodic() {
        let chain = TernaryMarkov::new([
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ]);
        assert!(chain.is_ergodic());
    }

    #[test]
    fn test_period_deterministic_cycle() {
        let chain = TernaryMarkov::new([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]);
        assert_eq!(chain.period(0), 3);
    }

    #[test]
    fn test_n_step_converges_to_stationary() {
        let chain = TernaryMarkov::new([
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ]);
        let pi = chain.stationary_distribution();
        let p100 = chain.n_step(100);
        for i in 0..3 {
            for j in 0..3 {
                assert!((p100[i][j] - pi[j]).abs() < 1e-6,
                    "p100[{}][{}] = {} vs π[{}] = {}", i, j, p100[i][j], j, pi[j]);
            }
        }
    }

    #[test]
    fn test_expected_hitting_time_nontrivial() {
        let chain = TernaryMarkov::new([
            [0.5, 0.25, 0.25],
            [0.25, 0.5, 0.25],
            [0.25, 0.25, 0.5],
        ]);
        // Hitting time from 0 to 0 is 0
        let h00 = chain.expected_hitting_time(0, 0);
        assert!(almost_equal(h00, 0.0));
        // Hitting time from 1 to 0 should be finite
        let h10 = chain.expected_hitting_time(1, 0);
        assert!(h10 > 0.0 && h10 < 100.0);
    }

    #[test]
    fn test_period_aperiodic_chain() {
        let chain = TernaryMarkov::new([
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ]);
        for i in 0..3 {
            assert_eq!(chain.period(i), 1);
        }
    }
}
