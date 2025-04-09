Okay class, let's delve into the powerful technique of Dynamic Programming. It's a cornerstone of algorithm design, especially for optimization problems.

---

### 1. Elements of Dynamic Programming

**Problem Statement:**
How can we solve complex problems efficiently by breaking them down, especially when simpler recursive approaches lead to redundant calculations of the same subproblems?

**Solution (Why):**
Dynamic Programming (DP) is an algorithmic technique for solving optimization problems by breaking them down into simpler, *overlapping* subproblems and solving each subproblem only once. The results of these subproblems are stored (memoized or tabulated) to avoid recomputation.

For a problem to be suitable for DP, it must exhibit two key properties:

1.  **Optimal Substructure:** An optimal solution to the problem contains within it optimal solutions to subproblems. If we make an optimal choice at one step, the remaining subproblem must also be solved optimally to achieve the overall optimal solution. This allows us to build the solution to the larger problem from solutions to smaller ones.
2.  **Overlapping Subproblems:** A recursive approach to the problem involves solving the same subproblems multiple times. DP exploits this by computing the solution to each subproblem just once and storing it, typically in an array or hash table. When the same subproblem is encountered again, the stored result is simply looked up.

There are two main ways to implement DP:

*   **Memoization (Top-Down):** Write the solution recursively, but before computing a subproblem, check if the result is already stored. If yes, return the stored value. If no, compute it, store it, and then return it.
*   **Tabulation (Bottom-Up):** Solve the subproblems in an order that ensures whenever you need the solution to a subproblem, it has already been computed. This usually involves filling a table iteratively, starting from the smallest subproblems and building up to the final solution.

**Pseudocode (General DP Design Steps):**

```pseudocode
// General Steps for Designing a Dynamic Programming Algorithm

1.  CHARACTERIZE the structure of an optimal solution.
    // How can the optimal solution be composed from optimal solutions to subproblems?

2.  RECURSIVELY DEFINE the value of an optimal solution.
    // Write a recurrence relation that expresses the value of an optimal solution
    // for a problem instance in terms of optimal solutions for smaller instances.
    // Identify base cases.

3.  COMPUTE the value of an optimal solution.
    // Implement the recurrence, usually using either:
    // a) Memoization (Top-Down): Recursive function with caching.
    // b) Tabulation (Bottom-Up): Iterative approach filling a table.

4.  (Optional) CONSTRUCT an optimal solution from computed information.
    // If not just the value, but the solution itself is needed, store additional
    // information during step 3 (e.g., choices made) to backtrack and build the solution.
```

---

### 2. 0/1 Knapsack Problem

**Problem Statement:**
Given a set of *n* items, each with a weight `w[i]` and a value `v[i]`, and a knapsack with a maximum weight capacity `W`. Determine the subset of items to include in the knapsack such that the total weight does not exceed `W` and the total value is maximized. You can either take an item entirely (1) or not take it at all (0); you cannot take fractions of items.

**Solution (Why):**
This problem exhibits optimal substructure and overlapping subproblems. Consider the *i*-th item. There are two choices:

1.  **Exclude item *i*:** The maximum value is the same as the maximum value achievable using the first *i-1* items with capacity `W`.
2.  **Include item *i*:** This is only possible if `w[i] <= W`. If included, the maximum value is `v[i]` plus the maximum value achievable using the first *i-1* items with the remaining capacity `W - w[i]`.

We choose the option that yields the higher value. Since the subproblem (e.g., finding the best value for `i-1` items and a certain capacity) might be needed whether we include or exclude item *i*, and also when considering item *i+1*, the subproblems overlap.

Let `dp[i][j]` be the maximum value that can be attained using items from 1 to *i* with a knapsack capacity of *j*.

*   **Base cases:** `dp[0][j] = 0` for all `j` (no items, no value), and `dp[i][0] = 0` for all `i` (no capacity, no value).
*   **Recurrence:**
    *   If `w[i] > j` (item *i* is too heavy for current capacity *j*):
        `dp[i][j] = dp[i-1][j]`
    *   If `w[i] <= j` (item *i* can fit):
        `dp[i][j] = max(dp[i-1][j], v[i] + dp[i-1][j - w[i]])`
        (max of excluding item *i* vs. including item *i*)

The final answer will be `dp[n][W]`.

**Pseudocode (Tabulation):**

```pseudocode
function Knapsack_01(W, weights, values, n):
  // W: Knapsack capacity
  // weights: array of item weights (1-indexed)
  // values: array of item values (1-indexed)
  // n: number of items

  // Create a DP table dp[n+1][W+1] initialized to 0
  let dp[0..n][0..W] be a new table
  for i from 0 to n:
    dp[i][0] = 0
  for j from 0 to W:
    dp[0][j] = 0

  // Fill the DP table
  for i from 1 to n:
    for j from 1 to W:
      // If item i's weight is more than the current capacity j,
      // we cannot include it.
      if weights[i] > j:
        dp[i][j] = dp[i-1][j]
      else:
        // Choose the max of:
        // 1. Not including item i (value is dp[i-1][j])
        // 2. Including item i (value is values[i] + dp[i-1][j - weights[i]])
        dp[i][j] = max(dp[i-1][j], values[i] + dp[i-1][j - weights[i]])

  // The result is the value in the bottom-right cell
  return dp[n][W]
```

---

### 3. Matrix-Chain Multiplication

**Problem Statement:**
Given a sequence (chain) of *n* matrices `A1, A2, ..., An`, where matrix `Ai` has dimensions `p[i-1] x p[i]`. Find the optimal parenthesization (order of multiplications) that minimizes the total number of scalar multiplications required to compute the product `A1 * A2 * ... * An`.

**Solution (Why):**
Matrix multiplication is associative, meaning `(A*B)*C = A*(B*C)`, but the number of scalar multiplications can vary significantly depending on the order. The problem has optimal substructure because if the final multiplication to compute `Ai * ... * Aj` is `(Ai * ... * Ak) * (Ak+1 * ... * Aj)`, then the parenthesizations of the sub-chains `(Ai * ... * Ak)` and `(Ak+1 * ... * Aj)` must themselves be optimal. The problem has overlapping subproblems because the optimal cost for multiplying a sub-chain (e.g., `A2*A3*A4`) might be needed when calculating the optimal cost for several larger chains (e.g., `A1*A2*A3*A4` and `A2*A3*A4*A5`).

Let `dp[i][j]` be the minimum number of scalar multiplications needed to compute the product `Ai * Ai+1 * ... * Aj`.
Let `p` be the array of dimensions, where `Ai` is `p[i-1] x p[i]`. The cost of multiplying matrix `X` (`a x b`) by matrix `Y` (`b x c`) is `a * b * c`.

*   **Base cases:** `dp[i][i] = 0` for all `i` (cost of multiplying a single matrix is 0).
*   **Recurrence:** For `i < j`, we need to find the best place `k` (where `i <= k < j`) to split the chain:
    `dp[i][j] = min { dp[i][k] + dp[k+1][j] + p[i-1] * p[k] * p[j] }` for `k` from `i` to `j-1`.
    This represents the cost of computing `Ai...Ak`, plus the cost of computing `Ak+1...Aj`, plus the cost of multiplying the resulting two matrices (dimensions `p[i-1] x p[k]` and `p[k] x p[j]`).

We compute the `dp[i][j]` values for increasing chain lengths `L = j - i + 1`, starting from `L=2` up to `L=n`. The final answer is `dp[1][n]`.

**Pseudocode (Tabulation):**

```pseudocode
function Matrix_Chain_Order(p, n):
  // p: array of dimensions p[0..n], where matrix Ai has dim p[i-1] x p[i]
  // n: number of matrices

  // Create DP table m[1..n][1..n] for costs
  let m[1..n][1..n] be a new table
  // (Optional) Create table s[1..n-1][2..n] to store optimal split points k
  // let s[1..n-1][2..n] be a new table

  // Cost is 0 for chains of length 1
  for i from 1 to n:
    m[i][i] = 0

  // L is the chain length
  for L from 2 to n:
    for i from 1 to n - L + 1:
      j = i + L - 1
      m[i][j] = infinity // Initialize with a large value
      // Check all possible split points k
      for k from i to j - 1:
        // Cost = cost(Ai..Ak) + cost(Ak+1..Aj) + cost(multiply results)
        cost = m[i][k] + m[k+1][j] + p[i-1] * p[k] * p[j]
        if cost < m[i][j]:
          m[i][j] = cost
          // s[i][j] = k // Store the optimal split point k (optional)

  // The minimum cost for the entire chain A1..An
  return m[1][n]
```

---

### 4. Longest Common Subsequence (LCS)

**Problem Statement:**
Given two sequences, `X = <x1, x2, ..., xm>` and `Y = <y1, y2, ..., yn>`, find a subsequence common to both `X` and `Y` that has the maximum possible length. A subsequence is obtained by deleting zero or more elements from the original sequence, maintaining the relative order of the remaining elements.

**Solution (Why):**
This problem exhibits optimal substructure. Let `Z = <z1, ..., zk>` be an LCS of `X` and `Y`.
1.  If `xm == yn`, then `zk = xm = yn`, and `Zk-1 = <z1, ..., zk-1>` must be an LCS of `Xm-1 = <x1, ..., xm-1>` and `Yn-1 = <y1, ..., yn-1>`.
2.  If `xm != yn`, then `zk != xm` implies `Z` is an LCS of `Xm-1` and `Y`.
3.  If `xm != yn`, then `zk != yn` implies `Z` is an LCS of `X` and `Yn-1`.

Therefore, the LCS of `X` and `Y` depends on the LCS of their prefixes. This leads to overlapping subproblems, as the LCS of smaller prefixes might be needed multiple times.

Let `dp[i][j]` be the length of the LCS of the prefixes `X[1..i]` and `Y[1..j]`.

*   **Base cases:** `dp[i][0] = 0` for all `i`, and `dp[0][j] = 0` for all `j` (LCS with an empty sequence is empty).
*   **Recurrence:**
    *   If `X[i] == Y[j]`:
        `dp[i][j] = 1 + dp[i-1][j-1]` (match the last characters and add 1 to LCS of shorter prefixes)
    *   If `X[i] != Y[j]`:
        `dp[i][j] = max(dp[i-1][j], dp[i][j-1])` (take the longer LCS obtained by excluding either `X[i]` or `Y[j]`)

The final answer (length of the LCS) is `dp[m][n]`.

**Pseudocode (Tabulation):**

```pseudocode
function LCS_Length(X, Y):
  // X: first sequence (1-indexed, length m)
  // Y: second sequence (1-indexed, length n)
  m = length(X)
  n = length(Y)

  // Create DP table dp[0..m][0..n]
  let dp[0..m][0..n] be a new table

  // Initialize base cases (first row and first column)
  for i from 0 to m:
    dp[i][0] = 0
  for j from 0 to n:
    dp[0][j] = 0

  // Fill the rest of the table
  for i from 1 to m:
    for j from 1 to n:
      if X[i] == Y[j]:
        dp[i][j] = 1 + dp[i-1][j-1]
      else:
        dp[i][j] = max(dp[i-1][j], dp[i][j-1])

  // The length of the LCS is in the bottom-right cell
  return dp[m][n]

// Note: To reconstruct the actual LCS sequence, you can backtrack through
// the dp table starting from dp[m][n], following the choices made at each step.
```

---

### 5. All Pairs Shortest Paths (Floyd-Warshall Algorithm)

**Problem Statement:**
Given a weighted directed graph `G = (V, E)` where `V = {1, 2, ..., n}`. The weights `w(i, j)` can be positive, negative, or zero, but assume there are no negative-weight cycles reachable from the source. Find the shortest path lengths between all pairs of vertices `(i, j)`.

**Solution (Why):**
The Floyd-Warshall algorithm uses dynamic programming based on intermediate vertices. Let `dp[i][j][k]` be the length of the shortest path from vertex *i* to vertex *j* using only intermediate vertices from the set `{1, 2, ..., k}`.

The key idea is that for a path from *i* to *j* using intermediates from `{1, ..., k}`, either:
1.  The path does *not* use vertex *k* as an intermediate. In this case, the shortest path length is `dp[i][j][k-1]`.
2.  The path *does* use vertex *k* as an intermediate. Since we assume no negative cycles, this shortest path consists of a shortest path from *i* to *k* (using intermediates from `{1, ..., k-1}`) followed by a shortest path from *k* to *j* (using intermediates from `{1, ..., k-1}`). The length is `dp[i][k][k-1] + dp[k][j][k-1]`.

Thus, `dp[i][j][k] = min(dp[i][j][k-1], dp[i][k][k-1] + dp[k][j][k-1])`.

We can optimize space by noticing that to compute the `k`-th iteration values, we only need the `(k-1)`-th iteration values. We can use a single 2D array `D[i][j]` representing the shortest path from *i* to *j* found *so far*. When considering intermediate vertex *k*, we update `D[i][j]` based on whether going through *k* provides a shorter path.

*   **Initialization:** `D[i][j]` is initialized to `w(i, j)` if `(i, j)` is an edge, 0 if `i == j`, and `infinity` otherwise.
*   **Iteration:** For each intermediate vertex `k` from 1 to `n`, and for all pairs `(i, j)`, update:
    `D[i][j] = min(D[i][j], D[i][k] + D[k][j])`

After iterating through all `k` from 1 to `n`, the matrix `D` will contain the shortest path lengths between all pairs.

**Pseudocode (Floyd-Warshall):**

```pseudocode
function Floyd_Warshall(W, n):
  // W: n x n matrix of edge weights (W[i][j] = weight of edge i->j)
  //    W[i][i] = 0
  //    W[i][j] = infinity if no direct edge from i to j
  // n: number of vertices

  // Initialize the distance matrix D with direct edge weights
  let D[1..n][1..n] be a new matrix
  for i from 1 to n:
    for j from 1 to n:
      D[i][j] = W[i][j]

  // Consider each vertex k as a potential intermediate vertex
  for k from 1 to n:
    // For all pairs of vertices (i, j)
    for i from 1 to n:
      for j from 1 to n:
        // If path i -> k -> j is shorter than the current path i -> j
        if D[i][k] + D[k][j] < D[i][j]:
          D[i][j] = D[i][k] + D[k][j]

  // D now contains the shortest path lengths between all pairs
  // Check for negative cycles: if D[i][i] < 0 for any i, a negative cycle exists.
  return D
```

---

### 5.1. Transitive Closure (Warshall Algorithm)

**Problem Statement:**
Given a directed graph `G = (V, E)`, determine for all pairs of vertices `(i, j)` whether there exists a path from *i* to *j* in `G`. Compute the transitive closure `G* = (V, E*)`, where `(i, j)` is in `E*` if and only if there is a path from *i* to *j* in `G`.

**Solution (Why):**
This is a boolean version of the all-pairs shortest path problem. We can adapt the Floyd-Warshall logic. Let `T[i][j][k]` be true if there is a path from *i* to *j* using only intermediate vertices from `{1, ..., k}`, and false otherwise.

A path exists from *i* to *j* using intermediates from `{1, ..., k}` if either:
1.  A path exists using only intermediates from `{1, ..., k-1}` (`T[i][j][k-1]` is true).
2.  A path exists from *i* to *k* (using intermediates `{1, ..., k-1}`) AND a path exists from *k* to *j* (using intermediates `{1, ..., k-1}`) (`T[i][k][k-1]` AND `T[k][j][k-1]` is true).

So, `T[i][j][k] = T[i][j][k-1] OR (T[i][k][k-1] AND T[k][j][k-1])`.

Again, we can optimize space using a single 2D boolean matrix `T[i][j]`.

*   **Initialization:** `T[i][j]` is true if `i == j` or if `(i, j)` is an edge in `E`, and false otherwise.
*   **Iteration:** For each intermediate vertex `k` from 1 to `n`, and for all pairs `(i, j)`, update:
    `T[i][j] = T[i][j] OR (T[i][k] AND T[k][j])`

After iterating through all `k`, `T[i][j]` will be true if a path exists from *i* to *j*.

**Pseudocode (Warshall Algorithm for Transitive Closure):**

```pseudocode
function Transitive_Closure(Adj, n):
  // Adj: n x n adjacency matrix (Adj[i][j] = 1 if edge i->j exists, 0 otherwise)
  // n: number of vertices

  // Initialize the reachability matrix T
  let T[1..n][1..n] be a new boolean matrix
  for i from 1 to n:
    for j from 1 to n:
      if i == j or Adj[i][j] == 1:
        T[i][j] = true
      else:
        T[i][j] = false

  // Consider each vertex k as a potential intermediate vertex
  for k from 1 to n:
    // For all pairs of vertices (i, j)
    for i from 1 to n:
      for j from 1 to n:
        // Is there a path i -> k and a path k -> j?
        T[i][j] = T[i][j] OR (T[i][k] AND T[k][j])

  // T now represents the transitive closure
  return T
```

---

Dynamic programming is a versatile technique applicable to a wide range of problems where optimal solutions can be built from optimal solutions to overlapping subproblems. Mastering these examples gives you a solid foundation for tackling many other algorithmic challenges.