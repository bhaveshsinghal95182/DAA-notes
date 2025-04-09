Alright class, let's discuss **Flow Networks**. This topic deals with problems involving capacities and flows, like modeling liquid flow through pipes, data through a network, or assignments between groups.

---

### 1. Flow Network & Maximum Flow Problem

**Problem Statement:**
Given a directed graph `G = (V, E)` where each edge `(u, v)` has a non-negative capacity `c(u, v) >= 0`. Two vertices are designated: a source `s` (where flow originates) and a sink `t` (where flow terminates). A **flow** in `G` is a function `f: V x V -> R` satisfying:

1.  **Capacity Constraint:** For all `u, v` in `V`, `f(u, v) <= c(u, v)`. (Flow cannot exceed capacity).
2.  **Skew Symmetry:** For all `u, v` in `V`, `f(u, v) = -f(v, u)`. (Flow out of `u` to `v` is the negative of flow into `u` from `v`).
3.  **Flow Conservation:** For all `u` in `V - {s, t}`, `sum(f(u, v) for v in V) = 0`. (Total flow entering a vertex equals total flow leaving it, except for source and sink).

The **value** of a flow `f`, denoted `|f|`, is the net flow leaving the source `s`: `|f| = sum(f(s, v) for v in V)`.
The **Maximum Flow Problem** is to find a flow `f` such that its value `|f|` is maximized.

---

### 2. Ford-Fulkerson Method

**Problem Statement:**
How can we find the maximum flow from a source `s` to a sink `t` in a given flow network `G = (V, E, c)`?

**Solution (Why):**
The Ford-Fulkerson method is a general *approach* (not a specific algorithm, but a template) for solving the maximum flow problem. It's an iterative method based on finding **augmenting paths** in the **residual graph**.

1.  **Residual Graph (`Gf`):** Given a network `G` and a flow `f`, the residual graph `Gf` represents the *remaining capacity* for pushing more flow. For each edge `(u, v)` in `G`:
    *   If `f(u, v) < c(u, v)`, `Gf` contains a **forward edge** `(u, v)` with **residual capacity** `cf(u, v) = c(u, v) - f(u, v)`. This represents how much *more* flow can be pushed along `(u, v)`.
    *   If `f(u, v) > 0`, `Gf` contains a **backward edge** `(v, u)` with **residual capacity** `cf(v, u) = f(u, v)`. This represents how much flow *already* pushed along `(u, v)` can be "canceled" or pushed back.

2.  **Augmenting Path:** An augmenting path `p` is a simple path from `s` to `t` in the *residual graph* `Gf`.

3.  **Residual Capacity of a Path:** The residual capacity of an augmenting path `p`, denoted `cf(p)`, is the minimum residual capacity of any edge along that path: `cf(p) = min{cf(u, v) | (u, v) is on p}`. This is the maximum amount of additional flow we can push along path `p`.

4.  **Augmenting the Flow:** If an augmenting path `p` with capacity `cf(p) > 0` is found, we can increase the flow `f` by `cf(p)` along this path. This involves:
    *   For each forward edge `(u, v)` on `p` in `Gf`, increase `f(u, v)` by `cf(p)`.
    *   For each backward edge `(v, u)` on `p` in `Gf`, decrease `f(v, u)` by `cf(p)` (which is equivalent to increasing `f(u, v)` by `cf(p)` due to skew symmetry).

5.  **Iteration:** The Ford-Fulkerson method repeatedly finds an augmenting path in the current residual graph `Gf` and augments the flow `f` along that path, until no more augmenting paths can be found from `s` to `t` in `Gf`.

**Why it works (Max-Flow Min-Cut Theorem):** The algorithm terminates because each augmentation increases the total flow value (assuming integer capacities). The process stops when no path exists from `s` to `t` in the residual graph. At this point, the set of vertices reachable from `s` in `Gf` defines an "s-t cut" (a partition of `V` into `S` and `T` with `s` in `S`, `t` in `T`). The capacity of this cut equals the value of the flow found. The **Max-Flow Min-Cut Theorem** states that the maximum flow value in a network equals the minimum capacity of an s-t cut. Since Ford-Fulkerson finds a flow whose value equals the capacity of *some* cut, it must have found the maximum flow (as it cannot exceed the minimum cut capacity).

**Note:** The efficiency of Ford-Fulkerson depends heavily on *how* the augmenting path is chosen.
*   If paths are chosen arbitrarily, it can be slow or even non-terminating for irrational capacities.
*   **Edmonds-Karp Algorithm:** A specific implementation of Ford-Fulkerson that uses **Breadth-First Search (BFS)** to find the *shortest* augmenting path (in terms of number of edges) in the residual graph. This guarantees termination and runs in `O(V * E^2)` time.
*   Other faster algorithms exist (e.g., Dinic's).

**Pseudocode (Ford-Fulkerson Method - General Template):**

```pseudocode
function Ford_Fulkerson(Graph G = (V, E, c), source s, sink t):
  // Initialize flow f to 0 for all edges
  flow = map[(u, v) -> 0 for all u, v in V]
  max_flow = 0

  while true:
    // 1. Construct the residual graph Gf based on the current flow f
    ResidualGraph Gf = Build_Residual_Graph(G, flow, c)

    // 2. Find an augmenting path p from s to t in Gf
    //    (e.g., using BFS for Edmonds-Karp, or DFS)
    path = Find_Path(Gf, s, t)

    // 3. If no augmenting path exists, we are done
    if path is null:
      break // Exit loop

    // 4. Calculate the residual capacity (bottleneck capacity) of the path
    path_flow = infinity
    for each edge (u, v) in path:
      path_flow = min(path_flow, Gf.capacity(u, v)) // cf(u,v)

    // 5. Augment the flow f along the path p by path_flow
    for each edge (u, v) in path:
      // Check if it's a forward or backward edge in the original graph G
      // This logic is often embedded in how flow is stored (using skew symmetry)
      // or how the residual graph tracks original edges vs. backward edges.

      // Simpler View: Update flow based on residual graph edge type
      if G.has_edge(u, v): // Corresponding forward edge exists
         flow[(u, v)] = flow[(u, v)] + path_flow
      else: // Must be residual backward edge corresponding to (v, u) in G
         flow[(v, u)] = flow[(v, u)] - path_flow

    // Update max_flow value (Optional inside loop, can be computed at end)
    # max_flow = max_flow + path_flow

  // Calculate final max_flow value from flow leaving source s
  final_max_flow = 0
  for v such that G.has_edge(s, v):
      final_max_flow += flow[(s, v)]

  return final_max_flow // Or return the flow map itself
```

---

### 3. Maximum Bipartite Matching

**Problem Statement:**
Given an undirected bipartite graph `G = (L U R, E)`, where `L` and `R` are the two disjoint sets of vertices and `E` contains only edges connecting a vertex in `L` to a vertex in `R`. Find a **matching** `M`, which is a subset of `E` such that no two edges in `M` share a common vertex. The goal is to find a matching `M` with the maximum possible number of edges (`|M|`).

**Solution (Why using Max Flow?):**
This problem can be reduced to a maximum flow problem in a specially constructed flow network.

1.  **Construct the Flow Network `G'`:**
    *   Create a new source node `s'` and a new sink node `t'`.
    *   For every vertex `u` in the left partition `L`, add a directed edge from `s'` to `u` with capacity `c(s', u) = 1`.
    *   For every vertex `v` in the right partition `R`, add a directed edge from `v` to `t'` with capacity `c(v, t') = 1`.
    *   For every original edge `(u, v)` in the bipartite graph `G` (where `u` is in `L` and `v` is in `R`), add a directed edge from `u` to `v` in `G'` with infinite capacity (or practically, capacity >= 1, e.g., 1 is sufficient if other capacities are 1).

2.  **Find Max Flow:** Compute the maximum flow from `s'` to `t'` in the constructed network `G'` using an algorithm like Ford-Fulkerson (e.g., Edmonds-Karp).

3.  **Interpret the Result:** The value of the maximum flow in `G'` is equal to the size of the maximum matching in the original bipartite graph `G`.
    *   **Why?** Consider an integer-valued flow found by Ford-Fulkerson (guaranteed if capacities are integers).
        *   The capacity constraint `c(s', u) = 1` means at most 1 unit of flow can leave `s'` towards any `u` in `L`.
        *   The capacity constraint `c(v, t') = 1` means at most 1 unit of flow can reach `t'` from any `v` in `R`.
        *   The infinite (or capacity 1) edges from `L` to `R` allow flow to pass if the source/sink capacities permit.
        *   An integer flow of 1 unit along a path `s' -> u -> v -> t'` corresponds to selecting the edge `(u, v)` for the matching. Since the flow into `u` and out of `v` is limited to 1, each `u` in `L` and each `v` in `R` can participate in at most one such unit-flow path. This directly mirrors the constraint of a matching (no two edges share a vertex). Maximizing the total flow maximizes the number of such paths, thus maximizing the number of edges in the corresponding matching.
    *   The actual edges in the maximum matching `M` correspond to those edges `(u, v)` (where `u` in `L`, `v` in `R`) that have a flow `f(u, v) = 1` in the maximum flow solution.

**Pseudocode (Max Bipartite Matching using Max Flow):**

```pseudocode
function Max_Bipartite_Matching(BipartiteGraph G = (L, R, E)):
  // L, R: disjoint vertex sets
  // E: edges between L and R

  // 1. Construct the flow network G'
  V_prime = L U R U {s_prime, t_prime}
  E_prime = empty set
  Capacities c_prime = empty map

  // Add edges from source s' to L
  for each u in L:
    Add edge (s_prime, u) to E_prime with c_prime[(s_prime, u)] = 1

  // Add edges from L to R
  for each edge (u, v) in E where u in L, v in R:
    Add edge (u, v) to E_prime with c_prime[(u, v)] = infinity // Or 1

  // Add edges from R to sink t'
  for each v in R:
    Add edge (v, t_prime) to E_prime with c_prime[(v, t_prime)] = 1

  // Create Graph G_prime = (V_prime, E_prime, c_prime)

  // 2. Compute Max Flow in G'
  // Use Ford-Fulkerson or a specific implementation like Edmonds-Karp
  max_flow_value, flow_map = Ford_Fulkerson(G_prime, s_prime, t_prime)

  // 3. The size of the max matching is the max flow value
  max_matching_size = max_flow_value

  // (Optional) Construct the matching M
  Matching M = empty set
  for each edge (u, v) in E where u in L, v in R:
    if flow_map[(u, v)] == 1: // Check flow on the L -> R edge
       Add edge (u, v) to M

  return max_matching_size // Or return the matching M itself
```

This reduction beautifully connects the problem of finding maximum matchings to the well-understood framework of network flows.