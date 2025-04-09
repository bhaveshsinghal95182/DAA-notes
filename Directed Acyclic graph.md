Okay class, today we'll focus on a special type of graph: **Directed Acyclic Graphs (DAGs)**, and algorithms related to them, along with more general shortest path algorithms like Bellman-Ford.

---

### 1. Topological Sort

**Problem Statement:**
Given a Directed Acyclic Graph (DAG) `G = (V, E)`, find a linear ordering of all its vertices such that for every directed edge `(u, v)` from vertex `u` to vertex `v`, vertex `u` comes before vertex `v` in the ordering. If the graph contains a cycle, a topological sort is not possible.

**Solution (Why):**
A topological sort represents a valid sequence for tasks that have dependencies. If task `u` must be completed before task `v`, we draw an edge `u -> v`. A topological sort gives an order to perform the tasks. This ordering is only possible if there are no cyclic dependencies (hence, a DAG).

There are two primary algorithms:

1.  **Kahn's Algorithm (BFS-based):**
    *   Compute the in-degree (number of incoming edges) for each vertex.
    *   Initialize a queue with all vertices having an in-degree of 0.
    *   While the queue is not empty:
        *   Dequeue a vertex `u`. Add `u` to the topological order result.
        *   For each neighbor `v` of `u`:
            *   Decrement the in-degree of `v`.
            *   If the in-degree of `v` becomes 0, enqueue `v`.
    *   If the count of vertices in the result equals the total number of vertices in the graph, the sort is successful. Otherwise, the graph contained a cycle.

2.  **DFS-based Algorithm:**
    *   Perform a Depth First Search (DFS) on the graph.
    *   Keep track of the visiting status of each node (unvisited, visiting, visited). If we encounter a node marked "visiting" during DFS from one of its ancestors, we've found a cycle.
    *   Maintain a list/stack to store the topological order.
    *   When a vertex `u` has finished exploring all its neighbors (i.e., the recursive call `DFS(u)` is about to return), add `u` to the *front* of the result list (or push onto a stack).
    *   The final list (or the reversed stack contents) gives the topological order.

**Pseudocode (Kahn's Algorithm):**

```pseudocode
function Topological_Sort_Kahn(Graph G = (V, E)):
  // V: vertices, E: edges
  n = |V|
  in_degree = array[V] initialized to 0
  Adj = adjacency list representation of G

  // Calculate in-degrees
  for each vertex u in V:
    for each neighbor v in Adj[u]:
      in_degree[v] = in_degree[v] + 1

  // Initialize queue with vertices having in-degree 0
  Queue Q
  for each vertex u in V:
    if in_degree[u] == 0:
      Enqueue(Q, u)

  TopologicalOrder = empty list
  count = 0 // Count of visited vertices

  while Q is not empty:
    u = Dequeue(Q)
    Add u to TopologicalOrder
    count = count + 1

    // Process neighbors
    for each neighbor v in Adj[u]:
      in_degree[v] = in_degree[v] - 1
      if in_degree[v] == 0:
        Enqueue(Q, v)

  // Check for cycle
  if count != n:
    return "Error: Graph has a cycle"
  else:
    return TopologicalOrder
```

**Pseudocode (DFS-based Algorithm):**

```pseudocode
function Topological_Sort_DFS(Graph G = (V, E)):
  Adj = adjacency list representation of G
  visited = set() // Stores fully visited nodes
  visiting = set() // Stores nodes currently in the recursion stack
  TopologicalOrder = empty list // Stores the result

  function DFS_Visit(u):
    nonlocal visited, visiting, TopologicalOrder, Adj
    
    Add u to visiting
    
    for each neighbor v in Adj[u]:
      if v is in visiting:
        // Cycle detected!
        return false // Indicate cycle
      if v is not in visited:
        if not DFS_Visit(v): // Propagate cycle detection
          return false 
          
    Remove u from visiting
    Add u to visited
    // Add u to the FRONT of the list
    Prepend u to TopologicalOrder 
    return true // Indicate success

  // Call DFS_Visit for all unvisited nodes
  for each vertex u in V:
    if u is not in visited:
      if not DFS_Visit(u):
        return "Error: Graph has a cycle"

  return TopologicalOrder
```

---

### 2. Strongly Connected Components (SCCs)

**Problem Statement:**
Given a directed graph `G = (V, E)`, find its Strongly Connected Components (SCCs). An SCC is a maximal subgraph such that for every pair of vertices `u, v` in the subgraph, there is a directed path from `u` to `v` and a directed path from `v` to `u`. (Note: This applies to any directed graph, not just DAGs. A DAG has only trivial SCCs - each node is its own SCC).

**Solution (Why):**
Identifying SCCs helps understand the structure of a graph. If we "collapse" each SCC into a single super-node, the resulting graph of SCCs (the "component graph") is always a DAG. This can simplify analysis or algorithms that work better on DAGs.

Two common algorithms are Kosaraju's and Tarjan's. We'll describe Kosaraju's.

**Kosaraju's Two-Pass Algorithm:**

1.  **Pass 1 (DFS on G):** Perform a DFS on the original graph `G`. Compute the finishing times (or simply the order in which nodes finish their DFS exploration) for all vertices. A stack is often used: push a vertex onto the stack when its DFS `Visit` function finishes.
2.  **Compute Transpose:** Compute the transpose graph `G^T`, which is `G` with all edge directions reversed.
3.  **Pass 2 (DFS on G^T):** Perform DFS on the transpose graph `G^T`. Process the vertices in the order determined by Pass 1 (e.g., by popping from the stack generated in Pass 1). Each tree rooted at a starting node in the DFS forest of `G^T` (when processing in the specified order) forms exactly one SCC.

**Pseudocode (Kosaraju's Algorithm):**

```pseudocode
function Kosaraju_SCC(Graph G = (V, E)):
  n = |V|
  Adj = adjacency list of G
  Adj_T = adjacency list of G^T (transpose)
  visited = array[V] initialized to false
  finish_stack = empty stack // To store finish order from Pass 1

  // --- Pass 1: DFS on G to compute finish order ---
  function DFS1(u):
    visited[u] = true
    for each neighbor v in Adj[u]:
      if not visited[v]:
        DFS1(v)
    Push(finish_stack, u)

  for i from 1 to n: // Iterate through all vertices
    if not visited[i]:
      DFS1(i)

  // --- Pass 2: DFS on G^T using the finish order ---
  fill(visited, false) // Reset visited array
  SCC_List = [] // List to store the SCCs found

  function DFS2(u, current_scc):
    visited[u] = true
    Add u to current_scc
    for each neighbor v in Adj_T[u]:
      if not visited[v]:
        DFS2(v, current_scc)

  while finish_stack is not empty:
    u = Pop(finish_stack)
    if not visited[u]:
      current_scc = empty list
      DFS2(u, current_scc)
      Add current_scc to SCC_List

  return SCC_List
```

---

### 3. Single Source Shortest Paths (SSSP) in DAGs

**Problem Statement:**
Given a weighted Directed Acyclic Graph (DAG) `G = (V, E, w)` (edge weights `w` can be positive, negative, or zero) and a source vertex `s`, find the shortest path lengths from `s` to all other vertices `v` in `V`.

**Solution (Why):**
Since the graph is a DAG, we can process vertices in a topological order. When we process a vertex `u`, we know we have already found the shortest path to `u` because any path to `u` must come from vertices that appear earlier in the topological sort. Therefore, when we relax edges `(u, v)` outgoing from `u`, we are using the finalized shortest path distance to `u` (`d[u]`). This avoids the complexities of cycles and potential repeated relaxations needed in algorithms like Bellman-Ford for general graphs. This makes SSSP on DAGs significantly more efficient.

**Algorithm:**

1.  Topologically sort the vertices of the DAG.
2.  Initialize distances `d[v]` to infinity for all `v` in `V`, and `d[s] = 0`. Initialize predecessors `parent[v]` to null.
3.  Iterate through the vertices `u` in the topological order:
    *   For each edge `(u, v)` outgoing from `u`:
        *   **Relax** the edge: If `d[u] + w(u, v) < d[v]`, then update `d[v] = d[u] + w(u, v)` and `parent[v] = u`.

**Pseudocode:**

```pseudocode
function DAG_Shortest_Paths(Graph G = (V, E, w), source s):
  // G must be a DAG
  // w: edge weights (can be negative)
  // s: source vertex

  TopologicalOrder = Topological_Sort(G) // Use Kahn's or DFS-based

  d = array[V] initialized to infinity
  parent = array[V] initialized to null
  d[s] = 0

  // Process vertices in topological order
  for each vertex u in TopologicalOrder:
    // Skip unreachable vertices if needed (d[u] == infinity)
    if d[u] != infinity:
      // Relax outgoing edges
      for each neighbor v of u:
        weight_uv = w(u, v) // Get edge weight
        if d[u] + weight_uv < d[v]:
          d[v] = d[u] + weight_uv
          parent[v] = u

  // d[v] now contains the shortest path length from s to v
  // parent[v] can be used to reconstruct the paths
  return (d, parent)
```
*Complexity: O(V + E) due to topological sort and single relaxation per edge.*

---

### 4. Single Source Shortest Path (Bellman-Ford Algorithm)

**Problem Statement:**
Given a weighted directed graph `G = (V, E, w)` where edge weights `w` can be positive, negative, or zero, and a source vertex `s`. Find the shortest path lengths from `s` to all other vertices `v` in `V`. If the graph contains a negative-weight cycle reachable from `s`, detect it and report its existence.

**Solution (Why):**
Dijkstra's algorithm fails if there are negative edge weights. Bellman-Ford handles negative weights. It works by iteratively relaxing *all* edges in the graph. If there's a shortest path from `s` to `v` using at most `k` edges, Bellman-Ford guarantees to find it after the `k`-th iteration. Since any simple shortest path in a graph with `|V|` vertices can have at most `|V|-1` edges, iterating `|V|-1` times ensures finding all shortest paths, provided there are no negative cycles.

A `|V|`-th iteration can be used to detect negative cycles: if any edge can still be relaxed in the `|V|`-th pass, it means a shorter path was found using `|V|` edges, which implies a negative cycle exists on that path.

**Algorithm:**

1.  Initialize distances `d[v]` to infinity for all `v` in `V`, and `d[s] = 0`. Initialize `parent[v]` to null.
2.  Repeat `|V|-1` times:
    *   For each edge `(u, v)` in `E`:
        *   **Relax** the edge: If `d[u] + w(u, v) < d[v]`, then update `d[v] = d[u] + w(u, v)` and `parent[v] = u`.
3.  **(Negative Cycle Detection):** Iterate one more time through all edges `(u, v)` in `E`:
    *   If `d[u] + w(u, v) < d[v]`, then a negative-weight cycle reachable from `s` exists. Report it (optionally, identify vertices involved).

**Pseudocode:**

```pseudocode
function Bellman_Ford(Graph G = (V, E, w), source s):
  n = |V|
  m = |E|
  d = array[V] initialized to infinity
  parent = array[V] initialized to null
  d[s] = 0

  // Step 1: Relax edges |V|-1 times
  for i from 1 to n - 1:
    for each edge (u, v) with weight w(u,v) in E:
      if d[u] != infinity and d[u] + w(u,v) < d[v]:
        d[v] = d[u] + w(u,v)
        parent[v] = u

  // Step 2: Check for negative-weight cycles
  for each edge (u, v) with weight w(u,v) in E:
    if d[u] != infinity and d[u] + w(u,v) < d[v]:
      return "Error: Graph contains a negative-weight cycle"

  // No negative cycle detected reachable from s
  // d[v] contains shortest path lengths
  return (d, parent)
```
*Complexity: O(V * E) because we iterate through all E edges, V-1 (+1) times.*

---

### 5. Difference Constraints and Shortest Paths

**Problem Statement:**
Given a system of *m* linear inequalities (difference constraints) on *n* variables `x1, x2, ..., xn`, where each constraint is of the form `xj - xi <= wk`. Find a feasible solution (values for `x1, ..., xn` that satisfy all constraints) or determine that no feasible solution exists.

**Solution (Why):**
This problem can be transformed into a shortest path problem on a graph called the **constraint graph**.

1.  **Construct the Constraint Graph:**
    *   Create a vertex `vi` for each variable `xi`.
    *   Create an additional source vertex `v0`.
    *   For each constraint `xj - xi <= wk`, add a directed edge from `vi` to `vj` with weight `wk`. (Edge `vi -> vj` with weight `wk`).
    *   For each variable `xi`, add an edge from the source `v0` to `vi` with weight 0. (Edge `v0 -> vi` with weight 0). This ensures all variables are reachable from the source and anchors the potential values.

2.  **Solve using Shortest Paths:**
    *   Run the Bellman-Ford algorithm on the constraint graph starting from the source `v0`.
    *   **Case 1: No Negative Cycles:** If Bellman-Ford completes successfully (no negative cycles detected), then a feasible solution exists. The shortest path distances `d[vi]` from `v0` provide a solution: set `xi = d[vi]`.
        *   *Why?* For any edge `vi -> vj` with weight `wk` representing `xj - xi <= wk`, the shortest path property guarantees `d[vj] <= d[vi] + wk`. Substituting `xi = d[vi]` and `xj = d[vj]`, we get `xj <= xi + wk`, which is equivalent to `xj - xi <= wk`. The edges from `v0` ensure finite values.
    *   **Case 2: Negative Cycle Detected:** If Bellman-Ford detects a negative cycle in the constraint graph, then no feasible solution exists for the system of difference constraints.
        *   *Why?* A negative cycle `v1 -> v2 -> ... -> vk -> v1` with total weight `W < 0` corresponds to constraints:
            `x2 - x1 <= w(v1,v2)`
            `x3 - x2 <= w(v2,v3)`
            ...
            `x1 - xk <= w(vk,v1)`
            Summing these inequalities, the `xi` terms cancel out, leaving `0 <= W`. Since `W < 0`, this is a contradiction, proving no solution exists.

**Pseudocode Idea:**

```pseudocode
function Solve_Difference_Constraints(Constraints C, num_vars n):
  // Constraints C: list of tuples (i, j, k) representing xj - xi <= wk

  // 1. Build Constraint Graph G = (V, E, w)
  V = {v0, v1, ..., vn}
  E = empty set
  w = empty map (edge -> weight)

  // Add edges for constraints
  for each constraint (i, j, k) in C:
    Add edge (vi, vj) to E with weight w(vi, vj) = k

  // Add edges from source v0
  for i from 1 to n:
    Add edge (v0, vi) to E with weight w(v0, vi) = 0

  // 2. Run Bellman-Ford from source v0
  result = Bellman_Ford(G, v0)

  // 3. Interpret Result
  if result == "Error: Graph contains a negative-weight cycle":
    return "No feasible solution exists"
  else:
    // Extract shortest path distances d from result
    (d, parent) = result
    solution = array[1..n]
    for i from 1 to n:
      solution[i] = d[vi] // xi = d[vi]
    return solution // A feasible solution
```

This demonstrates a powerful connection between systems of linear inequalities and graph algorithms.