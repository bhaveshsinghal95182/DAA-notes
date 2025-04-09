Alright class, today we shift our focus to another important algorithmic paradigm: **Greedy Algorithms**. While Dynamic Programming often explores many possibilities to guarantee optimality, Greedy algorithms make the choice that seems best *at the moment* and hope that these locally optimal choices lead to a globally optimal solution.

---

### 1. Elements of Greedy Algorithms

**Problem Statement:**
How can we solve optimization problems by making a sequence of choices that look best at each step, without revisiting previous choices? When does this strategy actually lead to a globally optimal solution?

**Solution (Why):**
Greedy algorithms work step-by-step, making a locally optimal choice at each stage with the hope of finding a global optimum. They don't reconsider choices once made. For a greedy algorithm to successfully find the global optimum, the problem must typically exhibit two key properties:

1.  **Greedy Choice Property:** A globally optimal solution can be arrived at by making a locally optimal (greedy) choice. In other words, the choice made at the current step, based on local information, should be consistent with *some* globally optimal solution. We don't need to know *which* optimal solution yet, just that the current choice doesn't preclude us from reaching one.
2.  **Optimal Substructure:** An optimal solution to the problem contains within it optimal solutions to subproblems. After making a greedy choice, we are left with a smaller subproblem. The optimal substructure property means that if we combine the greedy choice with an optimal solution to the resulting subproblem, we get an optimal solution for the original problem. (Note: This property is shared with Dynamic Programming, but the way it's used differs. DP typically explores multiple subproblems; Greedy commits to the subproblem resulting from the single greedy choice).

The main advantage of greedy algorithms is their simplicity and often better time complexity compared to DP. However, they don't work for all optimization problems (e.g., the 0/1 Knapsack problem requires DP, while the Fractional Knapsack problem can be solved greedily). Proving the correctness of a greedy algorithm often involves showing that the greedy choice property holds (e.g., using an exchange argument: assume there's an optimal solution that doesn't use the first greedy choice, then show you can modify it to use the greedy choice and get a solution that's at least as good).

**Pseudocode (General Greedy Structure):**

```pseudocode
function Greedy_Algorithm(InputSet):
  Solution = empty_set // Initialize the solution

  while InputSet is not empty:
    // Make the greedy choice
    x = Select_Greedy_Choice(InputSet)

    // Remove x from the input set (or mark as considered)
    InputSet = InputSet - {x}

    // Check if adding x to the solution is feasible/valid
    if Is_Feasible(Solution + {x}):
      // Add the choice to the solution
      Solution = Solution + {x}

  return Solution

// Note: The specific implementation of Select_Greedy_Choice and Is_Feasible
// depends heavily on the particular problem.
```

---

### 2. Activity-Selection Problem

**Problem Statement:**
Given a set `S = {a1, a2, ..., an}` of *n* activities that wish to use a common resource (e.g., a lecture hall), where each activity `ai` has a start time `s[i]` and a finish time `f[i]`. Two activities `ai` and `aj` are compatible if their intervals `[s[i], f[i))` and `[s[j], f[j))` do not overlap (assume `s[i] < f[i]`). Find a maximum-size subset of mutually compatible activities.

**Solution (Why):**
The greedy strategy is to repeatedly choose the activity that finishes earliest among those compatible with the already selected activities.

*   **Greedy Choice:** Select the activity `ak` with the minimum finish time `f[k]`.
*   **Why it works (Greedy Choice Property & Optimal Substructure):**
    *   Let `ak` be the activity with the earliest finish time. We claim there exists an optimal solution that includes `ak`. Suppose `A` is an optimal solution, and `aj` is the activity in `A` with the earliest finish time. If `aj = ak`, we are done. If `aj != ak`, then `f[k] <= f[j]` (by definition of `ak`). Since activities in `A` are compatible, all activities in `A` (except possibly `aj`) must start after `f[j]`. Since `f[k] <= f[j]`, they must also start after `f[k]`. Thus, we can replace `aj` with `ak` in `A` to get a new solution `A'` which is also compatible and has the same size as `A`. Therefore, `A'` is also optimal, and it includes the greedy choice `ak`.
    *   After choosing `ak`, the problem reduces to finding the maximum-size set of mutually compatible activities from the subset `S'` of activities that start after `ak` finishes (`s[i] >= f[k]`). The optimal substructure property holds because if we combine `ak` with an optimal solution for the subproblem `S'`, we get an optimal solution for the original problem.

**Pseudocode:**

```pseudocode
function Greedy_Activity_Selector(s, f, n):
  // s: array of start times
  // f: array of finish times
  // n: number of activities
  // Assume activities are pre-sorted by finish times (f[1] <= f[2] <= ... <= f[n])

  Selected_Activities = {a1} // Always select the first activity (earliest finish)
  last_finish_time = f[1]

  for i from 2 to n:
    // If activity ai starts after the last selected activity finishes
    if s[i] >= last_finish_time:
      // Select activity ai
      Selected_Activities = Selected_Activities + {ai}
      last_finish_time = f[i]

  return Selected_Activities
```
*Pre-sorting by finish time takes O(n log n). The selection process takes O(n). Total time: O(n log n).*

---

### 3. Optimal Merge Pattern & Huffman Codes

**Problem Statement:**
*   **(Optimal Merge Pattern):** Given *n* sorted files (or lists) of lengths `l1, l2, ..., ln`, find an optimal way to merge them into a single sorted file. The cost of merging two files of size *x* and *y* is `x + y`. We want to minimize the total merge cost.
*   **(Huffman Codes):** Given a set of characters and their frequencies, find a prefix code (no codeword is a prefix of another) with the minimum expected codeword length (which minimizes the total length of the encoded text).

**Solution (Why):**
These two problems are structurally similar. The optimal merge pattern problem is equivalent to constructing an optimal binary tree where the files are leaves, and the cost is the sum of weighted path lengths (file size * depth). Huffman coding directly addresses constructing such a tree where leaf weights are character frequencies.

The greedy strategy for Huffman coding (and thus optimal merging) is:
*   **Greedy Choice:** Repeatedly merge the two nodes (initially leaves representing characters/files) with the lowest frequencies (or file lengths/merge costs incurred so far). Create a new internal node whose frequency is the sum of the frequencies of its children.
*   **Why it works:** The core idea is to keep the lowest frequency characters (smallest files) deepest in the tree, as they contribute the least to the total weighted path length (total merge cost) at each level. By always merging the two smallest current nodes, we ensure that the elements contributing the least to the cost are combined first, effectively pushing them deeper. A formal proof involves showing that swapping a deeper, higher-frequency node with a shallower, lower-frequency node would improve or maintain the total cost, supporting the greedy choice property. The optimal substructure follows because if the merging of the two lowest frequency nodes `x` and `y` into `z` is part of an optimal tree, then the tree formed by replacing `x`, `y`, and `z` with a single leaf `z` (with frequency `freq[x] + freq[y]`) must be optimal for the reduced problem.

**Pseudocode (Huffman Tree Construction):**

```pseudocode
function Huffman(C):
  // C: set of characters with frequencies {c1: f1, c2: f2, ...}
  n = |C|
  // Use a min-priority queue, keyed by frequency
  Q = PriorityQueue(C) // Initialize queue with characters as leaf nodes

  for i from 1 to n - 1:
    // Allocate a new internal node
    z = Allocate_Node()

    // Extract the two nodes with the lowest frequencies
    x = Extract_Min(Q)
    y = Extract_Min(Q)

    // Make x and y children of z
    z.left = x
    z.right = y
    // Set z's frequency as the sum
    z.frequency = x.frequency + y.frequency

    // Insert the new internal node z back into the queue
    Insert(Q, z)

  // The remaining node in the queue is the root of the Huffman tree
  return Extract_Min(Q)

// Note: To get the codes, traverse the tree: 0 for left branch, 1 for right branch.
// The priority queue operations typically take O(log n) time. The loop runs n-1 times.
// Total time: O(n log n).
```

---

### 4. Job Scheduling with Deadlines (Task Scheduling)

**Problem Statement:**
Given a set of *n* jobs, each with a deadline `d[i]` and a profit `p[i]` (if completed by its deadline). Each job takes unit time to complete. We want to find a schedule (a subset of jobs and a time slot for each) that maximizes the total profit. Only one job can be processed at a time.

**Solution (Why):**
The greedy strategy is to consider jobs in decreasing order of profit. For each job, schedule it in the latest possible time slot `t` such that `t <= d[i]` and the slot is currently free.

*   **Greedy Choice:** Pick the highest profit job available.
*   **Placement Strategy:** Assign it to the latest possible valid time slot.
*   **Why it works:**
    *   **Greedy Choice Property:** Prioritizing higher profit jobs makes intuitive sense. If we have a time slot conflict between a high-profit job and a low-profit job, choosing the high-profit job first is more likely to lead to an optimal solution. An exchange argument can formalize this: If an optimal solution `S` doesn't pick the highest profit job `j1` but picks a lower profit job `j2` instead, and there was a valid slot for `j1`, we can potentially swap `j2` for `j1` (if `j1` fits) to get an equal or better solution.
    *   **Placement & Optimal Substructure:** Placing the job as late as possible keeps earlier slots free for jobs with potentially earlier deadlines. This maximizes the chances of fitting other jobs (potentially high-profit ones considered later). If we make the greedy choice (select highest profit job `j`) and schedule it optimally (latest possible slot), the remaining problem is to schedule the rest of the jobs in the remaining slots, respecting their deadlines. An optimal solution to this subproblem combined with the choice for `j` yields an overall optimal solution.

**Pseudocode:**

```pseudocode
function Job_Scheduling_With_Deadline(jobs, n, max_deadline):
  // jobs: list of jobs, each job = (id, deadline, profit)
  // n: number of jobs
  // max_deadline: the maximum deadline among all jobs

  // Sort jobs in descending order of profit
  Sort jobs by profit (descending)

  // Keep track of used time slots (e.g., using a boolean array or disjoint set)
  time_slots_used = array[1..max_deadline] initialized to false
  Result_Schedule = array[1..max_deadline] initialized to empty // Stores job IDs

  Total_Profit = 0
  Job_Count = 0

  for i from 1 to n: // Iterate through sorted jobs
    job = jobs[i]
    // Find the latest possible free slot for this job, up to its deadline
    // Search backwards from min(max_deadline, job.deadline) down to 1
    for t from min(max_deadline, job.deadline) down to 1:
      if time_slots_used[t] == false:
        // Found a free slot
        time_slots_used[t] = true
        Result_Schedule[t] = job.id
        Total_Profit = Total_Profit + job.profit
        Job_Count = Job_Count + 1
        break // Slot found for this job, move to the next job

  // Result_Schedule contains the IDs of scheduled jobs in their time slots
  // Total_Profit is the maximum profit
  return (Total_Profit, Result_Schedule)

// Using Disjoint Set Union (DSU) can optimize finding the latest free slot.
// Sorting takes O(n log n). With DSU, the scheduling loop takes nearly O(n * alpha(n)),
// where alpha is the very slow-growing inverse Ackermann function.
// Without DSU (simple linear scan for slot), it can be O(n * max_deadline).
// Overall complexity is typically dominated by sorting: O(n log n).
```

---

### 5. Knapsack Problem (Fractional)

**Problem Statement:**
Given *n* items, where item *i* has weight `w[i]` and value `v[i]`, and a knapsack with maximum capacity `W`. Unlike the 0/1 version, we *can* take fractions of items. Find the amount `x[i]` (where `0 <= x[i] <= 1`) of each item *i* to include in the knapsack such that the total weight `sum(x[i] * w[i]) <= W` and the total value `sum(x[i] * v[i])` is maximized.

**Solution (Why):**
The greedy strategy is to compute the value-per-unit-weight (`v[i] / w[i]`) for each item and take as much as possible of the items with the highest value density first.

*   **Greedy Choice:** Choose the item with the maximum value per unit weight (`v[i] / w[i]`).
*   **Why it works:** To maximize the total value within a fixed weight capacity, it's always best to prioritize the items that give the most "bang for the buck" (value per weight). By filling the knapsack with the highest density items first, we ensure that every unit of weight capacity used contributes the maximum possible value at that point. If we were to replace a fraction of a high-density item `i` with an equal weight fraction of a lower-density item `j`, the total value would decrease. This holds true until the knapsack is full. Since we can take fractions, we can always completely fill the knapsack (unless the total weight of all items is less than `W`) by taking a fraction of the first item that doesn't fit entirely.

**Pseudocode:**

```pseudocode
function Fractional_Knapsack(W, weights, values, n):
  // W: Knapsack capacity
  // weights: array of item weights
  // values: array of item values
  // n: number of items

  // 1. Calculate value density for each item
  items = []
  for i from 0 to n-1:
    density = values[i] / weights[i]
    items.append((density, weights[i], values[i])) // Store as (density, weight, value)

  // 2. Sort items by density in descending order
  Sort items by density (descending)

  Total_Value = 0
  Current_Weight = 0
  fractions = array[0..n-1] initialized to 0.0

  // 3. Fill the knapsack greedily
  for i from 0 to n-1:
    density, weight, value = items[i]

    if Current_Weight + weight <= W:
      // Take the whole item
      fractions[i] = 1.0 // Store the original index if needed
      Current_Weight = Current_Weight + weight
      Total_Value = Total_Value + value
    else:
      // Take a fraction of the item to fill the remaining capacity
      remaining_capacity = W - Current_Weight
      fraction_to_take = remaining_capacity / weight
      fractions[i] = fraction_to_take // Store the original index if needed
      Total_Value = Total_Value + fraction_to_take * value
      Current_Weight = W // Knapsack is now full
      break // Exit loop

  return Total_Value // Or return the fractions array if needed
```
*Calculating densities takes O(n). Sorting takes O(n log n). Filling takes O(n). Total time: O(n log n).*

---

### 6. Minimum Spanning Tree (MST) - Kruskal's Algorithm

**Problem Statement:**
Given a connected, undirected, weighted graph `G = (V, E, w)`, find a spanning tree `T` (a subgraph that connects all vertices and is acyclic) such that the sum of the weights of the edges in `T` is minimized. This is called a Minimum Spanning Tree (MST).

**Solution (Why):**
Kruskal's algorithm builds the MST by iteratively adding the next lightest edge that does not form a cycle with the edges already selected.

*   **Greedy Choice:** Pick the edge `(u, v)` from `E` with the minimum weight such that adding `(u, v)` to the set of already selected edges does not create a cycle.
*   **Why it works:** Kruskal's algorithm relies on the **Cut Property** of MSTs: For any cut (a partition of the vertices `V` into two disjoint sets `S` and `V-S`), if an edge `(u, v)` has the minimum weight among all edges crossing the cut (one endpoint in `S`, one in `V-S`), then this edge `(u, v)` belongs to *some* MST of the graph.
    When Kruskal's considers an edge `(u, v)`, if `u` and `v` are already connected by previously selected edges, adding `(u, v)` would form a cycle, so it's skipped. If `u` and `v` are in different connected components (forming a cut between the component containing `u` and the rest), and `(u, v)` is the lightest edge currently being considered, it must be the lightest edge crossing the cut separating `u`'s component from `v`'s component (among the *remaining* edges). By a more detailed argument related to the cut property, adding this edge is a "safe" move that guarantees it can be part of an MST. The algorithm maintains a forest of trees, merging them using the lightest possible safe edge until a single MST covering all vertices is formed. A Disjoint Set Union (DSU) data structure is typically used to efficiently detect cycles (check if `u` and `v` are already in the same component) and merge components.

**Pseudocode:**

```pseudocode
function Kruskal_MST(Graph G = (V, E, w)):
  // V: set of vertices
  // E: set of edges with weights w
  // n: number of vertices |V|
  // m: number of edges |E|

  MST_Edges = empty_set // Stores edges of the resulting MST
  Total_Weight = 0

  // Create a disjoint set data structure for all vertices
  DisjointSet ds = Create_DisjointSet(V)
  for each vertex v in V:
    Make_Set(ds, v)

  // Sort all edges in E by weight in non-decreasing order
  Sort E by weight w (non-decreasing)

  // Iterate through sorted edges
  for each edge (u, v) with weight w(u,v) in sorted E:
    // Check if adding the edge creates a cycle
    // i.e., if u and v are already in the same component
    if Find_Set(ds, u) != Find_Set(ds, v):
      // Add the edge to the MST
      MST_Edges = MST_Edges + {(u, v)}
      Total_Weight = Total_Weight + w(u,v)
      // Merge the components of u and v
      Union(ds, u, v)

      // Optional: Stop if |MST_Edges| == n - 1

  return (MST_Edges, Total_Weight)

// Complexity: Sorting edges takes O(m log m) or O(m log n).
// DSU operations (Make_Set, Find_Set, Union) take nearly constant time on average
// with path compression and union by rank/size, totaling roughly O(m * alpha(n)).
// Overall complexity is dominated by sorting: O(m log n) or O(m log m).
```

---

### 7. Minimum Spanning Tree (MST) - Prim's Algorithm

**Problem Statement:**
Given a connected, undirected, weighted graph `G = (V, E, w)`, find a Minimum Spanning Tree (MST).

**Solution (Why):**
Prim's algorithm builds the MST incrementally, starting from an arbitrary vertex. It maintains a set `S` of vertices already included in the MST and repeatedly adds the minimum-weight edge connecting a vertex in `S` to a vertex outside `S`.

*   **Greedy Choice:** At each step, choose the minimum-weight edge `(u, v)` such that `u` is in the current tree (`S`) and `v` is not (`v` in `V-S`). Add `v` to the tree.
*   **Why it works:** Prim's algorithm also relies on the **Cut Property**. At each step, the algorithm effectively considers the cut `(S, V-S)`. The greedy choice selects the lightest edge crossing this cut. By the Cut Property, this edge must belong to some MST. Since the algorithm only adds safe edges guaranteed to be in an MST, and continues until all vertices are connected, the resulting tree is indeed an MST. A min-priority queue is typically used to efficiently find the minimum-weight edge connecting a vertex in `S` to one in `V-S`. The priority queue stores vertices in `V-S`, keyed by the minimum weight of an edge connecting them to *any* vertex currently in `S`.

**Pseudocode (using Priority Queue):**

```pseudocode
function Prim_MST(Graph G = (V, E, w), start_vertex r):
  // V: set of vertices
  // E: set of edges with weights w
  // n: number of vertices |V|
  // r: arbitrary start vertex

  // key[v]: minimum weight of an edge connecting v to a vertex in the MST-so-far
  // parent[v]: parent of v in the MST
  key = array[V] initialized to infinity
  parent = array[V] initialized to null
  in_MST = array[V] initialized to false // Tracks vertices added to MST

  key[r] = 0 // Start vertex has 0 cost to connect to itself

  // Min-priority queue stores vertices not yet in MST, keyed by key[v]
  PQ = PriorityQueue(V, keys=key) // Initialize with all vertices

  MST_Edges = empty_set
  Total_Weight = 0

  while PQ is not empty:
    u = Extract_Min(PQ) // Get vertex u not in MST with smallest key value

    in_MST[u] = true
    if parent[u] is not null: // Add edge (parent[u], u) to MST (except for start node)
        edge_weight = w(parent[u], u) // Need graph access for weight
        MST_Edges = MST_Edges + {(parent[u], u)}
        Total_Weight = Total_Weight + edge_weight


    // Update keys of neighbors v of u that are still in the priority queue
    for each neighbor v of u:
      if v is in PQ and w(u, v) < key[v]:
        parent[v] = u
        key[v] = w(u, v)
        Decrease_Key(PQ, v, key[v]) // Update v's priority in the queue

  return (MST_Edges, Total_Weight)

// Complexity: With a binary heap PQ: O(m log n).
// With a Fibonacci heap PQ: O(m + n log n).
// (m = |E|, n = |V|)
```

---

### 8. Single Source Shortest Paths - Dijkstra's Algorithm

**Problem Statement:**
Given a weighted directed graph `G = (V, E, w)` with non-negative edge weights (`w(u, v) >= 0` for all edges) and a source vertex `s`, find the shortest path lengths from `s` to all other vertices `v` in `V`.

**Solution (Why):**
Dijkstra's algorithm finds the shortest paths from a source `s` iteratively. It maintains a set `S` of vertices for which the shortest path distance from `s` is known. Initially, `S = {}`. It repeatedly selects the vertex `u` in `V-S` with the minimum *tentative* shortest path distance `d[u]` from `s`, adds `u` to `S`, and updates the tentative distances of `u`'s neighbors (this update step is called "relaxation").

*   **Greedy Choice:** Select the vertex `u` not yet finalized (`u` in `V-S`) that has the smallest estimated shortest path distance `d[u]` from the source `s`.
*   **Why it works:** The correctness relies heavily on the non-negative edge weights. When Dijkstra selects a vertex `u` to add to `S`, its current tentative distance `d[u]` is guaranteed to be the true shortest path distance from `s` to `u`. Why? Suppose there was a shorter path `P` to `u`. Since `s` is in `S` and `u` is not, path `P` must leave `S` at some point. Let `y` be the first vertex on `P` that is not in `S`, and let `x` be the vertex just before `y` on `P` (so `x` is in `S`). The path `P` looks like `s -> ... -> x -> y -> ... -> u`. Since edge weights are non-negative, the distance `d(s, y)` must be less than or equal to the distance `d(s, u)` along path `P`. Also, since `x` was finalized *before* `u`, and Dijkstra picked `u` as having the minimum distance among vertices in `V-S`, we must have `d[u] <= d[y]`. But if path `P` is shorter than the path found by Dijkstra leading to `d[u]`, then `d(s,y)` (part of `P`) would have to be less than `d[u]`. However, `d[y]` (the tentative distance to `y` when `u` was selected) must be at least `d[u]` (because `u` was chosen as minimum). Since edge weights are non-negative, `d(s,y)` (the true shortest path to `y`) cannot be less than `d[y]`. This chain implies `d[u] <= d[y] <= d(s,y) <= d(s,u via P)`. If `P` was shorter than the path giving `d[u]`, this leads to `d[u] < d[u]`, a contradiction. Therefore, `d[u]` must be the shortest path distance when `u` is selected. The algorithm uses a min-priority queue to efficiently find the vertex `u` with the minimum `d[u]`.

**Pseudocode (using Priority Queue):**

```pseudocode
function Dijkstra(Graph G = (V, E, w), source s):
  // V: set of vertices
  // E: set of edges with non-negative weights w
  // s: source vertex

  // d[v]: estimated shortest distance from s to v
  // parent[v]: predecessor of v in the shortest path from s
  d = array[V] initialized to infinity
  parent = array[V] initialized to null

  d[s] = 0 // Distance from source to itself is 0

  // Min-priority queue stores vertices not yet finalized, keyed by d[v]
  PQ = PriorityQueue(V, keys=d) // Initialize with all vertices

  Finalized_Set S = empty_set

  while PQ is not empty:
    u = Extract_Min(PQ) // Get vertex u not in S with smallest d[u]
    Add u to S // Mark u as finalized

    // Relax edges outgoing from u
    for each neighbor v of u:
      // If a shorter path to v is found through u
      if d[u] + w(u, v) < d[v]:
        // Update distance and parent
        d[v] = d[u] + w(u, v)
        parent[v] = u
        // Update v's priority in the queue
        Decrease_Key(PQ, v, d[v])

  // d[v] now contains the shortest path length from s to v
  // parent[v] can be used to reconstruct the shortest paths
  return (d, parent)

// Complexity: Similar to Prim's. With a binary heap PQ: O(m log n).
// With a Fibonacci heap PQ: O(m + n log n).
// (m = |E|, n = |V|)
```

---

Remember, the key challenge with greedy algorithms is proving their correctness. Always ask yourself: "Does the locally optimal choice always lead to (or is it part of) a globally optimal solution?" If you can prove that, the greedy approach is often very efficient.