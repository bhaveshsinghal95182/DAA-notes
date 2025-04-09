Okay class, building upon backtracking, let's explore **Branch and Bound (B&B)**. This is another powerful technique, particularly suited for optimization problems (finding the minimum or maximum value), where we want to find the *best* solution without necessarily exploring *all* possibilities.

---

### Overview of Branch and Bound

**Problem Statement:**
How can we systematically search a large state space for an optimal solution (minimum or maximum) more efficiently than exhaustive search or basic backtracking? How can we use estimations about the potential quality of solutions in subproblems to prune the search space aggressively?

**Solution (Why):**
Branch and Bound is a state-space search algorithm designed for optimization problems. Like backtracking, it explores the state-space tree representing possible solutions. However, it adds a crucial element: **bounding**.

1.  **Branching:** This is the process of partitioning the problem into smaller subproblems (generating child nodes from a parent node in the state-space tree). This is similar to how backtracking explores choices.
2.  **Bounding:** For each node (representing a subproblem or partial solution), we calculate a **bound** on the value of the optimal solution that could be obtained by extending this partial solution.
    *   For **minimization** problems, we calculate a **lower bound**: an optimistic estimate of the *minimum* cost possible from this node onwards.
    *   For **maximization** problems, we calculate an **upper bound**: an optimistic estimate of the *maximum* value possible from this node onwards.
3.  **Pruning:** We maintain the value of the best *complete* solution found so far (called the **incumbent**). We can prune (discard) a node (and its entire subtree) if:
    *   Its bound is worse than the incumbent (e.g., for minimization, if `node_lower_bound >= incumbent_cost`, this node cannot lead to a better solution).
    *   The node represents an infeasible solution (e.g., exceeds knapsack capacity).
4.  **Search Strategy:** Unlike backtracking's typical Depth-First Search (DFS), Branch and Bound allows different strategies for selecting the next node to explore from the set of "live" nodes (nodes that have been generated but whose children haven't been fully explored). Common strategies include FIFO, LIFO (similar to backtracking's DFS), and Least Cost (LC).

The effectiveness of Branch and Bound hinges on the quality of the bounding function. A tighter bound (closer to the true optimal value for that subproblem) allows for more aggressive pruning and faster convergence to the optimal solution.

**Pseudocode (General Structure - Minimization):**

```pseudocode
function Branch_And_Bound(Problem P):
  Initialize ActiveNodeList // (e.g., Queue for FIFO, Priority Queue for LC)
  RootNode = Create_Root_Node(P)
  Calculate_Lower_Bound(RootNode)
  Add RootNode to ActiveNodeList

  IncumbentSolution = null
  IncumbentCost = infinity // Best solution cost found so far

  while ActiveNodeList is not empty:
    CurrentNode = Select_And_Remove_Node(ActiveNodeList) // Strategy-dependent

    // Pruning Step 1: Bound Check
    if Get_Lower_Bound(CurrentNode) >= IncumbentCost:
      continue // Prune this node

    // Check if CurrentNode represents a complete solution
    if Is_Complete_Solution(CurrentNode):
      SolutionCost = Calculate_Actual_Cost(CurrentNode)
      if SolutionCost < IncumbentCost:
        IncumbentCost = SolutionCost
        IncumbentSolution = Get_Solution(CurrentNode)
      continue // Reached a leaf, no further branching

    // Branching Step: Generate children nodes (subproblems)
    ChildNodes = Generate_Children(CurrentNode)

    for each Child in ChildNodes:
      // Pruning Step 2: Feasibility Check (if applicable)
      if Is_Feasible(Child):
        Calculate_Lower_Bound(Child)
        // Pruning Step 3: Bound Check before adding
        if Get_Lower_Bound(Child) < IncumbentCost:
          Add Child to ActiveNodeList

  return (IncumbentSolution, IncumbentCost)
```

---

### Search Strategies: LC, FIFO

1.  **LC (Least-Cost) Branch and Bound:**
    *   **Problem:** How to prioritize exploring nodes that seem most likely to lead to the optimal solution quickly?
    *   **Solution (Why):** Always select the live node with the *best* bound (lowest lower bound for minimization, highest upper bound for maximization) to explore next. This is a "best-first" search strategy. It uses a **Priority Queue** to store the live nodes, ordered by their bounds. The idea is that nodes with better optimistic bounds are more promising and should be investigated first.
    *   **Pseudocode Snippet (Selection):**
        ```pseudocode
        // ActiveNodeList is a Min-Priority Queue (for minimization)
        // keyed by the node's lower bound.
        CurrentNode = Extract_Min(ActiveNodeList)
        ```

2.  **FIFO (First-In, First-Out) Branch and Bound:**
    *   **Problem:** A simpler strategy for selecting the next node.
    *   **Solution (Why):** Select the live node that was generated earliest. This effectively performs a **Breadth-First Search (BFS)** on the state-space tree. It uses a standard **Queue** to store live nodes. While simpler to implement, it might explore many unpromising nodes before reaching the optimal area of the search space compared to LC search.
    *   **Pseudocode Snippet (Selection):**
        ```pseudocode
        // ActiveNodeList is a standard FIFO Queue.
        CurrentNode = Dequeue(ActiveNodeList)
        ```

*(LIFO Branch and Bound, using a Stack, behaves like backtracking's DFS but incorporates bounding for pruning.)*

---

### Bounding

**Problem Statement:**
How do we calculate an optimistic estimate (bound) for a node in the state-space tree that is both reasonably accurate (to allow effective pruning) and efficient to compute?

**Solution (Why):**
The bounding function is problem-specific. It needs to calculate a value that is guaranteed to be better than or equal to (for maximization) or worse than or equal to (for minimization) the actual best solution achievable from that node.

*   **Example (Minimization):** If a node represents a partial tour in TSP with cost `C`, a lower bound could be `C +` (estimated cost of optimally completing the tour from the current city, visiting all remaining cities). This estimate might be the sum of the cheapest edges out of each unvisited city.
*   **Example (Maximization - 0/1 Knapsack):** If a node represents having considered items `1..k` with current value `V` and weight `W'`, an upper bound could be `V +` (maximum possible value obtainable from the remaining items `k+1..n` within the remaining capacity `W - W'`). This is often calculated by solving the *fractional* knapsack problem for the remaining items, which provides an optimistic (upper) bound for the 0/1 version.

A good bound allows pruning nodes earlier, significantly reducing the search space. There's often a trade-off: more complex bounds might be tighter (better pruning) but take longer to compute.

---

### 0/1 Knapsack Problem using Branch and Bound

**Problem Statement:**
Given *n* items with weights `w[i]` and values `v[i]`, and capacity `W`, find the subset of items maximizing total value without exceeding `W`. (Using B&B, typically LC search).

**Solution (Why B&B?):**
This is a maximization problem suitable for B&B. We can improve upon backtracking by using an upper bound to prune branches that cannot possibly yield a better total value than the best complete solution found so far.

*   **State Representation:** A node `u` can represent a decision point for item `k`. It stores `current_weight`, `current_value`, and the `level` `k`.
*   **Branching:** From a node at level `k` (considering item `k`), create two children:
    1.  Left child: Excludes item `k`. `(current_weight, current_value, k+1)`
    2.  Right child: Includes item `k`. `(current_weight + w[k], current_value + v[k], k+1)`. Only generate this child if `current_weight + w[k] <= W`.
*   **Bounding (Upper Bound):** For a node `u` at level `k` with `current_value` and `current_weight`, calculate an upper bound `ub(u)` on the achievable value. A common method:
    `ub(u) = current_value + Fractional_Knapsack_Value(remaining items k..n-1, remaining capacity W - current_weight)`
    The fractional knapsack solution on the remaining items gives an optimistic (upper) bound. Pre-sorting items by value/weight ratio helps here.
*   **Pruning:** Maintain `max_profit` (incumbent). Prune node `u` if:
    *   `ub(u) <= max_profit` (Cannot possibly lead to a better solution).
    *   The node itself is infeasible (e.g., the 'include' branch resulted in `current_weight > W`).
*   **Search Strategy:** Typically **LC Branch and Bound** (using a Max-Priority Queue ordered by the upper bound). Nodes with higher potential profit are explored first.

**Pseudocode (LC Branch and Bound for 0/1 Knapsack):**

```pseudocode
function Knapsack_BnB(W, weights, values, n):
  // Assume items are pre-sorted by value/weight ratio descending.

  PQ = MaxPriorityQueue() // Stores nodes, prioritized by upper bound
  max_profit = 0 // Incumbent solution value

  // Define a function to calculate the upper bound for a node
  function Calculate_Upper_Bond(level, current_weight, current_value):
    bound = current_value
    remaining_capacity = W - current_weight
    k = level
    // Add whole items greedily (based on pre-sorted order)
    while k < n and weights[k] <= remaining_capacity:
      remaining_capacity -= weights[k]
      bound += values[k]
      k += 1
    // Add fraction of the next item if capacity remains
    if k < n and remaining_capacity > 0:
      bound += (remaining_capacity / weights[k]) * values[k]
    return bound

  // Root node: represents starting state before considering item 0
  root_weight = 0
  root_value = 0
  root_level = 0
  root_ub = Calculate_Upper_Bond(root_level, root_weight, root_value)
  Add (root_ub, root_level, root_weight, root_value) to PQ

  while PQ is not empty:
    ub, level, weight, value = Extract_Max(PQ)

    // Pruning check: If bound is not better than max_profit found so far
    if ub <= max_profit:
      continue

    // Base case check: If it's a potential complete solution (or deeper)
    // Technically, leaves are at level n
    # if level == n: # We often update max_profit when generating leaves
    #   max_profit = max(max_profit, value) # Update if better leaf found
    #   continue # Don't branch further from leaves

    # Branching: Consider item at 'level'

    # 1. Child: Include item 'level' (if feasible)
    if level < n and weight + weights[level] <= W:
      include_level = level + 1
      include_weight = weight + weights[level]
      include_value = value + values[level]
      # Update max_profit if this node itself represents a better complete solution
      # (Important: Some formulations check/update incumbent here)
      max_profit = max(max_profit, include_value) # Update if this selection is better
      include_ub = Calculate_Upper_Bond(include_level, include_weight, include_value)
      # Pruning check before adding
      if include_ub > max_profit:
        Add (include_ub, include_level, include_weight, include_value) to PQ

    # 2. Child: Exclude item 'level'
    if level < n:
      exclude_level = level + 1
      exclude_weight = weight
      exclude_value = value
      exclude_ub = Calculate_Upper_Bond(exclude_level, exclude_weight, exclude_value)
       # Pruning check before adding
      if exclude_ub > max_profit:
        Add (exclude_ub, exclude_level, exclude_weight, exclude_value) to PQ

  return max_profit
```
*(Note: Implementation details can vary, e.g., when exactly the incumbent `max_profit` is updated. Sorting items first is crucial for the bound calculation efficiency).*

---

### Traveling Salesman Problem (TSP) using Branch and Bound

**Problem Statement:**
Given a set of *n* cities and the distances `cost(i, j)` between each pair of cities, find a tour (a simple cycle visiting each city exactly once) with the minimum total distance.

**Solution (Why B&B?):**
TSP has a factorial search space (`(n-1)!` possible tours), making exhaustive search infeasible for non-trivial *n*. B&B provides a way to find the optimal tour by exploring promising partial tours and pruning those whose *optimistic* completion cost (lower bound) already exceeds the cost of the best full tour found so far.

*   **State Representation:** A node `u` can represent a partial path taken so far. It stores:
    *   `path`: The sequence of cities visited.
    *   `last_city`: The last city visited in the `path`.
    *   `visited_mask`: A bitmask or set indicating visited cities.
    *   `current_cost`: The cost of the `path` so far.
    *   `lower_bound`: An optimistic estimate of the minimum total tour cost achievable by extending this `path`.
*   **Branching:** From a node representing `path` ending at `last_city`, generate child nodes by choosing the *next* unvisited city `j` to visit. The child node represents `path + (j)`.
*   **Bounding (Lower Bound):** Calculating a good lower bound is critical. Common methods:
    1.  **Simple Bound:** `current_cost + sum_of_minimum_outgoing_edges` from `last_city` and all other unvisited cities (ensuring the path can return to the start). E.g., `current_cost + cost(last_city, nearest_unvisited) + sum(cost(k, nearest_unvisited_or_start))` for remaining unvisited `k`.
    2.  **Reduced Cost Matrix:** Start with the cost matrix. Reduce it by subtracting the minimum value from each row and then each column. The sum of subtracted values is an initial lower bound. For a child node corresponding to adding edge `(i, j)`, the bound is `parent_bound + reduced_cost(i, j) + further_reduction_in_child_matrix` (where the child matrix is derived by setting row `i`, column `j`, and edge `(j, i)` to infinity and reducing again). This is more complex but often yields tighter bounds.
*   **Pruning:** Maintain `min_tour_cost` (incumbent). Prune node `u` if `lower_bound(u) >= min_tour_cost`.
*   **Search Strategy:** Typically **LC Branch and Bound** using a Min-Priority Queue ordered by the lower bound. Nodes representing partial paths with lower potential total cost are explored first.

**Pseudocode (High-Level LC Branch and Bound for TSP):**

```pseudocode
function TSP_BnB(CostMatrix, n):
  PQ = MinPriorityQueue() // Stores nodes (lower_bound, path, last_city, visited_mask, current_cost)
  min_tour_cost = infinity // Incumbent cost
  best_tour = null

  // Define function to calculate lower bound (e.g., using reduced matrix or simpler estimate)
  function Calculate_Lower_Bound(path, last_city, visited_mask, current_cost):
      // ... implementation of bounding logic ...
      return calculated_lower_bound

  // Start node: path contains only the starting city (e.g., city 0)
  start_path = [0]
  start_mask = (1 << 0)
  start_cost = 0
  start_lb = Calculate_Lower_Bound(start_path, 0, start_mask, start_cost)
  Add (start_lb, start_path, 0, start_mask, start_cost) to PQ

  while PQ is not empty:
    lb, path, last, mask, cost = Extract_Min(PQ)

    // Pruning check
    if lb >= min_tour_cost:
      continue

    // Check if path represents a complete tour (all n cities visited)
    if len(path) == n:
      # Calculate final tour cost by adding return edge to start
      final_cost = cost + CostMatrix[last][path[0]]
      if final_cost < min_tour_cost:
        min_tour_cost = final_cost
        best_tour = path + [path[0]] # Store the complete tour
      continue # Reached a potential solution

    // Branching: Explore neighbors
    for next_city from 0 to n-1:
      // Check if next_city is not visited yet
      if not (mask & (1 << next_city)):
        # Create child node state
        new_path = path + [next_city]
        new_mask = mask | (1 << next_city)
        new_cost = cost + CostMatrix[last][next_city]

        # Calculate lower bound for the child
        new_lb = Calculate_Lower_Bound(new_path, next_city, new_mask, new_cost)

        # Pruning check before adding
        if new_lb < min_tour_cost:
          Add (new_lb, new_path, next_city, new_mask, new_cost) to PQ

  return (best_tour, min_tour_cost)
```
*(Note: TSP B&B implementations can be quite involved, especially with sophisticated bounds like the reduced cost matrix approach.)*

Branch and Bound is a fundamental technique for solving NP-hard optimization problems by intelligently pruning the search space using bounds, often providing optimal solutions for moderate-sized instances where exhaustive search fails.