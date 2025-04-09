Okay class, let's move on to another algorithmic technique called **Backtracking**. It's particularly useful for problems where you need to find *all* possible solutions or *a* solution that satisfies certain constraints, by exploring possibilities incrementally.

---

### 1. Backtracking: Overview

**Problem Statement:**
How can we systematically search for solutions to a problem among a potentially vast set of possibilities, especially when the problem involves constraints that must be satisfied? How do we avoid exhaustively checking every single possibility if we realize early on that a particular path of choices cannot lead to a valid solution?

**Solution (Why):**
Backtracking is an algorithmic technique for solving problems recursively by trying to build a solution incrementally, one piece at a time, removing those solutions that fail to satisfy the constraints of the problem at any point in time (this removal is the "backtracking").

Think of it as exploring a **state-space tree**. Each node in this tree represents a partial solution or a state reached after making a sequence of choices. Backtracking performs a depth-first search (DFS) on this state-space tree.

1.  **Start** at the root node (empty or initial state).
2.  **Explore:** Move from the current node to one of its children, representing the next incremental choice in building a solution.
3.  **Check Constraints (Pruning):** At each node, check if the current partial solution is still viable (doesn't violate any constraints) and if it *could* potentially lead to a complete valid solution.
    *   If the current path is **not promising** (violates constraints or cannot possibly lead to a valid/optimal solution), **prune** this branch of the tree. This means we **backtrack**: discard the last choice made and return to the parent node to explore other alternatives (other children).
    *   If the current path **is promising**, continue exploring deeper down that branch.
4.  **Goal:** If a node represents a complete and valid solution, record it. If we need only one solution, we can stop. If we need all solutions, we record it and continue backtracking to find others. If we are looking for an optimal solution (like in Knapsack), we update our "best solution found so far" and continue searching.

The key advantage over simple brute-force recursion is the **pruning** step. By checking constraints early and abandoning paths that are clearly incorrect or suboptimal, backtracking can significantly reduce the search space.

**Pseudocode (General Template):**

```pseudocode
function Backtrack(current_state):
  // Check if the current state represents a valid, complete solution
  if Is_Solution(current_state):
    Process_Solution(current_state) // Record it, update best, etc.
    return // Or return true if only one solution is needed

  // Iterate through all possible next choices/moves from the current state
  for each choice in Generate_Possible_Choices(current_state):

    // Check if the choice is promising (obeys constraints, could lead to solution)
    if Is_Promising(current_state, choice):

      // Apply the choice to move to the next state
      next_state = Apply_Choice(current_state, choice)

      // Recursively call Backtrack for the next state
      Backtrack(next_state)

      // *** Backtrack ***
      // Remove the choice (undo the changes made by Apply_Choice)
      // This is crucial to explore other branches from current_state
      Undo_Choice(current_state, choice) // Or simply let local variables go out of scope

// Initial call would be something like:
// initial_state = Get_Initial_State()
// Backtrack(initial_state)
```

---

### 2. 8-Queens Problem

**Problem Statement:**
Place *n* (typically 8) non-attacking queens on an *n x n* chessboard. This means no two queens should be in the same row, same column, or on the same diagonal. Find one or all possible arrangements.

**Solution (Why Backtracking?):**
This is a classic constraint satisfaction problem. The state space is the set of all possible placements of queens. Brute-forcing placements would be enormous (`(n*n) choose n`). Backtracking is ideal here.

We can build the solution incrementally, row by row (or column by column).
*   **State:** A partial solution can be represented by the column positions of queens placed in the first `k` rows. `board[1...k]` where `board[i]` is the column of the queen in row `i`.
*   **Choice:** For the next row (`k+1`), try placing a queen in each column `j` (from 1 to `n`).
*   **Constraint Check (Pruning):** Before placing a queen at `(k+1, j)`, check if this position is attacked by any queen already placed in rows `1` to `k`. Check:
    *   Same column: Is `board[i] == j` for any `i` from `1` to `k`?
    *   Same diagonal: Is `abs(board[i] - j) == abs(i - (k+1))` for any `i` from `1` to `k`? (Difference in rows equals difference in columns).
*   **Promising:** If the position `(k+1, j)` is safe (not attacked), place the queen (`board[k+1] = j`) and recursively call to place a queen in row `k+2`.
*   **Backtrack:** If the recursive call returns (meaning it either found a solution down that path or exhausted all possibilities from that placement), *or* if no column `j` in row `k+1` is safe, then we remove the queen from row `k+1` (conceptually, by returning from the function for row `k+1`) and try the next available column in row `k` (handled by the loop structure in the calling function for row `k`).
*   **Solution:** A solution is found when we successfully place a queen in row `n`.

**Pseudocode (Finding all solutions):**

```pseudocode
function Solve_N_Queens(n):
  // board[r] stores the column number for the queen in row r (1-based index)
  board = array[1..n]
  Solutions = [] // List to store valid board configurations

  function Is_Safe(row, col, board):
    // Check attack from queens in previous rows (1 to row-1)
    for prev_row from 1 to row - 1:
      prev_col = board[prev_row]
      // Check same column
      if prev_col == col:
        return false
      // Check diagonals
      if abs(prev_row - row) == abs(prev_col - col):
        return false
    return true // Position (row, col) is safe

  function Place_Queen(row):
    // Base Case: If all queens are placed (reached row n+1)
    if row == n + 1:
      // Found a solution, add a copy of the current board configuration
      Add copy(board) to Solutions
      return

    // Try placing a queen in each column of the current row
    for col from 1 to n:
      if Is_Safe(row, col, board):
        // Place the queen
        board[row] = col

        // Recurse to place the queen in the next row
        Place_Queen(row + 1)

        // Backtrack: Implicitly handled by the loop continuing
        // No need to explicitly reset board[row] here if we always overwrite it,
        // but conceptually, the placement is undone when moving to the next 'col'
        // or when returning from this function call.

  // Start the process by placing the first queen in row 1
  Place_Queen(1)
  return Solutions

// Initial call:
// all_solutions = Solve_N_Queens(8)
```

---

### 3. Knapsack Problem (0/1 using Backtracking)

**Problem Statement:**
Given a set of *n* items, each with a weight `w[i]` and a value `v[i]`, and a knapsack with a maximum weight capacity `W`. Find the subset of items that maximizes the total value without exceeding the weight capacity `W`. (Using backtracking, potentially less efficient than DP for the standard version, but illustrates the technique).

**Solution (Why Backtracking?):**
The problem involves making a sequence of decisions: for each item, either include it or exclude it. This creates a binary decision tree (state-space tree).

*   **State:** Can be defined by `(k, current_weight, current_value)`, representing the decision being made for item `k`, the accumulated weight so far, and the accumulated value so far.
*   **Choice:** For item `k`, the choices are:
    1.  Include item `k`.
    2.  Exclude item `k`.
*   **Constraint Check (Pruning):**
    1.  **Weight Pruning:** If including item `k` makes `current_weight + w[k] > W`, then this choice (and the entire subtree below it) cannot lead to a valid solution. Prune this branch (don't recurse down the "include" path).
*   **Promising:** A path is promising if the weight constraint is not violated.
*   **Backtrack:** Exploration proceeds down the tree (DFS). When a path is pruned or reaches a leaf node (all items considered), the algorithm backtracks to explore the alternative choice (e.g., if "include item `k`" was explored, backtrack and explore "exclude item `k`").
*   **Solution:** Keep track of the maximum value found (`max_value`) among all valid complete paths (leaf nodes reached without violating constraints). Update `max_value` whenever a path yields a higher total value.

*(Note: A more advanced optimization, often used in Branch and Bound which builds on backtracking, involves calculating an upper bound on the best possible value achievable from the current state. If `current_value + upper_bound_of_remaining_items < max_value_found_so_far`, we can prune that branch too. For basic backtracking, weight pruning is the primary mechanism).*

**Pseudocode (Finding max value):**

```pseudocode
function Knapsack_Backtrack(W, weights, values, n):
  // W: Knapsack capacity
  // weights: array of item weights (0-indexed)
  // values: array of item values (0-indexed)
  // n: number of items

  max_value_found = 0

  // Recursive helper function
  // k: index of the item currently being considered
  // current_weight: total weight of items included so far
  // current_value: total value of items included so far
  function Find_Max_Value(k, current_weight, current_value):
    nonlocal max_value_found // Allow modification of outer scope variable

    // Base Case: All items have been considered
    if k == n:
      // Update the maximum value found if the current solution is better
      if current_value > max_value_found:
        max_value_found = current_value
      return

    // --- Explore Branch 1: Exclude item k ---
    Find_Max_Value(k + 1, current_weight, current_value)

    // --- Explore Branch 2: Include item k (if possible) ---
    // Check Pruning Condition: Can item k be included without exceeding capacity?
    if current_weight + weights[k] <= W:
      // Include item k and recurse
      Find_Max_Value(k + 1, current_weight + weights[k], current_value + values[k])

    // Backtracking is implicit: when the function calls return, the state reverts
    // to before the choice was made for item k (for the calling function's level).

  // Initial call starting with item 0, weight 0, value 0
  Find_Max_Value(0, 0, 0)

  return max_value_found

// Example Usage:
// result = Knapsack_Backtrack(Capacity, weight_array, value_array, num_items)
```

Backtracking provides a structured way to explore possibilities, and its efficiency heavily depends on how effectively the search space can be pruned using constraints.