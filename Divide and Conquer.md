Okay class, today we'll explore a very intuitive yet powerful algorithmic paradigm: **Divide and Conquer**. Many efficient algorithms rely on this strategy.

---

### 1. Divide and Conquer: Methodology

**Problem Statement:**
How can we solve a large, complex problem by breaking it down into smaller, more manageable pieces, solving those pieces, and then putting their solutions together to solve the original problem?

**Solution (Why):**
The Divide and Conquer strategy involves three main steps, typically applied recursively:

1.  **Divide:** Break the original problem instance of size *n* into *a* smaller subproblems, each of size approximately *n/b*. The subproblems should ideally be smaller instances of the *same* original problem. (Often, *a*=2, *b*=2, meaning we divide into two equal halves).
2.  **Conquer:** Solve the *a* subproblems recursively. If the subproblem size is small enough (reaching a **base case**), solve it directly (e.g., an array of size 1 is already sorted).
3.  **Combine:** Combine the solutions of the subproblems into a solution for the original problem instance.

This strategy works well when the "Divide" and "Combine" steps are relatively efficient compared to solving the original problem brute-force. The recursive nature allows complex problems to be broken down until they become trivial (the base case). The efficiency often comes from the fact that solving multiple smaller problems and combining them can be faster than solving the large problem directly.

The running time of Divide and Conquer algorithms is often analyzed using **recurrence relations**, typically of the form `T(n) = aT(n/b) + f(n)`, where `T(n)` is the time for size `n`, `a` is the number of subproblems, `n/b` is the size of each subproblem, and `f(n)` is the time cost of the "Divide" and "Combine" steps. The Master Theorem is often used to solve such recurrences.

**Pseudocode (General Template):**

```pseudocode
function DivideAndConquer(Problem P):
  // n = size of problem P

  // Base Case: If the problem is small enough, solve directly
  if n <= BASE_CASE_SIZE:
    return Solve_Directly(P)

  // 1. Divide
  Subproblems[1..a] = Divide(P) // Divide P into a subproblems P1..Pa

  // 2. Conquer
  Solutions[1..a] = array of size a
  for i from 1 to a:
    Solutions[i] = DivideAndConquer(Subproblems[i]) // Solve recursively

  // 3. Combine
  FinalSolution = Combine(Solutions[1..a])

  return FinalSolution
```

---

### 2. Merge Sort

**Problem Statement:**
Given an array (or sequence) `A` of `n` elements `A[p..r]`, sort the elements in non-decreasing order using the Divide and Conquer strategy.

**Solution (Why):**
Merge Sort perfectly follows the Divide and Conquer paradigm:

1.  **Divide:** If the given array `A[p..r]` has more than one element (`p < r`), find the middle index `q = floor((p+r)/2)`. This divides the array into two subarrays: `A[p..q]` and `A[q+1..r]`. This step takes constant time, O(1).
2.  **Conquer:** Recursively sort the two subarrays `A[p..q]` and `A[q+1..r]` by calling `MergeSort` on them.
3.  **Combine:** Merge the two *sorted* subarrays `A[p..q]` and `A[q+1..r]` back into a single sorted subarray `A[p..r]`. This is done using a helper procedure `Merge(A, p, q, r)`. The `Merge` procedure takes two sorted subarrays and combines them by repeatedly comparing the smallest element of each subarray and placing the smaller of the two into the correct position in the original array segment `A[p..r]` (usually using temporary storage). This merging step takes linear time, O(n), where `n = r - p + 1` is the number of elements being merged.

**Base Case:** If `p >= r`, the subarray has 0 or 1 element, which is already sorted by definition. The recursion stops.

Merge Sort consistently achieves O(n log n) time complexity because the work at each level of recursion (the total merging cost) is O(n), and the depth of the recursion is O(log n). Its main disadvantage is the need for auxiliary space O(n) for the merging step.

**Pseudocode:**

```pseudocode
// Main Merge Sort function
function MergeSort(A, p, r):
  // A: array, p: start index, r: end index
  if p < r: // Check if there's more than one element (not base case)
    // 1. Divide
    q = floor((p + r) / 2)

    // 2. Conquer (Recursive calls)
    MergeSort(A, p, q)
    MergeSort(A, q + 1, r)

    // 3. Combine
    Merge(A, p, q, r)

// Helper Merge function
function Merge(A, p, q, r):
  // n1 = number of elements in left subarray A[p..q]
  n1 = q - p + 1
  // n2 = number of elements in right subarray A[q+1..r]
  n2 = r - q

  // Create temporary arrays L[1..n1] and R[1..n2]
  let L[1..n1 + 1] and R[1..n2 + 1] be new arrays

  // Copy data to temporary arrays
  for i from 1 to n1:
    L[i] = A[p + i - 1]
  for j from 1 to n2:
    R[j] = A[q + j]

  // Add sentinels (infinity) to simplify the merging loop
  L[n1 + 1] = infinity
  R[n2 + 1] = infinity

  // Merge the temporary arrays back into A[p..r]
  i = 1 // index for L
  j = 1 // index for R
  for k from p to r: // index for A
    if L[i] <= R[j]:
      A[k] = L[i]
      i = i + 1
    else:
      A[k] = R[j]
      j = j + 1
```

---

### 3. Quick Sort

**Problem Statement:**
Given an array `A` of `n` elements `A[p..r]`, sort the elements in non-decreasing order using the Divide and Conquer strategy, often aiming for good average-case performance and in-place sorting.

**Solution (Why):**
Quick Sort is another classic Divide and Conquer sorting algorithm, but its structure differs slightly from Merge Sort, particularly in where the main work occurs:

1.  **Divide:** Rearrange (partition) the array `A[p..r]` into two (possibly empty) subarrays `A[p..q-1]` and `A[q+1..r]` such that:
    *   An element `A[q]` (the **pivot**) is in its final sorted position.
    *   All elements in `A[p..q-1]` are less than or equal to `A[q]`.
    *   All elements in `A[q+1..r]` are greater than or equal to `A[q]`.
    This partitioning is done by a helper procedure `Partition(A, p, r)`, which returns the index `q` of the pivot. The `Partition` step takes linear time, O(n), where `n = r - p + 1`. This is where the main work of Quick Sort happens.
2.  **Conquer:** Recursively sort the two subarrays `A[p..q-1]` and `A[q+1..r]` by calling `QuickSort` on them.
3.  **Combine:** No explicit work is needed to combine the subarrays. Because the partitioning step places the pivot in its correct final position and arranges other elements relative to it, the entire array `A[p..r]` is sorted once the recursive calls return. This step takes constant time, O(1).

**Base Case:** If `p >= r`, the subarray has 0 or 1 element, which is already sorted. The recursion stops.

Quick Sort's performance heavily depends on the **pivot selection**.
*   **Best/Average Case:** If `Partition` consistently divides the array into roughly equal halves, the recurrence is `T(n) = 2T(n/2) + O(n)`, leading to O(n log n) time complexity.
*   **Worst Case:** If `Partition` consistently produces highly unbalanced splits (e.g., one subarray of size `n-1` and one of size 0, which happens if the pivot is always the smallest or largest element in a sorted/reverse-sorted array), the recurrence becomes `T(n) = T(n-1) + O(n)`, leading to O(n^2) time complexity.
Randomized pivot selection is often used to make the worst-case scenario highly unlikely, achieving O(n log n) performance on average. Quick Sort is often preferred in practice due to its typically lower constant factors and its in-place nature (requiring only O(log n) auxiliary stack space for recursion on average).

**Pseudocode:**

```pseudocode
// Main Quick Sort function
function QuickSort(A, p, r):
  // A: array, p: start index, r: end index
  if p < r: // Check if there's more than one element (not base case)
    // 1. Divide (Partition)
    q = Partition(A, p, r) // q is the index of the pivot after partitioning

    // 2. Conquer (Recursive calls)
    QuickSort(A, p, q - 1)
    QuickSort(A, q + 1, r)
    // 3. Combine: Trivial / No work needed

// Helper Partition function (Lomuto partition scheme example)
function Partition(A, p, r):
  // Choose the last element as the pivot
  pivot = A[r]
  // i tracks the index of the last element known to be <= pivot
  i = p - 1

  // Iterate through elements from p to r-1
  for j from p to r - 1:
    // If current element is less than or equal to pivot
    if A[j] <= pivot:
      // Move boundary of smaller elements forward
      i = i + 1
      // Swap A[i] with A[j] to place A[j] in the 'smaller' partition
      swap A[i] with A[j]

  // Place the pivot in its correct final position
  // Swap pivot (A[r]) with the element at index i+1
  swap A[i + 1] with A[r]

  // Return the index where the pivot ended up
  return i + 1
```

Both Merge Sort and Quick Sort beautifully illustrate the power of the Divide and Conquer strategy for designing efficient algorithms.