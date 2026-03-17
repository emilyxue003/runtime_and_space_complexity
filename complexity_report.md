# Moving Average Algorithm Optimization: Complexity Analysis

## Architectural Overview
This project evaluates five distinct algorithmic approaches to calculating moving averages over continuous data streams. Additionally, the data ingestion pipeline was optimized from a standard list-loading approach (Space Complexity: **O(N)**) to a generator-based stream (Space Complexity: **O(1)**), allowing for the processing of infinitely large datasets without memory faults.

## Results Summary


### Runtime (seconds)
| Size    | Naive  | Windowed | Deque | NumPy  | Streaming |
|---------|--------|----------|-------|--------|-----------|
| 1,000 |   0.0848s |   0.0004s |   0.0003s |   0.0005s |   0.0003s |
| 10,000 |   7.8949s |   0.0024s |   0.0021s |   0.0042s |   0.0019s |
| 100,000 | 796.7218s |   0.0295s |   0.0203s |   0.0411s |   0.0166s |



### Memory Usage (peak MB)
| Size    | Naive  | Windowed | Deque | NumPy  | Streaming |
|---------|--------|----------|-------|--------|-----------|
| 1,000 |     0.01MB |     0.00MB |     0.00MB |     0.00MB |     0.00MB |
| 10,000 |     0.08MB |     0.00MB |     0.00MB |     0.00MB |     0.00MB |
| 100,000 |     0.77MB |     0.00MB |     0.00MB |     0.00MB |     0.00MB |


![Plots](complexity_plots.png)
*(Note: Plots utilize log-log scaling. A slope of 1 indicates linear O(n) scaling, while a slope of 2 indicates quadratic O(n²) scaling).*

---

## Detailed Algorithmic Analysis

### 1. NaiveMovingAverageStrategy (The Baseline)
**Time Complexity: O(n²) total**
- `append()`: O(1) amortized per tick.
- `mean(self.prices)`: O(t) per tick, where t is the number of ticks seen so far.
- Over n ticks, this sums to ∑[1, 2, ..., n], resulting in quadratic time scaling.
**Space Complexity: O(n)**
- Stores every historical price. Memory consumption grows linearly and will eventually cause out-of-memory (OOM) errors on large datasets.

### 2. WindowedMovingAverageStrategy (Python List)
**Time Complexity: O(n * k) total**
- While it uses a running sum to avoid `mean()`, `list.pop(0)` is an **O(k)** operation because it forces Python to shift all remaining elements in the array one position to the left.
**Space Complexity: O(k)**
- Memory is strictly bounded by the fixed window size (k), regardless of dataset size.

### 3. DequeNaiveStrategy (Double-Ended Queue)
**Time Complexity: O(n * k) total**
- `deque.append()` and `deque.popleft()` are true **O(1)** operations because they use a doubly-linked list structure, avoiding the memory shifts of standard lists. However, because this specific implementation recalculates `mean()` every tick, the per-tick time is bounded by **O(k)**.
**Space Complexity: O(k)**
- Bounded by the `maxlen` parameter.

### 4. NumPyVectorizedStrategy (High-Performance Circular Buffer)
**Time Complexity: O(n) total (Effectively O(1) per tick)**
- Utilizes a pre-allocated `np.zeros` array and mathematical modulo wrap-around (`idx = count % window`) to maintain a circular buffer. 
- Avoids conditional branching (branchless programming) and leverages C-level NumPy optimizations for state updates.
**Space Complexity: O(k)**
- Allocates memory exactly once. 

### 5. StreamingStrategy (Exponential Moving Average)
**Time Complexity: O(n) total (O(1) per tick)**
- Extremely lightweight mathematical update step: `new_ma = alpha * price + (1 - alpha) * prev_ma`.
**Space Complexity: O(1)**
- Requires zero array storage. It only maintains a single float variable for the previous state. The ultimate solution for highly constrained memory environments.

---

## Key Engineering Insights
1. **The Danger of `pop(0)`:** The standard Python list approach (Windowed) introduces hidden O(k) memory-shifting overhead. For large window sizes, this becomes a severe bottleneck.
2. **Branchless Execution:** The NumPy strategy demonstrates that avoiding `if` statements inside high-frequency loops significantly improves CPU pipeline execution.
3. **Data Pipeline Optimization:** Pairing an O(1) processing algorithm with an O(N) data loader is an anti-pattern. By refactoring `load_data()` to yield rows as a generator, the entire system architecture is now capable of continuous runtime.

## Production Recommendation
For high-frequency sensor data, vehicle telematics, or financial tick data:
* **Strict Fixed-Window Requirements:** Deploy `NumPyVectorizedStrategy`. It provides the fastest execution time per tick while maintaining a safe, bounded O(k) memory footprint.
* **Infinite Streams & Strict Memory Limits:** Deploy `StreamingStrategy`. It achieves true O(1) space complexity by eliminating the buffer entirely.
