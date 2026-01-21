# Orchestrates ingestion, strategy execution, profiling

from data_loader import load_data
from strategies import (
    NaiveMovingAverageStrategy,
    WindowedMovingAverageStrategy,
    DequeNaiveStrategy,
    NumPyVectorizedStrategy,
    StreamingStrategy
)
from profiler import profile_cprofile, benchmark_strategies, plot_results
from reporting import generate_complexity_report

def main():
    # 1. Load CSV data: O(n) time to build the list, O(n) space for n ticks
    data = load_data()
    print(f"Loaded {len(data):,} ticks")

    # 2. Create strategy instances: O(1) time and space
    naive_strategy = NaiveMovingAverageStrategy()           # O(n) time per tick, O(n) space overall
    windowed_strategy = WindowedMovingAverageStrategy()     # O(1) time per tick, O(k) space where k = window size
    deque_strategy = DequeNaiveStrategy()
    numpy_strategy = NumPyVectorizedStrategy()
    streaming_strategy = StreamingStrategy()

    # 3. Run strategies
    for tick in data:
        naive_strategy.generate_signals(tick)
        windowed_strategy.generate_signals(tick)
        deque_strategy.generate_signals(tick)
        numpy_strategy.generate_signals(tick)
        streaming_strategy.generate_signals(tick)

    print("Naive strategy return:", naive_strategy.total_return())
    print("Windowed strategy return:", windowed_strategy.total_return())
    print("Deque strategy return:", deque_strategy.total_return())
    print("NumPy Strategy Return:", numpy_strategy.total_return())
    print("Streaming Strategy Return:", streaming_strategy.total_return())

    # 4. Call profiler functions
    results = benchmark_strategies()
    plot_results(results)

    print("\n=== cProfile: Naive Strategy (100K ticks) ===")
    data_100k = data[:100000]
    profile_cprofile(NaiveMovingAverageStrategy, data_100k)

    print("\n=== cProfile: Windowed Strategy (100K ticks) ===")
    profile_cprofile(WindowedMovingAverageStrategy, data_100k, window=10)

    # 5. Generate reports
    generate_complexity_report(results)

if __name__ == "__main__":
    main()