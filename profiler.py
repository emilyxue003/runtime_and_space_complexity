# Runtime and memory measurement

import time
import tracemalloc
import matplotlib.pyplot as plt
import cProfile
import pstats
from tqdm import tqdm
from data_loader import load_data
from strategies import (
    NaiveMovingAverageStrategy,
    WindowedMovingAverageStrategy,
    DequeNaiveStrategy,
    NumPyVectorizedStrategy,
    StreamingStrategy
)

def measure_runtime(strategy_class, data, **kwargs):
    # Measure execution time for strategy on data
    strategy = strategy_class(**kwargs)
    start = time.perf_counter()
    for tick in tqdm(data, desc=f"{strategy_class.__name__} runtime"):
        strategy.generate_signals(tick)
    return time.perf_counter() - start

def measure_memory(strategy_class, data, **kwargs):
    # Measure peak memory usage (MB)
    tracemalloc.start()
    strategy = strategy_class(**kwargs)
    for tick in data:
        strategy.generate_signals(tick)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)  # bytes to MB

def profile_cprofile(strategy_class, data, **kwargs):
    # cProfile the strategy execution
    strategy = strategy_class(**kwargs)
    pr = cProfile.Profile()
    pr.enable()
    for tick in data:
        strategy.generate_signals(tick)
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # top 10 functions

def benchmark_strategies():
    # Benchmark both strategies at 1K, 10K, 100K sizes
    data = load_data()
    sizes = [1000, 10000, 100000]

    strategies = {
        "Naive": NaiveMovingAverageStrategy,
        "Windowed": lambda: WindowedMovingAverageStrategy(window=50),
        "Deque": lambda: DequeNaiveStrategy(maxlen=50),
        "NumPy": lambda: NumPyVectorizedStrategy(window=50),
        "Streaming": StreamingStrategy,
    }

    results = {
        'sizes': [],
        **{name: [] for name, _ in strategies.items()}
    }

    for size in sizes:
        if size > len(data):
            print(f"Skipping {size:,} (only {len(data):,} ticks available)")
            continue

        test_data = data[:size]
        print(f"\n=== {size:,} ticks ===")

        results['sizes'].append(size)

        for name, strat_class in strategies.items():
            time_taken = measure_runtime(strat_class, test_data)
            mem_used = measure_memory(strat_class, test_data)
            results[name].append((time_taken, mem_used))
            print(f"  {name:12}: {time_taken:6.4f}s  {mem_used:6.2f}MB")

    # Flatten results for plotting
    for name in strategies:
        times, mems = zip(*results[name])
        results[f"{name}_time"] = list(times)
        results[f"{name}_memory"] = list(mems)

    return results

def plot_results(results):
    # Plot runtime and memory vs input size
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    strategy_names = ["Naive", "Windowed", "Deque", "NumPy", "Streaming"]
    sizes = results['sizes']

    # Runtime plot
    for name in strategy_names:
        if f"{name}_time" in results:
            ax1.plot(sizes, results[f"{name}_time"], 'o-',
                     label=name, linewidth=2, markersize=6)
    ax1.set_xlabel('Input Size (ticks)')
    ax1.set_ylabel('Runtime (s)')
    ax1.set_title('Runtime Scaling')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Memory plot
    for name in strategy_names:
        if f"{name}_memory" in results:
            ax2.plot(sizes, results[f"{name}_memory"], 'o-',
                     label=name, linewidth=2, markersize=6)
    ax2.set_xlabel('Input Size (ticks)')
    ax2.set_ylabel('Peak Memory (MB)')
    ax2.set_title('Memory Scaling')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    plt.tight_layout()
    plt.savefig('complexity_plots.png', dpi=300, bbox_inches='tight')
    plt.show()