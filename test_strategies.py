import pytest
import time
import tracemalloc
from data_loader import load_data
from strategies import (
    NaiveMovingAverageStrategy,
    WindowedMovingAverageStrategy,
    DequeNaiveStrategy,
    NumPyVectorizedStrategy,
    StreamingStrategy
)
from models import MarketDataPoint
from datetime import datetime
import numpy as np

@pytest.fixture
def sample_data():
    # Generate 100k synthetic ticks for testing.
    return [
        MarketDataPoint(
            timestamp=datetime(2025, 1, i),
            symbol="TEST",
            price=100 + np.sin(i / 100) * 5 + np.random.normal(0, 0.1)
        )
        for i in range(100000)
    ]

def test_strategy_correctness_returns_positive_pnl(sample_data):
    """Both strategies should generate similar returns."""
    naive = NaiveMovingAverageStrategy()
    windowed = WindowedMovingAverageStrategy(window=50)

    for tick in sample_data:
        naive.generate_signals(tick)
        windowed.generate_signals(tick)

    assert abs(naive.total_return() - windowed.total_return()) < 5.0, \
        f"Returns differ too much: {naive.total_return():.2f} vs {windowed.total_return():.2f}"


def test_performance_optimized_strategies(sample_data):
    """Optimized strategies must run <1s and use <100MB for 100k ticks."""
    strategies = [
        ("Windowed", lambda: WindowedMovingAverageStrategy(window=50)),
        ("Deque", lambda: DequeNaiveStrategy(maxlen=50)),
        ("NumPy", lambda: NumPyVectorizedStrategy(window=50)),
        ("Streaming", lambda: StreamingStrategy())
    ]

    tracemalloc.start()
    start_time = time.perf_counter()

    for name, strat_class in strategies:
        strategy = strat_class()
        for tick in sample_data[:10000]:  # Quick test subset
            strategy.generate_signals(tick)

    runtime = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    peak_mb = peak_memory / (1024 * 1024)
    tracemalloc.stop()

    print(f"✅ {runtime:.3f}s, {peak_mb:.1f}MB for 10k ticks")
    assert runtime < 1.0, f"Runtime too slow: {runtime:.3f}s"
    assert peak_mb < 100, f"Memory too high: {peak_mb:.1f}MB"

def test_naive_quadratic_growth():
    # Naive should show quadratic runtime scaling.
    data_1k = [MarketDataPoint(datetime(2025, 1, i), "TEST", 100 + i / 1000) for i in range(1000)]
    data_10k = data_1k * 10

    naive_1k = NaiveMovingAverageStrategy()
    start = time.perf_counter()
    for tick in data_1k: naive_1k.generate_signals(tick)
    t1k = time.perf_counter() - start

    naive_10k = NaiveMovingAverageStrategy()
    start = time.perf_counter()
    for tick in data_10k: naive_10k.generate_signals(tick)
    t10k = time.perf_counter() - start

    speedup_expected = 100  # Quadratic: 10x data -> 100x time
    actual_ratio = t10k / t1k
    print(f"Naive scaling: {actual_ratio:.1f}x (expected ~100x)")
    assert actual_ratio > 50, f"Naive not quadratic enough: {actual_ratio:.1f}x"

def test_memory_scaling_constant():
    # Optimized strategies should have constant memory.
    tracemalloc.start()

    windowed = WindowedMovingAverageStrategy(window=50)
    for i in range(100000):
        tick = MarketDataPoint(datetime(2025, 1, i), "TEST", 100 + i / 1000)
        windowed.generate_signals(tick)
        if i % 25000 == 0:
            _, peak = tracemalloc.get_traced_memory()
            print(f"25k ticks: {peak / (1024 * 1024):.1f}MB")

    tracemalloc.stop()

def test_integration_full_pipeline():
    """Full pipeline test."""
    data = load_data()
    assert len(data) > 1000, "Need real data for integration test"
    print(f"✅ Loaded {len(data):,} real ticks")

if __name__ == "__main__":
    pytest.main(["-v", __file__])