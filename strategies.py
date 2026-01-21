# Naive and optimized strategy implementations

from models import Strategy, MarketDataPoint
from statistics import mean
from collections import deque
import numpy as np
from functools import lru_cache

# For each tick, recompute the average price from scratch
# Time per tick: appending is O(1); mean(self.prices) is O(n) because it scans the whole list
# Space: self.prices stores all past prices, so total space is O(n)
class NaiveMovingAverageStrategy(Strategy):
    def __init__(self):
        self.prices = []            # Full price history
        self.ma = None              # Current moving average (scalar -> O(1) space)
        self.position = 0           # 0 = flat, 1 = long (scalar -> O(1) space)
        self.entry_price = None     # Price at which current position was opened (scalar -> O(1) space)
        self.realized_pnl = 0.0     # Cumulative realized profit/loss (scalar -> O(1) space)

    def generate_signals(self, tick: MarketDataPoint) -> list:
        price = tick.price
        signals = []

        # Add newest price: O(1) to append a single element
        self.prices.append(price)

        # First tick just initialize ma: : O(1) time
        if self.ma is None:
            self.ma = price
            return signals

        # Price above ma -> enter long; price below -> exit position
        # Trading rule checks and assignments are all O(1)
        if price > self.ma and self.position == 0:
            self.position = 1
            self.entry_price = price
            signals.append("BUY")
        elif price < self.ma and self.position == 1:
            self.position = 0
            self.realized_pnl += price - self.entry_price
            self.entry_price = None
            signals.append("SELL")

        # Update the moving average: iterates over all stored prices -> O(n) time
        self.ma = mean(self.prices)

        # Overall per-tick time: O(n); overall space: O(n)
        return signals

    def total_return(self):
        # O(1) time, just return a scalar
        return self.realized_pnl

# Maintain a fixed-size buffer and update the average incrementally with a running sum
# Time per tick: all operations are O(1); no loop over all past ticks
# Space: self.prices holds at most k prices where k = window size -> O(k) space regardless of total ticks
class WindowedMovingAverageStrategy(Strategy):
    def __init__(self, window=10):
        self.window = window        # Size of the moving average window (scalar -> O(1) space)
        self.prices = []            # Recent prices up to window length (O(k) space)
        self.sum_prices = 0.0       # Running sum of prices in window (scalar -> O(1) space)
        self.ma = None              # Current moving average (scalar -> O(1) space)
        self.position = 0           # 0 = flat, 1 = long (scalar -> O(1) space)
        self.entry_price = None     # Price at which current position was opened (scalar -> O(1) space)
        self.realized_pnl = 0.0     # Cumulative realized profit/loss (scalar -> O(1) space)

    def generate_signals(self, tick: MarketDataPoint) -> list:
        price = tick.price
        signals = []

        # Remove oldest price if window is full
        # Since k is a fixed small window, functionally it's O(1) with respect to total ticks
        if len(self.prices) == self.window:
            oldest = self.prices.pop(0)
            self.sum_prices -= oldest

        # Add newest price: O(1) time
        self.prices.append(price)
        self.sum_prices += price

        # First tick just initialize ma: O(1) time
        if len(self.prices) == 1:
            self.ma = price
            return signals

        # Update the moving average: O(1) time
        self.ma = self.sum_prices / len(self.prices)

        # Price above ma -> enter long; price below -> exit position
        # Trading rule checks and assignments are all O(1)
        if price > self.ma and self.position == 0:
            self.position = 1
            self.entry_price = price
            signals.append("BUY")
        elif price < self.ma and self.position == 1:
            self.position = 0
            self.realized_pnl += price - self.entry_price
            self.entry_price = None
            signals.append("SELL")

        # Overall per-tick time: O(1); space: O(k) for prices and O(1) for scalars
        return signals

    def total_return(self):
        # O(1) time, just return a scalar
        return self.realized_pnl

# Naive strategy but uses deque(maxlen=N) to keep only recent prices.
# Time: O(k) per tick where k = maxlen (still uses mean()).
# Space: O(k) instead of O(n).
class DequeNaiveStrategy(Strategy):
    def __init__(self, maxlen=1000):        # Capped to prevent O(n) memory
        self.prices = deque(maxlen=maxlen)
        self.ma = None
        self.position = 0
        self.entry_price = None
        self.realized_pnl = 0.0

    def generate_signals(self, tick: MarketDataPoint) -> list:
        price = tick.price
        signals = []

        # deque automatically drops oldest when maxlen reached
        self.prices.append(price)

        if self.ma is None:
            self.ma = price
            return signals

        # Still O(k) time due to mean(), but k is bounded
        self.ma = mean(self.prices)

        if price > self.ma and self.position == 0:
            self.position = 1
            self.entry_price = price
            signals.append("BUY")
        elif price < self.ma and self.position == 1:
            self.position = 0
            self.realized_pnl += price - self.entry_price
            self.entry_price = None
            signals.append("SELL")

        return signals

    def total_return(self):
        # O(1) time, just return a scalar
        return self.realized_pnl

# Use NumPy arrays for faster computation (O(n) total time, O(k) space)
# NumPy array + running sum for fastest execution
class NumPyVectorizedStrategy(Strategy):
    def __init__(self, window=1000):
        self.window = window
        self.prices = np.zeros(window, dtype=np.float64)
        self.count = 0
        self.sum_prices = 0.0
        self.ma = None
        self.position = 0
        self.entry_price = None
        self.realized_pnl = 0.0

    def generate_signals(self, tick: MarketDataPoint) -> list:
        price = tick.price
        signals = []

        # Circular buffer in NumPy array
        if self.count == self.window:
            self.sum_prices -= self.prices[self.count % self.window]

        self.prices[self.count % self.window] = price
        self.sum_prices += price
        self.count += 1

        if self.count == 1:
            self.ma = price
            return signals

        # O(1) average
        self.ma = self.sum_prices / min(self.count, self.window)

        # Trading logic
        if price > self.ma and self.position == 0:
            self.position = 1
            self.entry_price = price
            signals.append("BUY")
        elif price < self.ma and self.position == 1:
            self.position = 0
            self.realized_pnl += price - self.entry_price
            self.entry_price = None
            signals.append("SELL")

        return signals

    def total_return(self):
        # O(1) time, just return a scalar
        return self.realized_pnl

# Process data as a stream without storing prices (O(1) time, O(1) space)
# Generator-based: no price storage at all.
# Uses previous-MA as proxy (like EMA approximation).
# Time: O(1), Space: O(1).
class StreamingStrategy(Strategy):
    def __init__(self, alpha=0.1):  # EMA smoothing factor
        self.alpha = alpha
        self.prev_ma = None
        self.position = 0
        self.entry_price = None
        self.realized_pnl = 0.0

    @lru_cache(maxsize=1)  # memoize last computation
    def ema_update(self, price, prev_ma):
        if prev_ma is None:
            return price
        return self.alpha * price + (1 - self.alpha) * prev_ma

    def generate_signals(self, tick: MarketDataPoint) -> list:
        price = tick.price
        signals = []

        self.prev_ma = self.ema_update(price, self.prev_ma)

        if price > self.prev_ma and self.position == 0:
            self.position = 1
            self.entry_price = price
            signals.append("BUY")
        elif price < self.prev_ma and self.position == 1:
            self.position = 0
            self.realized_pnl += price - self.entry_price
            self.entry_price = None
            signals.append("SELL")

        return signals

    def total_return(self):
        # O(1) time, just return a scalar
        return self.realized_pnl