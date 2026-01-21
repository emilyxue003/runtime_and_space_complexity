# MarketDataPoint, Strategy base class

from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod

@dataclass(frozen=True)
class MarketDataPoint:
    timestamp: datetime
    symbol: str
    price: float

class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, tick: MarketDataPoint) -> list:
        pass