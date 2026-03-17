# CSV parsing and dataclass creation

from models import MarketDataPoint
from datetime import datetime
import csv

def load_data(path: str = "market_data.csv"):
    """
    Yields MarketDataPoint objects one at a time.
    Space Complexity: O(1) regardless of file size.
    """
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            yield MarketDataPoint(
                timestamp=datetime.fromisoformat(row["timestamp"]),
                symbol=row["symbol"],
                price=float(row["price"]),
            )