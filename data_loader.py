# CSV parsing and dataclass creation

from models import MarketDataPoint
from datetime import datetime
import csv

def load_data(path: str = "market_data.csv"):
    datapoints = []

    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            datapoint = MarketDataPoint(
                timestamp=datetime.fromisoformat(row["timestamp"]),
                symbol=row["symbol"],
                price=float(row["price"]),
            )
            datapoints.append(datapoint)

    return datapoints