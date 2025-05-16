import numpy as np
import pandas as pd
import time
import random
from datetime import datetime

class TransactionSimulator:
    def __init__(self, fraud_ratio=0.05):
        """Initialize the transaction simulator with a given fraud ratio."""
        self.fraud_ratio = fraud_ratio
        self.transaction_id = 0

    def generate_transaction(self):
        """Generate a single credit card transaction."""
        self.transaction_id += 1

        # Determine if this transaction will be fraudulent
        is_fraud = random.random() < self.fraud_ratio

        # Generate transaction amount (fraudulent transactions tend to have specific patterns)
        if is_fraud:
            amount = random.choice([
                random.uniform(1, 10),           # Small test transactions
                random.uniform(900, 5000)        # Large transactions
            ])
        else:
            amount = random.lognormvariate(4, 1)  # Regular transaction distribution

        # Generate merchant category
        categories = ['grocery', 'restaurant', 'entertainment', 'travel', 'online', 'retail', 'other']
        category = random.choice(categories)

        # Time of day (fraudulent transactions often happen at odd hours)
        hour = random.randint(0, 23)
        if is_fraud:
            hour = random.choice([0, 1, 2, 3, 4, 22, 23])  # Late night/early morning

        # Number of recent transactions (fraud often happens after a period of inactivity)
        recent_transactions = random.randint(0, 15)
        if is_fraud:
            recent_transactions = random.randint(0, 3)  # Few recent transactions

        # Create transaction data
        transaction = {
            'transaction_id': self.transaction_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'amount': round(amount, 2),
            'merchant_category': category,
            'hour_of_day': hour,
            'recent_transactions': recent_transactions,
            'is_fraud': 1 if is_fraud else 0
        }

        return transaction

    def generate_batch(self, batch_size=10):
        """Generate a batch of transactions."""
        transactions = [self.generate_transaction() for _ in range(batch_size)]
        return pd.DataFrame(transactions)

    def stream_data(self, interval=1.0):
        """Continuously generate transactions at given interval."""
        while True:
            transaction = self.generate_transaction()
            df = pd.DataFrame([transaction])
            yield df
            time.sleep(interval)

# Example usage
if __name__ == "__main__":
    simulator = TransactionSimulator(fraud_ratio=0.1)

    # Generate a sample batch
    batch = simulator.generate_batch(batch_size=5)
    print(batch)

    # Stream some data
    generator = simulator.stream_data(interval=2.0)
    for i, transaction in enumerate(generator):
        print(f"Transaction {i+1}:")
        print(transaction)
        if i >= 4:  # Just show 5 examples
            break