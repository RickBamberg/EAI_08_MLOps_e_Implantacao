# monitoring/state_manager.py

from collections import defaultdict, deque

class StateManager:

    def __init__(self):
        self.customers = {}

    def get_state(self, customer_id):
        if customer_id not in self.customers:
            self.customers[customer_id] = {
                "total_tx": 0,
                "sum_amount": 0.0,
                "sum_amount_sq": 0.0,
                "last_5_amounts": deque(maxlen=5),
                "zipcodes": set(),
                "merchants": defaultdict(int),
                "step_counts": defaultdict(int)
            }
        return self.customers[customer_id]

    def update_state(self, customer_id, transaction):

        state = self.get_state(customer_id)

        amount = float(transaction["amount"])
        step = transaction["step"]
        zipcode = transaction.get("zipcodeOri", 0)
        merchant = transaction["merchant"]

        state["total_tx"] += 1
        state["sum_amount"] += amount
        state["sum_amount_sq"] += amount ** 2
        state["last_5_amounts"].append(amount)
        state["zipcodes"].add(zipcode)
        state["merchants"][merchant] += 1
        state["step_counts"][step] += 1


state_manager = StateManager()

