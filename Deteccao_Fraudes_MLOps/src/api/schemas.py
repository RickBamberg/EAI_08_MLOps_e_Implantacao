from pydantic import BaseModel
from typing import List, Optional

class Transaction(BaseModel):
    step: int
    amount: float
    customer: Optional[str] = None
    merchant: Optional[str] = None
    category: Optional[str] = None

class PredictionResponse(BaseModel):
    fraud_proba: float
    fraud_pred: int
    version: str
