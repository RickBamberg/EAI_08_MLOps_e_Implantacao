"""
Schemas Pydantic para validação de dados da API
"""
from pydantic import BaseModel, Field
from typing import Optional


class Transaction(BaseModel):
    """
    Schema para entrada de transação
    
    Campos obrigatórios: step, amount, customer, merchant, category
    Campos opcionais: zipcodeOri, zipMerchant, gender, age
    """
    # Obrigatórios
    step: int = Field(..., description="Timestamp da transação", ge=0)
    amount: float = Field(..., description="Valor da transação", gt=0)
    customer: str = Field(..., description="ID do cliente")
    merchant: str = Field(..., description="ID do comerciante")
    category: str = Field(..., description="Categoria da transação")
    
    # Opcionais (com defaults)
    zipcodeOri: Optional[str] = Field(None, description="CEP de origem")
    zipMerchant: Optional[str] = Field(None, description="CEP do comerciante")
    gender: Optional[str] = Field(None, description="Gênero do cliente (M/F/U)")
    age: Optional[int] = Field(None, description="Categoria de idade", ge=0, le=6)
    
    class Config:
        schema_extra = {
            "example": {
                "step": 10,
                "amount": 950.0,
                "customer": "C123",
                "merchant": "M456",
                "category": "electronics",
                "zipcodeOri": "28007",
                "zipMerchant": "28007",
                "gender": "F",
                "age": 3
            }
        }


class PredictionResponse(BaseModel):
    """
    Schema para resposta de predição
    """
    request_id: str = Field(..., description="ID único da requisição")
    fraud_probability: float = Field(..., description="Probabilidade de fraude (0-1)", ge=0, le=1)
    fraud_prediction: int = Field(..., description="Predição binária (0=normal, 1=fraude)")
    model_version: str = Field(..., description="Versão do modelo")
    latency_ms: float = Field(..., description="Latência da predição em ms")
    error: Optional[str] = Field(None, description="Mensagem de erro, se houver")
    
    class Config:
        schema_extra = {
            "example": {
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "fraud_probability": 0.0234,
                "fraud_prediction": 0,
                "model_version": "v2",
                "latency_ms": 25.3,
                "error": None
            }
        }
