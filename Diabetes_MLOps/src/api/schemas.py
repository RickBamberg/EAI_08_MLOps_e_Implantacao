from pydantic import BaseModel, Field

class DiabetesRequest(BaseModel):
    gravidez: int = Field(..., alias="Gravidez")
    glicose: int = Field(..., alias="Glicose")
    pressao_arterial: int = Field(..., alias="Pressao_arterial")
    espessura_da_pele: int = Field(..., alias="Espessura_da_pele")
    insulina: int = Field(..., alias="Insulina")
    imc: float = Field(..., alias="IMC")
    diabetes_descendente: float = Field(..., alias="Diabetes_Descendente")
    idade: int = Field(..., alias="Idade")

    class Config:
        allow_population_by_field_name = True
