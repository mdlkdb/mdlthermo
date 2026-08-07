from pydantic import BaseModel, ConfigDict


class Compounds(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    smiles: str
    Tb: float | None = None
    Tm: float | None = None
    Tc: float | None = None
    Pc: float | None = None
    Ac: float | None = None
