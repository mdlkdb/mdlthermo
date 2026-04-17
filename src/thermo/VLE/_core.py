from typing import Literal, TypeAlias
from pydantic import BaseModel, ConfigDict

from .. import Compounds

Modes: TypeAlias = Literal["isobaric", "isothermal"]
FugacitiyModels: TypeAlias = Literal["IdealGas"]
ActivityModels: TypeAlias = Literal["UNIFAC", "COSMO-SAC", "NRTL"]
VaporPressureModels: TypeAlias = Literal["GC-GCN", "Wagner25", "Antonine"]


class BinaryVLECalculationModule(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    compounds: list[Compounds] = []
    x1: list[float] = []
    y1: list[float] = []
    T: list[float] | float = []
    P: list[float] | float = 101.315
    mode: Modes = "isobaric"
    fugacity_model: FugacitiyModels = "IdealGas"
    activity_model: ActivityModels = "UNIFAC"
    vapor_pressure_model: VaporPressureModels = "GC-GCN"
    fugacity_parameters: list[float] = []
    activity_parameters: list[float] = []
    vapor_pressure_parameters: list[float] = []

    def cal(self):
        pass
