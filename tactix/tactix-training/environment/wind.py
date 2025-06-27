from dataclasses import dataclass
from environment.vector import Vector2

@dataclass
class Wind:
    direction: float
    speed: float