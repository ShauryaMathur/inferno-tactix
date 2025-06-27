import math
from typing import Dict
# from tactix.utils.math_utils import dist, get_direction_factor
from dataclasses import dataclass
from environment.enums import VegetationType
# Vector simulation
from environment.vector import Vector2
from environment.cell import Cell

@dataclass
class CellProps:
    x: int
    y: int
    vegetation: VegetationType
    moisture_content: float
    elevation: float

@dataclass
class WindProps:
    speed:float
    direction:float

@dataclass
class Fuel:
    sav: float
    net_fuel_load: float
    fuel_bed_depth: float
    packing_ratio: float
    mx: float

FuelConstants: Dict[VegetationType, Fuel] = {
    # True surface fuels
    VegetationType.Grasslands: Fuel(2100, 0.294, 3.0, 0.00306, 0.15),
    VegetationType.ClosedShrublands: Fuel(1672, 0.239, 1.2, 0.01198, 0.30),
    VegetationType.OpenShrublands: Fuel(1500, 0.20, 1.0, 0.010, 0.30),

    # Forest litter
    VegetationType.EvergreenNeedleleaf: Fuel(1716, 0.0459, 0.10, 0.04878, 0.20),
    VegetationType.DeciduousNeedleleaf: Fuel(1650, 0.040, 0.10, 0.045, 0.20),
    VegetationType.EvergreenBroadleaf: Fuel(1500, 0.06, 0.12, 0.035, 0.25),
    VegetationType.DeciduousBroadleaf: Fuel(1400, 0.05, 0.10, 0.030, 0.25),
    VegetationType.MixedForest: Fuel((1716 + 1500) / 2, (0.0459 + 0.06) / 2, 0.11, 0.042, 0.22),

    # Savanna / woody grassland
    VegetationType.WoodySavannas: Fuel(1400, 0.18, 0.8, 0.008, 0.30),
    VegetationType.Savannas: Fuel(1450, 0.17, 0.7, 0.0085, 0.25),

    # Agricultural
    VegetationType.Croplands: Fuel(1200, 0.20, 0.50, 0.015, 0.30),
    VegetationType.CroplandMosaic: Fuel(1300, 0.18, 0.60, 0.012, 0.25),

    # Wet or sparsely vegetated
    VegetationType.PermanentWetlands: Fuel(800, 0.12, 0.50, 0.007, 0.50),
    VegetationType.Barren: Fuel(600, 0.01, 0.05, 0.002, 0.10),

    # Non-burnable
    VegetationType.UrbanBuilt: Fuel(0, 0, 0, 0, 0),
    VegetationType.SnowIce: Fuel(0, 0, 0, 0, 0),
    VegetationType.Water: Fuel(0, 0, 0, 0, 0),
}

UNBURNABLE_VEGETATION = {
    VegetationType.UrbanBuilt,
    VegetationType.SnowIce,
    VegetationType.Water
}

heat_content = 8000
total_mineral_content = 0.0555
effective_mineral_content = 0.01

def get_direction_factor(source_cell: Cell, target_cell : Cell, effective_wind_speed: float, max_spread_direction: float) -> float:
    """
    Calculate the direction factor for fire spread based on wind and relative direction.
    
    Args:
        source_cell (Cell): Cell with `x`, `y` attributes.
        target_cell (Cell): Neighbor cell with `x`, `y` attributes.
        effective_wind_speed (float): Wind speed in feet per minute.
        max_spread_direction (float): Angle (in radians) indicating max fire spread direction.

    Returns:
        float: Directional factor (scalar modifier for spread rate).
    """
    # Convert wind speed from ft/min to mph
    effective_wind_speed_mph = effective_wind_speed / 88.0
    Z = 1 + 0.25 * effective_wind_speed_mph
    e = math.sqrt(Z ** 2 - 1) / Z

    cell_vector = Vector2(target_cell.x - source_cell.x, target_cell.y - source_cell.y)
    relative_angle = abs(cell_vector.angle() - max_spread_direction)

    return (1 - e) / (1 - e * math.cos(relative_angle))

def dist(c1, c2):
    """
    Computes the distance between two cells.
    If they are aligned vertically or horizontally, returns Manhattan distance.
    If diagonal, returns Euclidean distance.
    
    Args:
        c1: Object or dict with 'x' and 'y' attributes or keys.
        c2: Same as c1.
    
    Returns:
        float: Distance between the cells.
    """
    x_diff = c1.x - c2.x
    y_diff = c1.y - c2.y

    if x_diff == 0 or y_diff == 0:
        return abs(x_diff + y_diff)
    else:
        return math.sqrt(x_diff ** 2 + y_diff ** 2)

def get_fire_spread_rate(source_cell: Cell, target_cell: Cell,
                         wind: WindProps, cell_size: float) -> float:
    
    if target_cell.vegetation in UNBURNABLE_VEGETATION:
        return 0.0
    fuel = FuelConstants[target_cell.vegetation]
    sav = fuel.sav
    packing_ratio = fuel.packing_ratio
    net_fuel_load = fuel.net_fuel_load
    mx = fuel.mx
    fuel_bed_depth = fuel.fuel_bed_depth

    moisture_content_ratio = target_cell.moistureContent / mx
    sav_factor = math.pow(sav, 1.5)

    a = 133 * math.pow(sav, -0.7913)
    b = 0.02526 * math.pow(sav, 0.54)
    c = 7.47 * math.exp(-0.133 * math.pow(sav, 0.55))
    e = 0.715 * (-0.000359 * sav)

    max_reaction_velocity = sav_factor / (495 + (0.0594 * sav_factor))
    optimum_packing_ratio = 3.348 * math.pow(sav, -0.8189)
    optimum_reaction_velocity = max_reaction_velocity * math.pow(packing_ratio / optimum_packing_ratio, a) * \
        math.exp(a * (1 - (packing_ratio / optimum_packing_ratio)))

    moisture_damping = 1 - (2.59 * moisture_content_ratio) + \
                       (5.11 * math.pow(moisture_content_ratio, 2)) - \
                       (3.52 * math.pow(moisture_content_ratio, 3))

    mineral_damping = 0.174 * math.pow(effective_mineral_content, -0.19)
    reaction_intensity = optimum_reaction_velocity * net_fuel_load * heat_content * moisture_damping * mineral_damping

    propagating_flux = 1 / (192 + (0.2595 * sav)) * math.exp((0.792 + (0.681 * math.sqrt(sav))) * (packing_ratio + 0.1))
    fuel_load = net_fuel_load / (1 - total_mineral_content)
    bulk_density = fuel_load / fuel_bed_depth
    effective_heating_number = math.exp(-138 / sav)
    heat_pre_ignition = 250 + (1116 * target_cell.moistureContent)

    r0 = reaction_intensity * propagating_flux / (bulk_density * effective_heating_number * heat_pre_ignition)

    wind_speed_ft_per_min = wind.speed * 88
    wind_factor = c * math.pow(wind_speed_ft_per_min, b) * math.pow(packing_ratio / optimum_packing_ratio, -e)

    dist_in_ft = dist(source_cell, target_cell) * cell_size
    elevation_diff = target_cell.elevation - source_cell.elevation
    slope_tan = elevation_diff / dist_in_ft if dist_in_ft != 0 else 0
    slope_factor = 5.275 * math.pow(packing_ratio, -0.3) * math.pow(slope_tan, 2)

    ORIGIN = Vector2(0, 0)

    wind_vector = Vector2(0, -1).rotateAround(ORIGIN, -math.radians(wind.direction))
    if target_cell.elevation >= source_cell.elevation:
        upslope_vector = Vector2(target_cell.x - source_cell.x, target_cell.y - source_cell.y)
    else:
        upslope_vector = Vector2(source_cell.x - target_cell.x, source_cell.y - target_cell.y)

    dw = r0 * wind_factor
    wind_vector.setLength(dw)

    ds = r0 * slope_factor
    upslope_vector.setLength(ds)

    max_spread_vector = wind_vector.add(upslope_vector)

    rh = r0 + max_spread_vector.length()
    effective_wind_factor = rh / r0 - 1

    # effective_wind_speed = math.pow(effective_wind_factor / (c * math.pow(packing_ratio / optimum_packing_ratio, -e)), 1 / b)
    ratio = packing_ratio / optimum_packing_ratio
    if ratio <= 0:
        return 0.0  # Invalid for exponentiation

    denominator = c * math.pow(ratio, -e)
    if denominator == 0:
        return 0.0

    base = effective_wind_factor / denominator
    if base <= 0:
        return 0.0  # Can't take fractional power of non-positive number

    effective_wind_speed = math.pow(base, 1 / b)
    direction_factor = get_direction_factor(source_cell, target_cell, effective_wind_speed, max_spread_vector.angle())

    return rh * direction_factor

