import config
import math
import random
from typing import List, Dict, Callable
from environment.vector import Vector2
from environment.cell import Cell
from environment.enums import BurnIndex,FireState
from fire_engine.utils import dist, within_dist, get_grid_index_for_location, for_each_point_between, get_grid_cell_neighbors
from fire_engine.fire_spread_rate import get_fire_spread_rate
from environment.wind import Wind

model_day = 1440  # minutes
rng = random.Random("inferno-tactics")

end_of_low_intensity_fire_probability = {
    0: 0.0,
    1: 0.1,
    2: 0.1,
    3: 0.2,
    4: 0.3,
    5: 0.5,
    6: 0.7,
    7: 1.0
}

# def nonburnable_cell_between(cells: List[Cell], width: int, x0: int, y0: int, x1: int, y1: int, burn_index: BurnIndex):
#     result = False
#     def check_point(x, y):
#         nonlocal result
#         idx = get_grid_index_for_location(x, y, width)
#         if not cells[idx].is_burnable_for_bi(burn_index):
#             result = True
#     for_each_point_between(x0, y0, x1, y1, check_point)
#     return result

# def get_grid_cell_neighbors(cells: List[Cell], i: int, width: int, height: int, neighbors_dist: int, burn_index: BurnIndex):
#     neighbours = []
#     queue = [i]
#     processed = {i: True}
#     x0, y0 = i % width, i // width
#     any_nonburnable_cells = False

#     while queue:
#         j = queue.pop(0)
#         x1, y1 = j % width, j // width
#         for diff in direct_neighbours:
#             nx, ny = x1 + diff.x, y1 + diff.y
#             n_idx = get_grid_index_for_location(nx, ny, width)
#             if 0 <= nx < width and 0 <= ny < height and not processed.get(n_idx) and within_dist(x0, y0, nx, ny, neighbors_dist):
#                 if not cells[n_idx].is_burnable_for_bi(burn_index):
#                     any_nonburnable_cells = True
#                 elif not any_nonburnable_cells or not nonburnable_cell_between(cells, width, nx, ny, x0, y0, burn_index):
#                     neighbours.append(n_idx)
#                     queue.append(n_idx)
#                 processed[n_idx] = True
#     return neighbours

class FireEngine:
    def __init__(self, wind : Wind, config):
        # self.cells : List[Cell] = cells
        self.wind : Wind = wind
        self.grid_width = config.gridWidth
        self.grid_height = config.gridHeight
        self.cell_size = config.cellSize
        self.min_cell_burn_time = config.minCellBurnTime
        self.neighbors_dist = config.neighborsDist
        self.fire_survival_probability = config.fireSurvivalProbability
        self.end_of_low_intensity_fire = False
        self.fire_did_stop = False
        self.day = 0
        self.burned_cells_in_zone = {}

    def cell_at(self, cells : List[Cell],x, y):
        grid_x = int(x // self.cell_size)
        grid_y = int(y // self.cell_size)
        return cells[get_grid_index_for_location(grid_x, grid_y, self.grid_width)]

    def update_fire(self, cells : List[Cell],time):
        new_day = int(time // model_day)
        if new_day != self.day:
            self.day = new_day
            if random.random() <= end_of_low_intensity_fire_probability.get(new_day, 0.0):
                self.end_of_low_intensity_fire = True

        new_ignition_data = {}
        new_fire_state_data = {}
        self.fire_did_stop = True

        for i, cell in enumerate(cells):
            if cell.isBurningOrWillBurn:
                self.fire_did_stop = False

            ignition_time = cell.ignitionTime
            if cell.fireState == FireState.Burning and time - ignition_time > cell.burnTime:
                new_fire_state_data[i] = FireState.Burnt
                if cell.canSurviveFire and random.random() < self.fire_survival_probability:
                    cell.isFireSurvivor = True

            elif cell.fireState == FireState.Unburnt and time > ignition_time:
                new_fire_state_data[i] = FireState.Burning
                self.burned_cells_in_zone[cell.zoneIdx] = self.burned_cells_in_zone.get(cell.zoneIdx, 0) + 1

                fire_should_spread = not self.end_of_low_intensity_fire or cell.burnIndex != BurnIndex.Low
                if fire_should_spread:
                    neighbors = get_grid_cell_neighbors(
                        cells, i, self.grid_width, self.grid_height, self.neighbors_dist, cell.burnIndex
                    )
                    for n in neighbors:
                        neigh_cell : Cell = cells[n]
                        dist_ft = dist(cell.x,cell.y, neigh_cell.x,neigh_cell.y) * self.cell_size
                        spread_rate = get_fire_spread_rate(cell, neigh_cell, self.wind, self.cell_size)
                        ignition_delta = dist_ft / spread_rate if spread_rate != 0 else float("inf")
                        if neigh_cell.fireState == FireState.Unburnt:
                            current_ignition = new_ignition_data.get(n, neigh_cell.ignitionTime)
                            new_ignition_data[n] = min(ignition_time + ignition_delta, current_ignition)
                            new_burn_time = (new_ignition_data[n] - ignition_time) + self.min_cell_burn_time
                            if new_burn_time < neigh_cell.burnTime:
                                neigh_cell.burnTime = new_burn_time
                            if spread_rate > neigh_cell.spreadRate:
                                neigh_cell.spreadRate = spread_rate

        for i, state in new_fire_state_data.items():
            cells[i].fireState = state
        for i, time_val in new_ignition_data.items():
            cells[i].ignitionTime = time_val
