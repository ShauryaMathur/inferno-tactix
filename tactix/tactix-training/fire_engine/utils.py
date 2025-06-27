from environment.helper import get_grid_index_for_location
from environment.cell import Cell

direct_neighbours = [
    {"x": -1, "y": 0},
    {"x": 1, "y": 0},
    {"x": 0, "y": -1},
    {"x": 0, "y": 1}
]

def dist(x0, y0, x1, y1):
    return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5

def within_dist(x0, y0, x1, y1, max_dist):
    return (x0 - x1) ** 2 + (y0 - y1) ** 2 <= max_dist ** 2

def get_grid_cell_neighbors(cells : list[Cell], i, width, height, neighbors_dist, burn_index):
    neighbours = []
    queue = []
    processed = {}
    
    x0 = i % width
    y0 = i // width
    any_nonburnable_cells = False

    queue.append(i)
    processed[i] = True

    while queue:
        j = queue.pop(0)
        x1 = j % width
        y1 = j // width

        for diff in direct_neighbours:  # assumed list of {'x': dx, 'y': dy}
            nx = x1 + diff["x"]
            ny = y1 + diff["y"]
            if 0 <= nx < width and 0 <= ny < height:
                n_idx = get_grid_index_for_location(nx, ny, width)
                if not processed.get(n_idx) and within_dist(x0, y0, nx, ny, neighbors_dist):
                    if not cells[n_idx].isBurnableForBI(burn_index):
                        any_nonburnable_cells = True
                    elif not any_nonburnable_cells or not nonburnable_cell_between(cells, width, nx, ny, x0, y0, burn_index):
                        neighbours.append(n_idx)
                        queue.append(n_idx)
                    processed[n_idx] = True

    return neighbours

def nonburnable_cell_between(cells : list[Cell], width, x0, y0, x1, y1, burn_index):
    result = False

    def callback(x, y, _):
        nonlocal result
        idx = get_grid_index_for_location(x, y, width)
        if not cells[idx].isBurnableForBI(burn_index):
            result = True

    for_each_point_between(x0, y0, x1, y1, callback)
    return result

def for_each_point_between(x0, y0, x1, y1, callback):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    idx = 0

    while True:
        callback(x0, y0, idx)
        idx += 1
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy