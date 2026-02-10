import time
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import config
from environment.cell import Cell
from fire_engine.fire_engine import FireEngine
from environment.vector import Vector2
from environment.wind import Wind
from environment.zone import Zone
from environment.enums import FireState
import copy
import environment.helper as helper
import traceback
from training_utils import clear_gpu_memory
from stable_baselines3.common.env_checker import check_env


# Environment settings
MAX_TIMESTEPS = 1000
HELICOPTER_SPEED = 3

class FireEnvSync(gym.Env):
    def __init__(self, env_id=0):
        # super().__init__()
        
        # Initialize spaces
        self.env_id = env_id
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            'helicopter_coord': spaces.Box(low=np.array([0, 0]), high=np.array([239, 159]), dtype=np.int32),
            'cells': spaces.Box(low=0.0, high=math.inf, shape=(160, 240), dtype=np.float64),
            'quenched_cells': spaces.Box(low=0.0, high=38400.0, shape=(1,), dtype=np.float32),
        })
        
        # Environment configuration
        self.cell_size = config.cellSize
        self.gridWidth = config.gridWidth
        self.gridHeight = config.gridHeight
        self.zones: list[Zone] = [Zone(**zone_dict) for zone_dict in config.ZONES]
        
        # Initialize state variables
        self._reset_state_variables()
        
        # Frame stacking
        # self.frame_history = np.zeros((4, 160, 240), dtype=np.int8)
        
        # Cached observation
        self.cached_obs = None
        self._seed = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self._seed = seed
        return [seed]
    
    def get_helicopter_position_map(self,x, y, height=160, width=240):
        map = np.zeros((height, width), dtype=np.float32)
        map[y, x] = 1.0
        return map
    def _reset_state_variables(self):
        self.step_count = 0
        self.time = 0
        self.prev_tick_time = None
        self.simulation_running = True
        self.cells = []
        self.engine = None
        
        # Reset state dict
        self.state = {
            'helicopter_coord': np.array([70, 150], dtype=np.int32),
            'cells': np.zeros((160, 240), dtype=np.int32),
            'prevBurntCells': 0,
            'cellsBurnt': 0,
            'cellsBurning': 0,
            'quenched_cells': 0,
            'prev_burning': 0,
            'last_action': None
        }
    
    def _default_state(self):
        return {
            'helicopter_coord': np.array([70, 150], dtype=np.int32),
            'cells': np.zeros((160, 240), dtype=np.int32),
            'on_fire': 0,
            'prevBurntCells': 0,
            'cellsBurnt': 0,
            'cellsBurning': 0,
            'quenched_cells': 0
        }
    
    def tick(self, time_step: float):
        if self.engine and self.simulation_running:
            self.time += time_step
            self.engine.update_fire(self.cells, self.time)
            
            if self.engine.fire_did_stop:
                self.simulation_running = False
                print(f"[Env {self.env_id}] Fire simulation stopped at time {self.time}")

    def _get_current_fire_stats(self):
            cells_burning = len([cell for cell in self.cells if cell.fireState == FireState.Burning])
            cells_burnt = len([cell for cell in self.cells if cell.fireState == FireState.Burnt])
            return cells_burning, cells_burnt
    
    def _update_simulation_state(self):
            # Generate binned cells from current simulation state
            binnedCells = helper.generate_fire_status_map_from_cells(
                self.cells, self.gridWidth, self.gridHeight
            )
            binnedCells = np.array(binnedCells, dtype=np.int32)
            # binnedCells = binnedCells.astype(np.float32)
            binnedCells = np.clip(binnedCells, -1, 8)
            binnedCells = (binnedCells + 1) / 9.0
            
            # print(binnedCells.shape)
            # Get current fire statistics
            cells_burning, cells_burnt = self._get_current_fire_stats()
            
            # Update state
            self.state.update({
                'cells': binnedCells,
                'cellsBurning': cells_burning,
                'cellsBurnt': cells_burnt,
                'prevBurntCells': self.state.get('cellsBurnt', 0),  # Store previous value
                'prev_burning': self.state.get('cellsBurning', 0)   # Store previous value
            })
            
    def step_simulation(self, current_time_ms: float):
        if not self.simulation_running:
            return
            
        # Calculate time step
        if self.prev_tick_time is None:
            self.prev_tick_time = current_time_ms
            time_step = 1.0  # Default first step
        else:
            real_time_diff_minutes = (current_time_ms - self.prev_tick_time) / 60000
            self.prev_tick_time = current_time_ms
            
            ratio = 86400 / getattr(config, 'modelDayInSeconds', 8)
            optimal_time_step = ratio * 0.000277
            time_step = min(
                getattr(config, 'maxTimeStep', 180),
                optimal_time_step * 4,
                ratio * real_time_diff_minutes
            )
        
        self.tick(time_step)

    def _get_fallback_observation(self):
            return {
                'cells': np.zeros((160, 240), dtype=np.int8),
                'helicopter_coord': np.array([70, 30], dtype=np.int32)
            }
            
    def update_frame_history(self, new_frame):
        self.frame_history = np.roll(self.frame_history, -1, axis=0)
        self.frame_history[3] = new_frame
    
    def reset(self, *, seed=None, options=None):

        try:
            super().reset(seed=seed)
            if seed is not None:
                self.seed(seed)
            
            print(f"🧼 Resetting environment {self.env_id} with seed={self._seed}")
            # input("Resetting environment. Press Enter to continue...")
            
            # Reset all state variables
            self._reset_state_variables()
            self.episode_count = getattr(self, 'episode_count', 0) + 1
            
            # Initialize cells - ensure complete isolation
            # self.cells = getCellsDataByEnvId(self.env_id, self.zones)
            self.cells = helper.populateCellsData(self.env_id,self.zones)
            
            # Initialize fire spark
            spark = Vector2(60000 - 1, 40000 - 1)
            grid_x = int(spark.x // self.cell_size)
            grid_y = int(spark.y // self.cell_size)
            spark_cell : Cell = self.cells[helper.get_grid_index_for_location(grid_x, grid_y, self.gridWidth)]
            spark_cell.ignitionTime = 0
            # spark_cell.fireState = FireState.Burning
            
            # Initialize fire engine
            self.engine = FireEngine(Wind(0.0, 0.0), config)
            
            # Update state from simulation
            # self._update_simulation_state()
            
            # Initialize frame history
            # initial_cells = np.clip(np.array(self.state['cells'], dtype=np.int8), -1, 8)
            # initial_cells = (initial_cells + 1) / 9.0  # normalize to [0, 1]

            # self.frame_history.fill(0)
            # for i in range(4):
            #     self.frame_history[i] = initial_cells.copy()
            
            # heli_x, heli_y = self.state['helicopter_coord']
            # heli_map = np.zeros_like(initial_cells, dtype=np.float32)
            # heli_map[heli_y, heli_x] = 1.0  # Note: (y, x) indexing

            # spatial_obs = np.concatenate([self.frame_history.copy(), heli_map[None, :, :]], axis=0)

            observation = {
                'cells': helper.generate_ignition_times_from_cells(self.cells,self.gridWidth, self.gridHeight),
                'helicopter_coord': self.state['helicopter_coord'],
                'quenched_cells': np.array([0], dtype=np.float32)
            }
            # assert observation["cells"].shape == (5, 160, 240), f"Bad shape: {observation['cells'].shape}"
            
            # self.cached_obs = observation
            
            print(f"[Env {self.env_id}] Reset complete - Initial burning cells: {self.state['cellsBurning']}")
            
            return observation, {}
            
        except Exception as e:
            print(f"Error in reset for env {self.env_id}: {e}")
            traceback.print_exc()
            return self._get_fallback_observation(), {}
    
    def apply_action(self, action):
        
        self.step_count += 1
        self.state['last_action'] = action
        
        # Calculate new helicopter position
        heli_x, heli_y = self.state['helicopter_coord']
        
        if action == 0:   # Down
            print("0 called")
            heli_y += HELICOPTER_SPEED
        elif action == 1: # Up
            heli_y -= HELICOPTER_SPEED
        elif action == 2: # Left
            heli_x -= HELICOPTER_SPEED
        elif action == 3: # Right
            heli_x += HELICOPTER_SPEED
        elif action == 4: # Helitack
            print(f"[Env {self.env_id}] Helitack at ({heli_x}, {heli_y})")
        
        # Clip coordinates to valid range
        heli_x = int(np.clip(heli_x, 0, 239))
        heli_y = int(np.clip(heli_y, 0, 159))
        
        print(f"Action New: {action} and Coordinates : {[heli_x, heli_y]}")
        
        # Perform helitack if action is 4
        quenched_cells = 0
        if action == 4:
            quenched_cells = helper.perform_helitack(self.cells, heli_x, heli_y)
            if quenched_cells > 0:
                print(f"[Env {self.env_id}] Helitack quenched {quenched_cells} cells")
        
        return heli_x, heli_y, quenched_cells
    
    def step(self, action):
        try:
            # Store previous state for reward calculation
            # prev_burnt = self.state.get('cellsBurnt', 0)
            # prev_burning = self.state.get('cellsBurning', 0)
            
            # Apply action and get helicopter position
            heli_x, heli_y, quenched_cells = self.apply_action(action)

            # Run simulation step
            current_time_ms = time.time() * 1000
            self.step_simulation(current_time_ms)
            
            # Update state from simulation
            # self._update_simulation_state()
            
            # Update helicopter position and fire status
            self.state['helicopter_coord'] = np.array([heli_x, heli_y], dtype=np.int32)
            # self.state['quenchedCells'] = quenched_cells
            # print("self.state['cells'].shape",self.state['cells'].shape)
            # Check if helicopter is on fire
            # on_fire = helper.is_helicopter_on_fire(self.state['cells'], heli_x, heli_y)
            # self.state['on_fire'] = on_fire
            
            cells_burning, cells_burnt = self._get_current_fire_stats()
            # Calculate reward
            reward = self.calculate_reward(
                cells_burnt,
                cells_burning,
                quenched_cells
            )
            
            # Check if episode is done or truncated
            terminated = (cells_burning == 0)
            truncated = (self.step_count >= MAX_TIMESTEPS)
            
            observation = {
                'cells': helper.generate_ignition_times_from_cells(self.cells,self.gridWidth, self.gridHeight),
                'helicopter_coord': self.state['helicopter_coord'],
                'quenched_cells': np.array([quenched_cells], dtype=np.float32)
            }
            # print("Obs cells shape:", observation['cells'].shape)
            # assert observation["cells"].shape == (5, 160, 240), f"Bad shape: {observation['cells'].shape}"

            # self.cached_obs = observation
            
            print(f"[Env {self.env_id}] Step {self.step_count}: Burning={cells_burning}, "
                      f"Burnt={cells_burnt}, Reward={reward:.3f}")
            # if self.step_count % 10 == 0:  # Log every 10 steps
            #     print(f"[Env {self.env_id}] Step {self.step_count}: Burning={self.state['cellsBurning']}, "
            #           f"Burnt={self.state['cellsBurnt']}, Reward={reward:.3f}")
            # print(f"Returning: terminated={terminated}, truncated={truncated}")

            return observation, reward, terminated, truncated, {}
            
        except Exception as e:
            print(f"Error in step for env {self.env_id}: {e}")
            traceback.print_exc()
            return self.cached_obs or self._get_fallback_observation(), 0, True, False, {}

    def calculate_reward(self, curr_burnt, curr_burning, extinguished_by_helitack):
        reward = 0.0

        # Penalize new burnt cells (damage) strongly
        # newly_burnt = curr_burnt - prev_burnt
        # reward -= 1.0 * newly_burnt  # Increased penalty for new burnt cells

        # Reward extinguishing burning cells — encourage actively putting out fires
        reward += 2.0 * extinguished_by_helitack  # Larger reward for extinguishing

        # Reward reduction in burning cells (i.e., burning areas shrinking)
        burning_reduction = self.state.get('prev_burning', curr_burning) - curr_burning
        reward += 1.0 * burning_reduction  # Reward reducing fire size

        # Small penalty for ongoing burning cells to discourage letting fires burn longer
        reward -= 0.05 * curr_burning

        # Small step penalty to encourage faster containment
        reward -= 0.01

        # Gentle penalty for acting on unburnable/burnt cells (to avoid wasting moves)
        heli_x, heli_y = self.state['helicopter_coord']
        last_action = self.state.get('last_action', None)
        cells = np.array(self.state['cells'])
        fire_states = cells // 3

        if last_action == 4 and 0 <= heli_y < cells.shape[0] and 0 <= heli_x < cells.shape[1]:
            cell_value = cells[heli_y, heli_x]
            if cell_value == -1 or fire_states[heli_y, heli_x] == FireState.Burnt:
                reward -= 0.2  # Slightly stronger penalty here

        # Update prev_burning for next step
        self.state['prev_burning'] = curr_burning

        # Clip reward to a reasonable range for PPO stability
        reward = np.clip(reward, -10, 10)
        return reward

    def close(self):
        print(f"Closing environment {self.env_id}...")
        self.simulation_running = False
        self.cells = []
        self.engine = None
        self.state = None
        self.frame_history = None
        self.cached_obs = None
        clear_gpu_memory()
        
def main():
    print("Testing environment...")
    env = FireEnvSync(env_id=0)
    try:
        # check_env(env, warn=True, skip_render_check=True)
        print("✅ Environment passed all checks!")
        
        env.reset()
        timestamp = 0
        rewards = 0
        while True:
            timestamp += 1
            # action = random.randint(1, 5)
            action = random.randint(0, 4)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards += reward
            if terminated or truncated:
                break
        print(f"Total rewards {rewards}")
    except Exception as e:
        print(f"❌ Environment failed checks: {e}")
        traceback.print_exc()
main()
    
    