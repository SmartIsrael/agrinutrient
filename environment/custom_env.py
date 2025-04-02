import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

class HydroponicEnv(gym.Env):
    """
    Custom Gymnasium environment for hydroponics nutrient optimization.
    
    State Space:
        - pH level (float): 0.0 to 14.0
        - EC (float): 0.0 to 5.0 mS/cm
        - Water temperature (float): 10.0 to 40.0 °C
        - Growth stage (int): 0 (seedling), 1 (vegetative), 2 (flowering), 3 (fruiting)
    
    Action Space:
        - Discrete actions for nutrient adjustments:
          0: Decrease nutrient concentration
          1: No change in nutrients
          2: Increase nutrient concentration
        - Discrete actions for pH adjustments:
          3: Decrease pH level
          4: No change in pH
          5: Increase pH level
        - Discrete actions for water cycle time:
          6: Decrease water cycle time
          7: No change in water cycle
          8: Increase water cycle time
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, render_mode=None):
        super(HydroponicEnv, self).__init__()
        
        # State space boundaries
        self.min_ph = 0.0
        self.max_ph = 14.0
        self.min_ec = 0.0
        self.max_ec = 5.0
        self.min_temp = 10.0
        self.max_temp = 40.0
        self.min_growth = 0
        self.max_growth = 3
        
        # Optimal ranges for different growth stages
        # Format: [min_pH, max_pH, min_EC, max_EC, min_temp, max_temp]
        self.optimal_ranges = {
            0: [5.5, 6.5, 0.8, 1.2, 19.0, 24.0],  # Seedling
            1: [5.5, 6.5, 1.2, 2.0, 18.0, 26.0],  # Vegetative
            2: [5.8, 6.8, 1.5, 2.5, 18.0, 26.0],  # Flowering
            3: [6.0, 7.0, 1.8, 3.0, 18.0, 28.0]   # Fruiting
        }
        
        # State variables initialization
        self.ph = None
        self.ec = None
        self.temperature = None
        self.growth_stage = None
        self.water_cycle = None
        self.days_in_stage = None
        self.total_days = None
        
        # Observation space (4 continuous variables)
        self.observation_space = spaces.Box(
            low=np.array([self.min_ph, self.min_ec, self.min_temp, self.min_growth], dtype=np.float32),
            high=np.array([self.max_ph, self.max_ec, self.max_temp, self.max_growth], dtype=np.float32)
        )
        
        # Action space (9 discrete actions)
        self.action_space = spaces.Discrete(9)
        
        self.render_mode = render_mode
        self.renderer = None
        
        # For seeding random number generation
        self.np_random = None
        self.seed()
        
        # Initialize state
        self.reset()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed=None, options=None):
        # Reset the environment state to initial values
        if seed is not None:
            self.seed(seed)
        
        # Initialize with random values within reasonable ranges
        self.ph = self.np_random.uniform(5.0, 7.0)
        self.ec = self.np_random.uniform(0.5, 2.0)
        self.temperature = self.np_random.uniform(18.0, 26.0)
        self.growth_stage = 0  # Start at seedling stage
        self.water_cycle = 4  # Hours between watering cycles
        self.days_in_stage = 0
        self.total_days = 0
        
        # Build observation
        observation = self._get_observation()
        
        info = {}
        return observation, info
    
    def step(self, action):
        # Apply the action to the environment
        self._apply_action(action)
        
        # Environmental dynamics (simulating natural changes)
        self._update_environment()
        
        # Advance time
        self.days_in_stage += 1
        self.total_days += 1
        
        # Check for growth stage transitions
        self._check_growth_stage()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Build observation
        observation = self._get_observation()
        
        # Check if the episode is done
        done = self.total_days >= 120  # End after 120 simulated days
        
        # Additional info
        info = {
            'ph': self.ph,
            'ec': self.ec,
            'temperature': self.temperature,
            'growth_stage': self.growth_stage,
            'days_in_stage': self.days_in_stage,
            'total_days': self.total_days
        }
        
        return observation, reward, done, False, info
    
    def _apply_action(self, action):
        # Nutrient concentration adjustments
        if action == 0:  # Decrease
            self.ec = max(self.min_ec, self.ec - 0.2)
        elif action == 2:  # Increase
            self.ec = min(self.max_ec, self.ec + 0.2)
        
        # pH adjustments
        elif action == 3:  # Decrease
            self.ph = max(self.min_ph, self.ph - 0.3)
        elif action == 5:  # Increase
            self.ph = min(self.max_ph, self.ph + 0.3)
        
        # Water cycle adjustments
        elif action == 6:  # Decrease cycle time (more frequent)
            self.water_cycle = max(1, self.water_cycle - 1)
        elif action == 8:  # Increase cycle time (less frequent)
            self.water_cycle = min(12, self.water_cycle + 1)
    
    def _update_environment(self):
        # Simulate natural environmental fluctuations
        self.ph += self.np_random.normal(0, 0.1)
        self.ph = np.clip(self.ph, self.min_ph, self.max_ph)
        
        self.ec += self.np_random.normal(0, 0.05)
        self.ec = np.clip(self.ec, self.min_ec, self.max_ec)
        
        self.temperature += self.np_random.normal(0, 0.5)
        self.temperature = np.clip(self.temperature, self.min_temp, self.max_temp)
    
    def _check_growth_stage(self):
        # Transition between growth stages based on time
        if self.growth_stage == 0 and self.days_in_stage >= 14:  # Seedling -> Vegetative
            self.growth_stage = 1
            self.days_in_stage = 0
        elif self.growth_stage == 1 and self.days_in_stage >= 30:  # Vegetative -> Flowering
            self.growth_stage = 2
            self.days_in_stage = 0
        elif self.growth_stage == 2 and self.days_in_stage >= 40:  # Flowering -> Fruiting
            self.growth_stage = 3
            self.days_in_stage = 0
    
    def _calculate_reward(self):
        # Get optimal ranges for current growth stage
        opt = self.optimal_ranges[self.growth_stage]
        
        # Base reward
        reward = 0.0
        
        # pH reward component
        if opt[0] <= self.ph <= opt[1]:
            # Optimal pH range
            reward += 1.0
        else:
            # Penalty for being outside optimal range
            distance = min(abs(self.ph - opt[0]), abs(self.ph - opt[1]))
            reward -= distance * 0.5
        
        # EC reward component
        if opt[2] <= self.ec <= opt[3]:
            # Optimal EC range
            reward += 1.0
        else:
            # Penalty for being outside optimal range
            distance = min(abs(self.ec - opt[2]), abs(self.ec - opt[3]))
            reward -= distance * 0.5
        
        # Temperature reward component
        if opt[4] <= self.temperature <= opt[5]:
            # Optimal temperature range
            reward += 1.0
        else:
            # Penalty for being outside optimal range
            distance = min(abs(self.temperature - opt[4]), abs(self.temperature - opt[5]))
            reward -= distance * 0.3
        
        # Water cycle reward component
        optimal_water_cycle = {
            0: 6,  # Seedlings need less frequent watering
            1: 4,  # Vegetative stage needs moderate watering
            2: 3,  # Flowering stage needs more frequent watering
            3: 2   # Fruiting stage needs most frequent watering
        }
        
        water_cycle_diff = abs(self.water_cycle - optimal_water_cycle[self.growth_stage])
        reward -= water_cycle_diff * 0.2
        
        return reward
    
    def _get_observation(self):
        return np.array([
            self.ph,
            self.ec,
            self.temperature,
            float(self.growth_stage)
        ], dtype=np.float32)
    
    def render(self):
        if self.render_mode is None:
            return
        
        if self.renderer is None:
            from environment.rendering import HydroponicRenderer
            self.renderer = HydroponicRenderer(self.render_mode)
        
        return self.renderer.render(
            self.ph, 
            self.ec, 
            self.temperature, 
            self.growth_stage,
            self.water_cycle
        )
    
    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None