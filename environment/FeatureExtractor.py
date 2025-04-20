import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class FireEnvCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for FireEnv observations.
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Extract shapes from observation space
        cells_shape = observation_space['cells'].shape  # (160, 240)
        
        # CNN for processing the grid cells (fire intensities)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=4, padding=1),  # Larger stride for efficiency
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()  # Flatten the output for the fully connected layers
        )
        
        # Calculate CNN output size - simplify by testing with a sample input
        test_tensor = th.zeros(1, 1, cells_shape[0], cells_shape[1])
        with th.no_grad():
            cnn_output = self.cnn(test_tensor)
        cnn_output_dim = cnn_output.shape[1]
        
        # Network for helicopter coordinates
        self.coord_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )
        
        # Network for on_fire flag (one-hot encoded by SB3)
        self.fire_net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU()
        )
        
        # Combine all features
        combined_dim = cnn_output_dim + 16 + 8
        
        # Final fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        # Process cells - add channel dimension
        cells = observations['cells'].float().unsqueeze(1)  # [B, 1, H, W]
        cells_features = self.cnn(cells)
        
        # Debug: Print the shape of cells_features
        # print(f"cells_features shape: {cells_features.shape}")
        
        # Flatten the cells_features for concatenation
        cells_features = cells_features.view(cells_features.size(0), -1)
        
        # Process helicopter coordinates
        coord_features = self.coord_net(observations['helicopter_coord'])
        
        # Debug: Print the shape of coord_features
        # print(f"coord_features shape: {coord_features.shape}")
        
        # Process on_fire flag (already one-hot encoded by SB3)
        fire_features = self.fire_net(observations['on_fire'])

        fire_features = fire_features.squeeze(1)  # This will change the shape from [10, 1, 8] to [10, 8]
        
        # Debug: Print the shape of fire_features
        # print(f"fire_features shape: {fire_features.shape}")
        
        # Concatenate features
        try:
            combined_features = th.cat([cells_features, coord_features, fire_features], dim=1)
        except RuntimeError as e:
            print(f"Error in concatenation: {e}")
            print(f"cells_features shape: {cells_features.shape}")
            print(f"coord_features shape: {coord_features.shape}")
            print(f"fire_features shape: {fire_features.shape}")
            raise e
        
        # Final processing
        return self.fc(combined_features)
