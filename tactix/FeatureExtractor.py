import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np

class FireEnvCNN(BaseFeaturesExtractor):
    """
    Enhanced CNN feature extractor with multi-scale processing and attention mechanism
    for better understanding of fire dynamics and spatial relationships.
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        # Extract shapes from observation space
        cells_shape = observation_space['cells'].shape  # (160, 240)
        
        # Multi-scale CNN for different receptive fields
        self.multi_scale_cnn = nn.ModuleList([
            # Small receptive field - local fire details
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ),
            # Medium receptive field - fire clusters
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU()
            ),
            # Large receptive field - global fire patterns
            nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=11, stride=4, padding=5),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(16),
                nn.ReLU()
            )
        ])
        
        # Calculate output dimensions for each scale
        test_tensor = th.zeros(1, 1, cells_shape[0], cells_shape[1])
        scale_outputs = []
        with th.no_grad():
            for scale_cnn in self.multi_scale_cnn:
                output = scale_cnn(test_tensor)
                scale_outputs.append(output.shape)
        
        # Spatial attention mechanism for fire hotspots
        self.attention_conv = nn.Sequential(
            nn.Conv2d(64 + 32 + 16, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Fire state decomposition layers
        self.fire_state_conv = nn.Conv2d(1, 3, kernel_size=1)  # Split into Unburnt, Burning, Burnt
        self.burn_index_conv = nn.Conv2d(1, 3, kernel_size=1)  # Split into Low, Medium, High
        
        # Flatten layer for concatenation
        self.spatial_flatten = nn.Flatten()
        
        # Calculate total flattened dimension
        combined_channels = 64 + 32 + 16  # Sum of channels from all scales
        total_spatial_elements = scale_outputs[0][2] * scale_outputs[0][3]  # Assuming same spatial size after alignment
        flattened_dim = combined_channels * total_spatial_elements
        
        # Helicopter position encoding with relative positioning
        self.coord_net = nn.Sequential(
            nn.Linear(4, 32),  # x, y, rel_to_center_x, rel_to_center_y
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # On-fire flag processing
        self.fire_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )
        
        # Combine all features
        combined_dim = flattened_dim + 64 + 16
        
        # Final processing with residual connections
        self.fc1 = nn.Linear(combined_dim, features_dim)
        self.fc2 = nn.Linear(features_dim, features_dim)
        self.fc3 = nn.Linear(features_dim, features_dim)
        
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(features_dim)
    
    def forward(self, observations):
        # Process cells - add channel dimension
        cells = observations['cells'].float().unsqueeze(1)  # [B, 1, H, W]
        
        # Decompose fire states and burn indices
        fire_states_raw = cells // 3
        burn_indices_raw = cells % 3
        
        # Process through multi-scale CNNs
        scale_features = []
        for scale_cnn in self.multi_scale_cnn:
            scale_output = scale_cnn(cells)
            scale_features.append(scale_output)
        
        # Align spatial dimensions (upsample smaller scales to match the largest)
        target_h, target_w = scale_features[0].shape[2], scale_features[0].shape[3]
        aligned_features = [scale_features[0]]
        
        for i in range(1, len(scale_features)):
            if scale_features[i].shape[2:] != (target_h, target_w):
                upsampled = nn.functional.interpolate(
                    scale_features[i], 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False
                )
                aligned_features.append(upsampled)
            else:
                aligned_features.append(scale_features[i])
        
        # Concatenate aligned features
        combined_features = th.cat(aligned_features, dim=1)
        
        # Apply spatial attention
        attention_map = self.attention_conv(combined_features)
        attended_features = combined_features * attention_map + combined_features
        
        # Flatten the features
        flattened_features = self.spatial_flatten(attended_features)
        
        # Process helicopter coordinates with relative positioning
        coords = observations['helicopter_coord']
        center_x, center_y = 120, 80  # Grid center
        rel_coords = th.stack([
            coords[:, 0],  # absolute x
            coords[:, 1],  # absolute y
            coords[:, 0] - center_x,  # relative to center x
            coords[:, 1] - center_y   # relative to center y
        ], dim=1)
        coord_features = self.coord_net(rel_coords)
        
        # Process on_fire flag
        fire_features = self.fire_net(observations['on_fire'])
        
        # Handle shape mismatch if fire_features has an extra dimension
        if len(fire_features.shape) == 3:
            fire_features = fire_features.squeeze(1)
        
        # Combine all features
        combined = th.cat([flattened_features, coord_features, fire_features], dim=1)
        
        # Process through fully connected layers with residual connections
        x = self.fc1(combined)
        x = self.relu(x)
        
        identity = x
        x = self.fc2(x)
        x = self.relu(x)
        x = x + identity  # Residual connection
        
        x = self.layer_norm(x)
        
        identity = x
        x = self.fc3(x)
        x = self.relu(x)
        x = x + identity  # Residual connection
        
        return x
    
    def get_attention_map(self, observations):
        """Get the attention map for visualization purposes"""
        cells = observations['cells'].float().unsqueeze(1)
        
        scale_features = []
        for scale_cnn in self.multi_scale_cnn:
            scale_output = scale_cnn(cells)
            scale_features.append(scale_output)
        
        # Align features as in forward pass
        target_h, target_w = scale_features[0].shape[2], scale_features[0].shape[3]
        aligned_features = [scale_features[0]]
        
        for i in range(1, len(scale_features)):
            if scale_features[i].shape[2:] != (target_h, target_w):
                upsampled = nn.functional.interpolate(
                    scale_features[i], 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False
                )
                aligned_features.append(upsampled)
            else:
                aligned_features.append(scale_features[i])
        
        combined_features = th.cat(aligned_features, dim=1)
        attention_map = self.attention_conv(combined_features)
        
        return attention_map