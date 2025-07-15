import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from collections import deque

class FireEnvLSTMCNN(BaseFeaturesExtractor):
    """
    Enhanced CNN feature extractor with LSTM for temporal understanding.
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        self.n_envs = 8
        
        # Extract shapes from observation space
        cells_shape = observation_space['cells'].shape  # (4, 160, 240) with frame stacking

        self.hidden_size = 256   # must match LSTM hidden_size
        self.n_layers = 2        # must match LSTM num_layers
        self.device = th.device("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")
        
        # Determine the number of channels (frames)
        if len(cells_shape) == 3:
            channels = cells_shape[0]  # 4 frames
            height = cells_shape[1]    # 160
            width = cells_shape[2]     # 240
        else:
            channels = 1
            height = cells_shape[0]    # 160 (fallback for non-stacked)
            width = cells_shape[1]     # 240
        
        # Multi-scale CNN for different receptive fields
        self.multi_scale_cnn = nn.ModuleList([
            # Small receptive field - local fire details
            nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
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
                nn.Conv2d(channels, 16, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU()
            ),
            # Large receptive field - global fire patterns
            nn.Sequential(
                nn.Conv2d(channels, 8, kernel_size=11, stride=4, padding=5),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(16),
                nn.ReLU()
            )
        ])
        
        # Calculate output dimensions for each scale
        test_tensor = th.zeros(1, channels, height, width)
        
        scale_outputs = []
        with th.no_grad():
            for scale_cnn in self.multi_scale_cnn:
                output = scale_cnn(test_tensor)
                scale_outputs.append(output.shape)
        
        # Spatial attention mechanism
        self.attention_conv = nn.Sequential(
            nn.Conv2d(64 + 32 + 16, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Flatten layer for concatenation
        self.spatial_flatten = nn.Flatten()
        
        # Calculate total flattened dimension
        combined_channels = 64 + 32 + 16
        total_spatial_elements = scale_outputs[0][2] * scale_outputs[0][3]
        flattened_dim = combined_channels * total_spatial_elements
        
        # LSTM layer for temporal understanding
        self.lstm = nn.LSTM(
            input_size=flattened_dim,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Helicopter position encoding with relative positioning
        self.coord_net = nn.Sequential(
            nn.Linear(4, 32),
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
        combined_dim = 256 + 64 + 16  # LSTM output + coord + fire
        
        # Final processing with residual connections
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim)
        )
        
        # Initialize hidden states
        self.hidden = None

    
    def reset_hidden(self, env_indices=None):
        """
        Reset hidden states of LSTM for specified env indices.
        If env_indices is None, reset all.
        """
        if self.hidden is None:
            return
        
        h, c = self.hidden  # each of shape (num_layers, batch_size, hidden_size)

        if env_indices is None:
            h.zero_()
            c.zero_()
        else:
            for idx in env_indices:
                h[:, idx, :].zero_()
                c[:, idx, :].zero_()

        self.hidden = (h, c)

    def forward(self, observations):
        # Process cells - expecting shape (batch, 4, 160, 240)
        cells = observations['cells'].float()
        cells = th.clamp(cells, min=-1, max=8)
        cells = (cells + 1) / 9.0  # maps [-1, 8] → [0.0, 1.0]

        if cells.dim() == 3:  # ensure batch and channel dims
            cells = cells.unsqueeze(1)
        
        # Ensure correct shape
        if len(cells.shape) == 3:  # If missing batch dimension
            cells = cells.unsqueeze(0)
        
        # Process through multi-scale CNNs
        scale_features = []
        for scale_cnn in self.multi_scale_cnn:
            scale_output = scale_cnn(cells)
            scale_features.append(scale_output)
        
        # Align spatial dimensions and concatenate
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
        
        # Apply spatial attention
        attention_map = self.attention_conv(combined_features)
        attended_features = combined_features * attention_map + combined_features
        
        # Flatten the features
        flattened_features = self.spatial_flatten(attended_features)
        
        # LSTM processing
        batch_size = flattened_features.size(0)
        lstm_input = flattened_features.unsqueeze(1)  # Add sequence dimension
        
        # Initialize hidden states if None or batch size changed
        if self.hidden is None or self.hidden[0].size(1) != batch_size:
            h0 = th.zeros(self.n_layers, batch_size, self.hidden_size, device=lstm_input.device)
            c0 = th.zeros(self.n_layers, batch_size, self.hidden_size, device=lstm_input.device)
            self.hidden = (h0, c0)

        lstm_out, self.hidden = self.lstm(lstm_input, self.hidden)
        
        # Detach hidden states to avoid backprop through time beyond current step
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        
        lstm_features = lstm_out.squeeze(1)
        
        # Process helicopter coordinates
        coords = observations['helicopter_coord'].float()
        center_x, center_y = 120, 80
        rel_coords = th.stack([
            coords[:, 0],
            coords[:, 1],
            coords[:, 0] - center_x,
            coords[:, 1] - center_y
        ], dim=1)
        coord_features = self.coord_net(rel_coords)
        
        # Process on_fire flag
        fire_features = self.fire_net(observations['on_fire'].float())
        if len(fire_features.shape) == 3:
            fire_features = fire_features.squeeze(1)
        
        # Combine all features
        combined = th.cat([lstm_features, coord_features, fire_features], dim=1)
        
        # Final processing
        return self.fc(combined)

class FireEnvLSTMPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# class FireEnvLSTMPolicy(ActorCriticPolicy):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, 
#                         features_extractor_class=FireEnvLSTMCNN,
#                         features_extractor_kwargs=dict(features_dim=512),
#                         **kwargs)