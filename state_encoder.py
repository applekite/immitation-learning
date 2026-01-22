import sys
import numpy as np
import torch
from scene_reasoning.encoder2d.r18s_encoder import ResNet18Encoder
from scene_reasoning.encoder3d.cnn3d_encoder import CNN3D
from scene_reasoning.encoder3d.cnntransformer_encoder import CNNTransformer

class EncodeState():
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.conv_encoder = ResNet18Encoder(self.latent_dim).to(self.device)
            try:
                self.conv_encoder.load()
                print('Encoder model loaded successfully.')
            except (FileNotFoundError, RuntimeError) as e:
                print(f'Warning: Could not load pre-trained encoder model: {e}')
                print('Using randomly initialized encoder weights.')
            self.conv_encoder.eval()

            for params in self.conv_encoder.parameters():
                params.requires_grad = False
        except Exception as e:
            print(f'Encoder could not be initialized: {e}')
            sys.exit()
    
    def process(self, observation):
        """
        Process observation: encode image and concatenate with navigation data.
        
        Args:
            observation: List containing [image_obs, navigation_obs]
                - image_obs: numpy array of shape (height, width, channels) or (channels, height, width)
                - navigation_obs: numpy array of navigation features
        
        Returns:
            Concatenated tensor of encoded image and navigation features
        """
        # Validate observation format
        if not isinstance(observation, (list, tuple)) or len(observation) < 2:
            raise ValueError(f"Observation must be a list/tuple with at least 2 elements. Got: {type(observation)}, length: {len(observation) if hasattr(observation, '__len__') else 'N/A'}")
        
        image_obs = observation[0]
        navigation_obs = observation[1]
        
        # Debug: Check observation format
        if image_obs is None:
            raise ValueError("Image observation is None!")
        
        # Debug: Print observation types and shapes (only first few times to avoid spam)
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        if self._debug_count < 3:
            print(f"[DEBUG] Observation type: {type(observation)}, length: {len(observation)}")
            print(f"[DEBUG] Image obs type: {type(image_obs)}, shape: {getattr(image_obs, 'shape', 'no shape') if hasattr(image_obs, 'shape') else type(image_obs)}")
            if hasattr(image_obs, '__len__') and len(image_obs) < 10:
                print(f"[DEBUG] Image obs value (first few): {image_obs[:min(5, len(image_obs))] if hasattr(image_obs, '__getitem__') else 'cannot index'}")
            print(f"[DEBUG] Navigation obs type: {type(navigation_obs)}, shape: {getattr(navigation_obs, 'shape', 'no shape') if hasattr(navigation_obs, 'shape') else type(navigation_obs)}")
            self._debug_count += 1
        
        # Handle scalar or wrong type - this shouldn't happen but let's catch it
        if isinstance(image_obs, (int, float, bool)):
            raise ValueError(f"Image observation is a scalar ({type(image_obs).__name__}: {image_obs}). "
                           f"Expected numpy array or tensor of shape (H, W, C). "
                           f"This suggests the environment is not returning images correctly.")
        
        # Convert numpy array to tensor
        if isinstance(image_obs, np.ndarray):
            # Check if it's a valid image array
            if image_obs.size == 0:
                raise ValueError(f"Image observation is empty numpy array with shape {image_obs.shape}")
            if image_obs.ndim == 0:
                raise ValueError(f"Image observation is a 0D numpy array (scalar) with value {image_obs.item()}. "
                               f"Expected 2D or 3D array. Check environment reset/step functions.")
            image_obs_tensor = torch.from_numpy(image_obs).float().to(self.device)
        elif isinstance(image_obs, torch.Tensor):
            image_obs_tensor = image_obs.float().to(self.device)
        else:
            # Try to convert to numpy first, then tensor
            try:
                image_obs_np = np.array(image_obs)
                if image_obs_np.size == 0:
                    raise ValueError(f"Image observation converts to empty array. Original type: {type(image_obs)}, value: {image_obs}")
                if image_obs_np.ndim == 0:
                    raise ValueError(f"Image observation converts to 0D array (scalar) with value {image_obs_np.item()}. "
                                   f"Original type: {type(image_obs)}, value: {image_obs}. "
                                   f"Expected image array of shape (H, W, C).")
                image_obs_tensor = torch.from_numpy(image_obs_np).float().to(self.device)
            except Exception as e:
                raise ValueError(f"Cannot convert image observation to tensor. Type: {type(image_obs)}, Value: {image_obs}, Error: {e}")
        
        image_obs = image_obs_tensor
        
        # Check if tensor is empty or scalar
        if image_obs.numel() == 0:
            raise ValueError(f"Image observation tensor is empty with shape {image_obs.shape}")
        if image_obs.dim() == 0:
            raise ValueError(f"Image observation is a scalar (0D tensor) with value {image_obs.item()}. "
                           f"This suggests the observation format is incorrect. "
                           f"Expected image array of shape (H, W, C), got scalar. "
                           f"Check that env.reset() and env.step() return [image_array, navigation_array].")
        
        # Debug: Print original shape
        original_shape = image_obs.shape
        original_dims = image_obs.dim()
        
        # Check and fix image shape
        if image_obs.dim() == 1:
            # If 1D, try to reshape (this shouldn't happen normally)
            print(f"Warning: Image observation is 1D with shape {image_obs.shape}. Attempting to reshape.")
            # Try to infer shape - assuming it's flattened (height * width * channels)
            # Default semantic segmentation image: 128x256x3 = 98304
            total_elements = image_obs.shape[0]
            if total_elements == 98304:  # 128 * 256 * 3
                image_obs = image_obs.view(128, 256, 3)
            elif total_elements == 32768:  # 128 * 256 * 1
                image_obs = image_obs.view(128, 256, 1)
                # Repeat to 3 channels
                image_obs = image_obs.repeat(1, 1, 3)
            elif total_elements == 38400:  # Legacy: 80 * 160 * 3 (for backward compatibility)
                image_obs = image_obs.view(80, 160, 3)
            elif total_elements == 19200:  # Legacy: 80 * 160 * 1 (for backward compatibility)
                image_obs = image_obs.view(80, 160, 1)
                image_obs = image_obs.repeat(1, 1, 3)
            else:
                raise ValueError(f"Cannot reshape 1D image observation of size {total_elements}. "
                               f"Expected 98304 (128x256x3), 32768 (128x256x1), 38400 (80x160x3), or 19200 (80x160x1)")
        
        # Ensure image is 3D (height, width, channels) before adding batch dimension
        if image_obs.dim() == 2:
            # If 2D, add channel dimension (grayscale)
            image_obs = image_obs.unsqueeze(-1)
            # If single channel, repeat to 3 channels for RGB-like format
            if image_obs.shape[-1] == 1:
                image_obs = image_obs.repeat(1, 1, 3)
        elif image_obs.dim() == 3:
            # Already in (H, W, C) format - good
            # Ensure it has 3 channels
            if image_obs.shape[2] == 1:
                image_obs = image_obs.repeat(1, 1, 3)
            elif image_obs.shape[2] != 3:
                print(f"Warning: Image has {image_obs.shape[2]} channels, expected 3. Using first 3 or repeating.")
                if image_obs.shape[2] > 3:
                    image_obs = image_obs[:, :, :3]
                else:
                    # Repeat channels to get 3
                    repeats = 3 // image_obs.shape[2] + 1
                    image_obs = image_obs.repeat(1, 1, repeats)[:, :, :3]
        elif image_obs.dim() == 4:
            # Already has batch dimension - might be (1, H, W, C) or (1, C, H, W)
            if image_obs.shape[0] == 1:
                # Remove batch dimension, will add it back
                image_obs = image_obs.squeeze(0)
            else:
                raise ValueError(f"Unexpected 4D image shape: {image_obs.shape}. Expected batch size 1.")
        else:
            raise ValueError(f"Unexpected image observation shape: {image_obs.shape} (dims: {image_obs.dim()}). "
                           f"Expected (H, W, C) format.")
        
        # At this point, image_obs should be 3D: (H, W, C)
        if image_obs.dim() != 3:
            raise ValueError(f"After shape fixing, image should be 3D but got {image_obs.dim()}D with shape {image_obs.shape}")
        
        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        image_obs = image_obs.unsqueeze(0)
        
        # Permute to (batch, channels, height, width) format expected by encoder
        # From (1, H, W, C) to (1, C, H, W)
        if image_obs.dim() == 4:
            image_obs = image_obs.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Image observation has wrong number of dimensions after unsqueeze: {image_obs.dim()}. "
                           f"Expected 4, got shape {image_obs.shape}. Original shape was {original_shape} (dims: {original_dims})")
        
        # Encode image through encoder
        image_obs = self.conv_encoder(image_obs)
        
        # Process navigation observation
        if isinstance(navigation_obs, np.ndarray):
            navigation_obs = torch.tensor(navigation_obs, dtype=torch.float).to(self.device)
        else:
            navigation_obs = torch.tensor(navigation_obs, dtype=torch.float).to(self.device)
        
        # Concatenate encoded image features with navigation features
        state_vec = torch.cat((image_obs.view(-1), navigation_obs), -1)
        
        return state_vec