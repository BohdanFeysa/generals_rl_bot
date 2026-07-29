import jax.numpy as jnp
import flax.linen as nn

class GeneralsPolicy(nn.Module):
    @nn.compact
    def __call__(self, x, h_in):
        # x input is NCHW: (Batch, 10, H, W)
        x = jnp.transpose(x, (0, 2, 3, 1)) # Convert to NHWC for Conv layers
        
        # 3-Layer Spatial CNN
        x = nn.relu(nn.Conv(features=32, kernel_size=(3,3), padding='SAME', name='Conv_0')(x))
        x = nn.relu(nn.Conv(features=64, kernel_size=(3,3), padding='SAME', name='Conv_1')(x))
        x = nn.relu(nn.Conv(features=64, kernel_size=(3,3), padding='SAME', name='Conv_2')(x))
        
        x = x.reshape((x.shape[0], -1)) # Flatten for GRU
        
        # Fog of War Memory (GRU)
        gru = nn.GRUCell(name='GRU_0')
        h_out, gru_out = gru(h_in, x)
        
        # Output Heads
        logits_cell = nn.Dense(features=400, name='Policy_Cell')(gru_out)
        logits_dir = nn.Dense(features=4, name='Policy_Dir')(gru_out)
        value = nn.Dense(features=1, name='Critic')(gru_out)
        
        return logits_cell, logits_dir, value, h_out
