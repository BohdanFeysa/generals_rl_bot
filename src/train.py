import jax
import jax.numpy as jnp
import numpy as np
import optax
import engine # Our compiled nanobind module
from network import GeneralsPolicy

NUM_ENVS = 4096
MAP_H, MAP_W = 20, 20
HIDDEN_DIM = 256

model = GeneralsPolicy()

# JIT-compile the inference pass to run purely on GPU registers
@jax.jit
def select_actions(params, obs, h_in, key):
    logits_cell, logits_dir, val, h_out = model.apply(params, obs, h_in)
    
    # Gumbel-max sampling for fast exploration
    key, k1, k2 = jax.random.split(key, 3)
    a_cell = jnp.argmax(logits_cell - jnp.log(-jnp.log(jax.random.uniform(k1, logits_cell.shape))), axis=-1)
    a_dir = jnp.argmax(logits_dir - jnp.log(-jnp.log(jax.random.uniform(k2, logits_dir.shape))), axis=-1)
    
    return a_cell, a_dir, val, h_out, key

def main():
    batched_env = engine.BatchedGeneralsEnv(NUM_ENVS, MAP_H, MAP_W)
    key = jax.random.PRNGKey(42)
    
    # Init model params
    dummy_obs = jnp.zeros((NUM_ENVS, 10, MAP_H, MAP_W))
    dummy_hidden = jnp.zeros((NUM_ENVS, HIDDEN_DIM))
    params = model.init(key, dummy_obs, dummy_hidden)
    
    batched_env.reset()
    hidden_states = jnp.zeros((NUM_ENVS, HIDDEN_DIM))
    
    print(f"Booting 1PFlop/s Training Loop with {NUM_ENVS} parallel environments...")
    
    for step in range(1_000_000):
        # 1. Zero-Copy pull from C++ to JAX
        raw_obs = batched_env.get_batched_observation()
        jax_obs = jnp.asarray(raw_obs)
        
        # 2. XLA-Accelerated Forward Pass
        a_cell, a_dir, values, hidden_states, key = select_actions(params, jax_obs, hidden_states, key)
        
        # 3. Pull actions back to CPU
        actions_cell = np.array(jax.device_get(a_cell)).tolist()
        actions_dir = np.array(jax.device_get(a_dir)).tolist()
        
        # 4. Advance C++ Engine
        batched_env.step(actions_cell, actions_dir)
        
        # (Standard PPO Rollout & Update logic goes here)
        # Note: Apply your reward shaping here: +1.0 for <800 win, -0.5 for 1200 draw.

    # Once finished, pass the nested params to the exporter
    import export
    export.export_to_binary(params['params'])

if __name__ == "__main__":
    main()
