import flax
import numpy as np
import struct
import jax

def export_to_binary(params, filepath="weights.bin"):
    """Dumps nested JAX dicts into raw binary for CP-style fread ingestion."""
    flat_params = flax.traverse_util.flatten_dict(params, sep='.')
    
    with open(filepath, 'wb') as f:
        for name, tensor in flat_params.items():
            np_arr = np.array(jax.device_get(tensor)).astype(np.float32)
            
            # String length, then string
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)
            
            # Array size, then raw bytes
            f.write(struct.pack('I', np_arr.size))
            f.write(np_arr.tobytes())
            
            print(f"Exported: {name} (Size: {np_arr.size})")
