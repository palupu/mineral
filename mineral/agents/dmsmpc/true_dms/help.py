def states_equal(state1, state2, atol=1e-6):
    """Check if two warp State objects have equal data."""
    import numpy as np
    
    # Compare each attribute
    attrs_to_compare = ['particle_q', 'particle_qd', 'particle_f', 
                        'body_q', 'body_qd', 'body_f', 
                        'joint_q', 'joint_qd']
    
    for attr in attrs_to_compare:
        arr1 = getattr(state1, attr, None)
        arr2 = getattr(state2, attr, None)
        
        if arr1 is None and arr2 is None:
            continue
        if arr1 is None or arr2 is None:
            return False
        
        # Convert warp arrays to numpy for comparison
        if not np.allclose(arr1.numpy(), arr2.numpy(), atol=atol):
            return False
    
    return True