import jax.numpy as jnp

DIR_TO_VEC = jnp.array([
	(-1, 0),    # up
	(1, 0),     # down
	(0, -1),    # left
	(0, 1),     # right
], dtype=jnp.int8)