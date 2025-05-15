from jax import numpy as jnp
from flax.core import FrozenDict

# Fruit Salad layouts explanation:
#   Agents are placed in a grid with walls, fruit, and sometimes switches + gates.
#   Grids can have up to three different fruits (apples, bananas, and cherries), and
#   each type of fruit comes in a regular and ripe variety. Switches are used to
#   open gates (by standing on the marked location and taking the 'interact' action).
#   Importantly, all switches must be pressed at the same time to activate; once
#   activated, all gates in the level will open permanently (i.e. disappear), and
#   switches will have no use thereafter.

# Implementation details:
#   - dict format uses index as counted from top left to bottom right
#   - agent_idx lists agent locations in numerical order (positions are agent-agnostic)

small_2p = {
    "height" : 6,
    "width" : 8,
    "wall_idx" : jnp.array([0,1,2,3,4,5,6,7,
                            8,15,
                            16,23,
                            24,31,
                            32,39,
                            40,41,42,43,44,45,46,47], dtype=jnp.uint8),
    "agent_idx" : jnp.array([34, 37], dtype=jnp.uint8),
    "apple_idx" : jnp.array([17], dtype=jnp.uint8),
    "ripe_apple_idx" : jnp.array([], dtype=jnp.uint8),
    "banana_idx" : jnp.array([12], dtype=jnp.uint8),
    "ripe_banana_idx" : jnp.array([11], dtype=jnp.uint8),
    "cherry_idx" : jnp.array([22], dtype=jnp.uint8),
    "ripe_cherry_idx" : jnp.array([], dtype=jnp.uint8),
    "switch_idx" : jnp.array([], dtype=jnp.uint8),
    "gate_idx" : jnp.array([], dtype=jnp.uint8),
}

# Example of simple layout
small_2p_grid = """
# # # # # # # #
# . . B b . . #
# a . . . . c #
# . . . . . . #
# . O . . O . #
# # # # # # # #
"""

# Example of layout with gated room
gated_2p_grid = """
# # # # # # # #
# . . . . b b #
# O . . # # # #
# . . s | A C #
# . . s | A C #
# O . . # # # #
# . . . . . . #
# # # # # # # #
"""

# Example of layout with more players
compass_4p_grid = """
# # # # # # # # # # #
# . . b . O . c . . #
# . . . . s . . . . #
# b . . # = # . . c #
# . . # a A a # . . #
# O s | A A A | s O #
# . . # a A a # . . #
# c . . # = # . . b #
# . . . . s . . . . #
# . . c . O . b . . #
# # # # # # # # # # #
"""

def layout_grid_to_dict(grid):
    """Assumes `grid` is string representation of the layout, with 1 line per row, and the following symbols:
    #       : wall
    O       : agent
    a / A   : apple (regular/RIPE)
    b / B   : banana (regular/RIPE)
    c / C   : cherry (regular/RIPE)
    s       : switch
    | or =  : gate
    .       : empty cell
    """

    rows = grid.split('\n')

    if len(rows[0]) == 0:
        rows = rows[1:]
    if len(rows[-1]) == 0:
        rows = rows[:-1]

    keys = ["wall_idx", "agent_idx",
            "apple_idx", "ripe_apple_idx",
            "banana_idx", "ripe_banana_idx",
            "cherry_idx", "ripe_cherry_idx",
            "switch_idx", "gate_idx"]
    symbol_to_key = {"#" : "wall_idx", "O" : "agent_idx",
                     "a" : "apple_idx", "A" : "ripe_apple_idx",
                     "b" : "banana_idx", "B" : "ripe_banana_idx",
                     "c" : "cherry_idx", "C" : "ripe_cherry_idx",
                     "s" : "switch_idx",
                     "|" : "gate_idx", "=" : "gate_idx",
                     }

    layout_dict = {key : [] for key in keys}
    layout_dict["height"] = len(rows)
    layout_dict["width"] = len(rows[0].split())
    width = len(rows[0].split())

    for j, row in enumerate(rows):
        for i, obj in enumerate(row.split()):
            idx = width * j + i
            if obj in symbol_to_key.keys():
                # Add object
                layout_dict[symbol_to_key[obj]].append(idx)
            if obj in ["|", "="]:
                # initially, gates are also walls!
                layout_dict["wall_idx"].append(idx)
            elif obj == ".":
                # Empty cell
                continue

    for key in symbol_to_key.values():
        # Transform lists to arrays
        layout_dict[key] = jnp.array(layout_dict[key], dtype=jnp.uint8)

    return FrozenDict(layout_dict)

fruit_salad_layouts = {
    "small_2p" : FrozenDict(small_2p),
    "gated_2p" : layout_grid_to_dict(gated_2p_grid),
    "compass_4p" : layout_grid_to_dict(compass_4p_grid),
}
