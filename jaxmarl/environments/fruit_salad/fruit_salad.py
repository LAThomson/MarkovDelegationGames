from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
from jaxmarl.environments.fruit_salad.layouts import fruit_salad_layouts
from jaxmarl.environments import spaces
from jaxmarl.environments.fruit_salad.common import DIR_TO_VEC

import jax, flax, chex
from jax import numpy as jnp
import numpy as np
from flax.core import FrozenDict

import struct
from functools import partial
from typing import Tuple, Dict, Optional
from enum import IntEnum

REWARDS = {
    "agent_0": {
        "apple": 3,
        "ripe_apple": 4,
        "banana": 1,
        "ripe_banana": 2,
        "cherry": 2,
        "ripe_cherry": 3,
        "agent_0_coefficient": 1,
        "agent_1_coefficient": 0.5,
        "agent_2_coefficient": 0.5,
        "agent_3_coefficient": 0.5,
        "agent_4_coefficient": 0.5,
        "agent_5_coefficient": 0.5,
        "agent_6_coefficient": 0.5,
        "agent_7_coefficient": 0.5,
    },
    "agent_1": {
        "apple": 6,
        "ripe_apple": 5,
        "banana": 3,
        "ripe_banana": 3,
        "cherry": 4,
        "ripe_cherry": 1,
        "agent_0_coefficient": 0.5,
        "agent_1_coefficient": 1,
        "agent_2_coefficient": 0.5,
        "agent_3_coefficient": 0.5,
        "agent_4_coefficient": 0.5,
        "agent_5_coefficient": 0.5,
        "agent_6_coefficient": 0.5,
        "agent_7_coefficient": 0.5,
    },
    "agent_2": {
        "apple": 2,
        "ripe_apple": 2,
        "banana": 3,
        "ripe_banana": 6,
        "cherry": 1,
        "ripe_cherry": 2,
        "agent_0_coefficient": 0.5,
        "agent_1_coefficient": 0.5,
        "agent_2_coefficient": 1,
        "agent_3_coefficient": 0.5,
        "agent_4_coefficient": 0.5,
        "agent_5_coefficient": 0.5,
        "agent_6_coefficient": 0.5,
        "agent_7_coefficient": 0.5,
    },
    "agent_3": {
        "apple": 0,
        "ripe_apple": 0,
        "banana": 3,
        "ripe_banana": 2,
        "cherry": 4,
        "ripe_cherry": 5,
        "agent_0_coefficient": 0.5,
        "agent_1_coefficient": 0.5,
        "agent_2_coefficient": 0.5,
        "agent_3_coefficient": 1,
        "agent_4_coefficient": 0.5,
        "agent_5_coefficient": 0.5,
        "agent_6_coefficient": 0.5,
        "agent_7_coefficient": 0.5,
    },
    "agent_4": {
        "apple": 0,
        "ripe_apple": 0,
        "banana": 0,
        "ripe_banana": 0,
        "cherry": 0,
        "ripe_cherry": 0,
        "agent_0_coefficient": 0.5,
        "agent_1_coefficient": 0.5,
        "agent_2_coefficient": 0.5,
        "agent_3_coefficient": 0.5,
        "agent_4_coefficient": 1,
        "agent_5_coefficient": 0.5,
        "agent_6_coefficient": 0.5,
        "agent_7_coefficient": 0.5,
    },
    "agent_5": {
        "apple": 0,
        "ripe_apple": 0,
        "banana": 0,
        "ripe_banana": 0,
        "cherry": 0,
        "ripe_cherry": 0,
        "agent_0_coefficient": 0.5,
        "agent_1_coefficient": 0.5,
        "agent_2_coefficient": 0.5,
        "agent_3_coefficient": 0.5,
        "agent_4_coefficient": 0.5,
        "agent_5_coefficient": 1,
        "agent_6_coefficient": 0.5,
        "agent_7_coefficient": 0.5,
    },
    "agent_6": {
        "apple": 0,
        "ripe_apple": 0,
        "banana": 0,
        "ripe_banana": 0,
        "cherry": 0,
        "ripe_cherry": 0,
        "agent_0_coefficient": 0.5,
        "agent_1_coefficient": 0.5,
        "agent_2_coefficient": 0.5,
        "agent_3_coefficient": 0.5,
        "agent_4_coefficient": 0.5,
        "agent_5_coefficient": 0.5,
        "agent_6_coefficient": 1,
        "agent_7_coefficient": 0.5,
    },
    "agent_7": {
        "apple": 0,
        "ripe_apple": 0,
        "banana": 0,
        "ripe_banana": 0,
        "cherry": 0,
        "ripe_cherry": 0,
        "agent_0_coefficient": 0.5,
        "agent_1_coefficient": 0.5,
        "agent_2_coefficient": 0.5,
        "agent_3_coefficient": 0.5,
        "agent_4_coefficient": 0.5,
        "agent_5_coefficient": 0.5,
        "agent_6_coefficient": 0.5,
        "agent_7_coefficient": 1,
    }}

class Actions(IntEnum):
    up = 0
    down = 1
    left = 2
    right = 3
    interact = 4
    stay = 5

@struct.dataclass
class State:                        # IMPORTANT: shape of each State channel must stay same throughout game for better efficiency
    agent_pos: chex.Array           # shape: (num_agents x 2), stores agent i's position as agent_pos[i] = [y, x]
    wall_pos: chex.Array            # shape: (height x width), stores wall positions with 1s and empty with 0s
    switch_pos: chex.Array          # same as wall_pos
    gate_pos: chex.Array            # same as wall_pos
    apple_pos: chex.Array           # shape: (height x width), stores 1s for present apples, -1s for picked up apples, and 0s for empty
    ripe_apple_pos: chex.Array      # same as apple_pos
    banana_pos: chex.Array          # same as apple_pos
    ripe_banana_pos: chex.Array     # same as apple_pos
    cherry_pos: chex.Array          # same as apple_pos
    ripe_cherry_pos: chex.Array     # same as apple_pos
    gate_open: bool                 # bool storing whether gates have been opened
    time: int                       # int storing current timestep of the environment
    terminal: bool                  # bool storing whether the current environment state is terminal

class FruitSalad(MultiAgentEnv):
    """Fruit Salad Environment"""

    def __init__(
        self,
        num_agents: int,
        max_steps: int = 25,
        layout = fruit_salad_layouts["small_2p"],
    ) -> None:
        """
        num_agents (int): number of agents within the environment
        max_steps (int): number of steps a game should be played for before termination
        layout: gridworld representation to instantiate
        """
        
        # cap num_agents to be no greater than the number of agent positions there are in the layout
        self.num_agents = min(num_agents, len(layout["agent_idx"]))
        self.max_steps = max_steps

        super().__init__(self.num_agents)

        self.height = layout["height"]
        self.width = layout["width"]
        self.obs_shape = (self.width, self.height, self.num_agents + 9)  # full gridworld observation encoded through (num_agents + 9) channels

        self.layout = layout
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        self.action_set = jnp.array([
            Actions.up,
            Actions.down,
            Actions.left,
            Actions.right,
            Actions.interact,
            Actions.stay,
        ])

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Performs resetting of the environment."""
        # No significant randomisation necessary: just shuffle which agent starts in which position
        
        layout = self.layout
        height = self.height
        width = self.width
        num_agents = self.num_agents

        # get agent_idx and convert to y-x coordinates, then shuffle order and limit to num_agents
        agent_idx = layout.get("agent_idx")
        agent_pos = jnp.array([agent_idx // width, agent_idx % width], dtype=jnp.uint32).transpose()
        key, subkey = jax.random.split(key)
        agent_pos = jax.random.permutation(subkey, agent_pos)
        agent_pos = agent_pos[:num_agents]

        # create an array to help convert from index to matrix format
        all_pos = np.arange(np.prod([height, width]), dtype=jnp.uint8)
        
        # set up walls, switches, and gates
        wall_idx = layout.get("wall_idx")
        wall_occupied = jnp.zeros_like(all_pos).at[wall_idx].set(1)
        wall_pos = wall_occupied.reshape((height, width))

        switch_idx = layout.get("switch_idx")
        switch_occupied = jnp.zeros_like(all_pos).at[switch_idx].set(1)
        switch_pos = switch_occupied.reshape((height, width))

        gate_idx = layout.get("gate_idx")
        gate_occupied = jnp.zeros_like(all_pos).at[gate_idx].set(1)
        gate_pos = gate_occupied.reshape((height, width))

        # also set up all the fruit
        apple_idx = layout.get("apple_idx")
        apple_occupied = jnp.zeros_like(all_pos).at[apple_idx].set(1)
        apple_pos = apple_occupied.reshape((height, width))

        ripe_apple_idx = layout.get("ripe_apple_idx")
        ripe_apple_occupied = jnp.zeros_like(all_pos).at[ripe_apple_idx].set(1)
        ripe_apple_pos = ripe_apple_occupied.reshape((height, width))
        
        banana_idx = layout.get("banana_idx")
        banana_occupied = jnp.zeros_like(all_pos).at[banana_idx].set(1)
        banana_pos = banana_occupied.reshape((height, width))

        ripe_banana_idx = layout.get("ripe_banana_idx")
        ripe_banana_occupied = jnp.zeros_like(all_pos).at[ripe_banana_idx].set(1)
        ripe_banana_pos = ripe_banana_occupied.reshape((height, width))
        
        cherry_idx = layout.get("cherry_idx")
        cherry_occupied = jnp.zeros_like(all_pos).at[cherry_idx].set(1)
        cherry_pos = cherry_occupied.reshape((height, width))

        ripe_cherry_idx = layout.get("ripe_cherry_idx")
        ripe_cherry_occupied = jnp.zeros_like(all_pos).at[ripe_cherry_idx].set(1)
        ripe_cherry_pos = ripe_cherry_occupied.reshape((height, width))

        initial_state = State(
            agent_pos=agent_pos,
            wall_pos=wall_pos,
            switch_pos=switch_pos,
            gate_pos=gate_pos,
            apple_pos=apple_pos,
            ripe_apple_pos=ripe_apple_pos,
            banana_pos=banana_pos,
            ripe_banana_pos=ripe_banana_pos,
            cherry_pos=cherry_pos,
            ripe_cherry_pos=ripe_cherry_pos,
            gate_open=False,
            time=0,
            terminal=False,
        )

        initial_obs = self.get_obs(initial_state)

        return jax.lax.stop_gradient(initial_obs), jax.lax.stop_gradient(initial_state)

    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""

        def _get_agent_action(i : int, actions : Dict[str, chex.Array]) -> Actions:
            return self.action_set.take(indices=actions[f"agent_{i}"])
        batch_get_action = jax.vmap(_get_agent_action, in_axes=[0, None])
        acts = batch_get_action(jnp.arange(self.num_agents), actions)

        state, rewards = self.step_agents(key, state, acts)

        state = state.replace(time=state.time + 1)

        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        obs = self.get_obs(state)
        dones = {f"agent_{i}": done for i in range(self.num_agents)}
        dones["__all__": done]

        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(state),
            rewards,
            dones,
            {},
        )
    
    def is_terminal(self, state: State) -> bool:
        """Check whether state is terminal."""
        done_steps = state.time >= self.max_steps
        return done_steps | state.terminal
    
    def step_agents(
        self, key: chex.PRNGKey, state: State, actions: chex.Array,
    ) -> Tuple[State, Dict[str, float]]:
        """Take in agents' actions and return the resulting state as well as rewards for each agent"""
        width = self.width
        height = self.height
        
        # step 1: AGENT MOVEMENT
        #    (a): calculate agents' end positions after action
        agent_pos_before = state.agent_pos
        is_move = jnp.logical_and(actions != Actions.interact, actions != Actions.stay)
        reshaped_is_move = jnp.expand_dims(is_move, axis=1) # needed to make broadcasting work!
        agent_pos_after = (reshaped_is_move)*(agent_pos_before + DIR_TO_VEC[jnp.minimum(actions, 3)]) + (1-reshaped_is_move)*agent_pos_before

        #    (b): check end positions don't clash with impassable tiles or other agents' end positions (else no move)
        batch_compare = jax.vmap(lambda agent_pos, other_pos : jax.vmap(lambda x, y : jnp.all(x == y), in_axes=[None, 0])(agent_pos, other_pos), in_axes=[0, None])
        wall_idx = jnp.flatnonzero(state.wall_pos)
        wall_coords = [wall_idx // width, wall_idx % width]
        wall_collisions = jnp.any(batch_compare(agent_pos_after, wall_coords), axis=1)
        other_agent_collisions = jnp.sum(batch_compare(agent_pos_after, agent_pos_after), axis=1) > 1
        collision_mask = wall_collisions | other_agent_collisions
        
        #    (c): set agents' new positions to (collision_mask)*new_positions + (1-collision_mask)*old_positions
        reshaped_collision_mask = jnp.expand_dims(collision_mask, axis=1) # needed to make broadcasting work!
        new_agent_pos = (reshaped_collision_mask)*agent_pos_after + (1 - reshaped_collision_mask)*agent_pos_before

        # step 2: AGENT INTERACTION
        #    (a): assume agent takes interact action and process result
        is_interact = actions == Actions.interact
        interact_fruit_maps, interact_rewards = jax.vmap(self.process_interact, in_axes=[None, 0])(state, jnp.arange(self.num_agents))

        #    (b): collate effects on fruit positions (i.e. remove picked-up fruit) - done by taking a minimum
        reshaped_is_interact = jnp.expand_dims(is_interact, axis=[1,2,3]) # needed to make broadcasting work!
        original_fruit_maps = jnp.array([state.apple_pos, state.ripe_apple_pos, state.banana_pos, state.ripe_banana_pos, state.cherry_pos, state.ripe_cherry_pos])
        new_fruit_maps = jnp.min((reshaped_is_interact)*(interact_fruit_maps) + (1-reshaped_is_interact)*(original_fruit_maps), axis=0)

        #    (c): dish out rewards to agents
        reshaped_is_interact = jnp.expand_dims(is_interact, axis=1) # needed to make broadcasting work!
        new_rewards = jnp.sum((reshaped_is_interact)*(interact_rewards) + (1-reshaped_is_interact)*(jnp.zeros(self.num_agents)), axis=0)
        
        # step 3: SWITCHES
        #    (a): check if gates are already open
        gate_open = state.gate_open

        #    (b): check if all switches are interacted with simultaneously
        switch_idx = jnp.flatnonzero(state.switch_pos)
        switch_coords = [switch_idx // width, switch_idx % width]
        all_agent_switch_collisions = batch_compare(state.agent_pos, switch_coords)
        agent_interacts_on_switch = reshaped_is_interact * all_agent_switch_collisions
        switch_is_interacted_with = jnp.any(agent_interacts_on_switch, axis=0)
        all_switches_pressed = jnp.all(switch_is_interacted_with)

        #    (c): set gate_open to True if either of the above is True
        new_gate_open = gate_open | all_switches_pressed

        return (
            state.replace(
                agent_pos=new_agent_pos,
                apple_pos=new_fruit_maps[0],
                ripe_apple_pos=new_fruit_maps[1],
                banana_pos=new_fruit_maps[2],
                ripe_banana_pos=new_fruit_maps[3],
                cherry_pos=new_fruit_maps[4],
                ripe_cherry_pos=new_fruit_maps[5],
                gate_open=new_gate_open,
                terminal=False),
            new_rewards,
        )

    def process_interact(
        self,
        state: State,
        agent_idx: int,
    ) -> chex.Array[chex.Array, Dict[str, float]]:
        """Assume agent took interact actions. Result depends on whether the agent is standing on a fruit."""
        # get agent's and fruits' positions
        agent_pos = state.agent_pos[agent_idx]
        fruit_maps = jnp.array([state.apple_pos, state.ripe_apple_pos, state.banana_pos, state.ripe_banana_pos, state.cherry_pos, state.ripe_cherry_pos])

        # get corresponding rewards and coefficients
        rds = REWARDS[f"agent_{agent_idx}"]
        agent_reward_values = jnp.array([rds["apple"], rds["ripe_apple"], rds["banana"], rds["ripe_banana"], rds["cherry"], rds["ripe_cherry"],])
        batch_get_coefficients = jax.vmap(lambda i, agent_idx : REWARDS[f"agent_{i}"][f"agent_{agent_idx}_coefficient"], in_axes=[0, None])
        coefficients = batch_get_coefficients(jnp.arange(self.num_agents), agent_idx)

        # check position against fruit locations
        batch_is_on = jax.vmap(lambda fruit_map, agent_pos : fruit_map[agent_pos[0], agent_pos[1]] == 1, in_axes=[0, None])
        agent_is_on = batch_is_on(fruit_maps, agent_pos) # assumption is AT MOST ONE of agent_is_on == True (max one fruit per grid position)

        # update fruit maps if agent is on fruit
        reshaped_agent_is_on = jnp.expand_dims(agent_is_on, axis=1) # needed to make broadcasting work!
        new_fruit_maps = (reshaped_agent_is_on)*fruit_maps.at[agent_pos[0], agent_pos[1]].set(-1) + (1-reshaped_agent_is_on)*fruit_maps

        # collate appropriate rewards for all agents
        # fiddly calculation of raw_reward_value is to default to 0 if any(agent_is_on)==False [i.e. if agent not on fruit]
        raw_reward_value = jnp.concat([agent_reward_values, jnp.array([0])])[jnp.concat([agent_is_on, jnp.array([True])])].at[0].get()
        rewards = raw_reward_value * coefficients

        return new_fruit_maps, rewards
        
        
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Return a full observation of size (num_agents x num_channels x height x width), where n_channels = num_agents + 9.
        Channels are of shape (height x width) and are binary (0/1) except where indicated otherwise.

        The list of channels is below. Agent-specific channels are ordered so that an agent perceives its channel first.
        Environment channels are the same (and in the same order) for all agents.

        Agent positions:
        0. position of agent i (1 at agent loc, 0 otherwise)
        1-7*. position of all other agents (channel i doesn't exist if environment has num_agents <= i)

        Variable env channels (1 where object of type X is located, 0 otherwise):
        +1. impassable tile locations (i.e. walls, closed gates)
        +2. apple locations
        +3. ripe apple locations
        +4. banana locations
        +5. ripe banana locations
        +6. cherry locations
        +7. ripe cherry locations
        +8. switch locations (1 where switch is until switches pressed, then all 0s)
        +9. gate locations (1 where gate is until switches pressed, then all 0s)
        """

        width = self.obs_shape[0]
        height = self.obs_shape[1]
        n_channels = self.obs_shape[2]

        # agent channels                                                # channels 0-7*
        agent_pos_channels = jnp.zeros((self.num_agents, height, width), dtype=jnp.uint8)
        batch_add_agent_pos = jax.vmap(
            lambda i, agent_pos_channel, state : agent_pos_channel.at[state.agent_pos[i, 1], state.agent_pos[i, 0]].set(1),
            in_axes=[0, 0, None]
        )
        agent_pos_channels = batch_add_agent_pos(range(self.num_agents), agent_pos_channels, state)

        # static env channels
        
        # if gates not yet open, treat them as impassable. Otherwise, just mark walls
        impassable_channel = jax.lax.select(                            # channel +1
            state.gate_open,
            state.wall_pos,
            state.wall_pos + state.gate_pos
        )

        # variable env channels
        
        # fruit channels ignore negative values (i.e. ignore fruit that has been picked up)
        apple_channel = jnp.maximum(state.apple_pos, 0)                 # channel +2
        ripe_apple_channel = jnp.maximum(state.ripe_apple_pos, 0)       # channel +3
        banana_channel = jnp.maximum(state.banana_pos, 0)               # channel +4
        ripe_banana_channel = jnp.maximum(state.ripe_banana_pos, 0)     # channel +5
        cherry_channel = jnp.maximum(state.cherry_pos, 0)               # channel +6
        ripe_cherry_channel = jnp.maximum(state.ripe_cherry_pos, 0)     # channel +7

        # switches and gates marked as 1s if gates not open, but 0s otherwise
        switch_channel = jax.lax.select(                                # channel +8
            state.gate_open,
            state.switch_pos,
            jnp.zeros(height, width, dtype=jnp.uint8)
        )
        gate_channel = jax.lax.select(                                  # channel +9
            state.gate_open,
            state.gate_pos,
            jnp.zeros(height, width, dtype=jnp.uint8)
        )

        # stack all the environment channels together
        env_channels = jnp.vstack(
            impassable_channel,
            apple_channel,
            ripe_apple_channel,
            banana_channel,
            ripe_banana_channel,
            cherry_channel,
            ripe_cherry_channel,
            switch_channel,
            gate_channel
        )

        # permute agent position channels to be agent-centred
        batch_permute = jax.vmap(
            lambda i, channels : jnp.moveaxis(channels, i, 0),
            in_axes=[0, None]
        )
        agent_centred_pos_channels = batch_permute(range(self.num_agents), agent_pos_channels)
        
        # compile the full observations for each agent
        batch_compile = jax.vmap(
            lambda agent_channels, env_channels : jnp.concat(agent_channels, env_channels),
            in_axes=[0, None]
        )
        all_observations = batch_compile(agent_centred_pos_channels, env_channels)

        # return a dictionary containing each agent's observation
        return dict(zip(self.agents, all_observations))

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: State) -> Dict[str, chex.Array]:
        """Returns the available actions for each agent."""
        return {agent: self.action_set for agent in self.agents}

    @property
    def name(self) -> str:
        """Environment name."""
        return "Fruit Salad"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self, agent_id="") -> spaces.Discrete:
        """Action space of the environment. Agent_id not used since action_space is uniform for all agents"""
        return spaces.Discrete(
            len(self.action_set),
            dtype=jnp.uint32
        )

    def observation_space(self) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, self.obs_shape)

    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        h = self.height
        w = self.width
        return spaces.Dict({
            "agent_pos": spaces.Box(0, max(w, h), (2,), dtype=jnp.uint32),
            "wall_pos": spaces.Box(0, 1, (w, h,), dtype=jnp.uint32),
            "switch_pos": spaces.Box(0, 1, (w, h,), dtype=jnp.uint32),
            "gate_pos": spaces.Box(0, 1, (w, h,), dtype=jnp.uint32),
            "apple_pos": spaces.Box(-1, 1, (w, h,), dtype=jnp.uint32),
            "ripe_apple_pos": spaces.Box(-1, 1, (w, h,), dtype=jnp.uint32),
            "banana_pos": spaces.Box(-1, 1, (w, h,), dtype=jnp.uint32),
            "ripe_banana_pos": spaces.Box(-1, 1, (w, h,), dtype=jnp.uint32),
            "cherry_pos": spaces.Box(-1, 1, (w, h,), dtype=jnp.uint32),
            "ripe_cherry_pos": spaces.Box(-1, 1, (w, h,), dtype=jnp.uint32),
            "gates_open": spaces.Discrete(2),
            "time": spaces.Discrete(self.max_steps),
            "terminal": spaces.Discrete(2),
        })

    def max_steps(self) -> int:
        return self.max_steps
