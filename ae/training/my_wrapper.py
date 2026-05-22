import gymnasium as gym
import numpy as np
from til_environment.bomberman_env import Bomberman
from til_environment.config import default_config
from til_environment.actions import Action
from omegaconf import OmegaConf


cfg = default_config()
cfg = OmegaConf.merge(cfg, {
    "rewards": {
        "step_penalty": -0.01,         # small cost per step
        "stationary_penalty": -0.05,   # discourage staying still
        "agent_collide_wall": -0.1,    # penalise bumping walls
        "collect_mission": 4.0,        # emphasise mission tiles
    },

                      
    "env": {
    "render_mode": "human",
    "grid_size": 16,
    "num_teams": 6,
    "num_iters": 200,
    "novice" : False, # can try changing
    "tile_respawn_steps": 40,
    }
})


class BombermanEnv(gym.Env):
    metadata = {"render_modes": ["human","rgb_array"]}

    def __init__(self, cfg=None):
        self.cfg = cfg
        self._env = Bomberman()
        self.controlled = "agent_0"
        raw_space = self._env.observation_space(self.controlled)
        flat_dim = gym.spaces.utils.flatdim(raw_space)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (flat_dim,), np.float32)
        self.action_space = gym.spaces.Discrete(6)

    def reset(self, seed=None, options=None):
        self._env.reset(seed=seed)
        self._skip_to_agent()
        obs = self._env.observe(self.controlled)
        return self._flat(obs), {}

    def step(self, action):
        # Step controlled agent
        mask = self._env.observe(self.controlled)["action_mask"]
        if not mask[action]:
            action = 4
        self._env.step(int(action))
    
        # Step all other agents in correct AEC order
        for agent in self._env.agent_iter():
            if agent == self.controlled:
                break  # back to our agent — stop
            obs, _, term, trunc, _ = self._env.last()
            a = None if (term or trunc) else self._env.action_space(agent).sample()
            self._env.step(a)

        # Now last() correctly reflects agent_0's state
        obs, rew, term, trunc, info = self._env.last()
        return self._flat(obs), rew, term, trunc, info



    def _skip_to_agent(self):
        for agent in self._env.agent_iter():
            if agent == self.controlled: return
            obs, _, term, trunc, _ = self._env.last()
            a = None if (term or trunc) else self._env.action_space(agent).sample()
            self._env.step(a)

    def _step_others(self):
        for agent in self._env.agent_iter():
            if agent == self.controlled: continue
            obs, _, term, trunc, _ = self._env.last()
            a = None if (term or trunc) else self._env.action_space(agent).sample()
            self._env.step(a)

    def _flat(self, obs):
        sp = self._env.observation_space(self.controlled)
        return gym.spaces.utils.flatten(sp, obs).astype(np.float32)
    def close(self): self._env.close()

    






class SelfPlayWrapper(gym.Wrapper):
    def set_opponent_model(self, model):
        self.opp_model = model

    def step(self, action):
        # 1. Apply action mask for controlled agent
        mask = self.env.observe(self.controlled)["action_mask"]
        if not mask[action]:
            action = 4
        self.env.step(int(action))

        # 2. Step opponents in correct AEC order
        for agent in self.env.agent_iter():
            if agent == self.controlled:
                break  # back to our turn — stop
            obs, _, term, trunc, _ = self.env.last()  # last() now correctly reflects `agent`
            if term or trunc:
                a = None
            elif hasattr(self, 'opp_model') and self.opp_model:
                flat_obs = self._flat(obs)             # 3. fixed method name
                a, _ = self.opp_model.predict(flat_obs, deterministic=True)
            else:
                a = self.env.action_space(agent).sample()
            self.env.step(a)

        # 4. last() now correctly reflects agent_0's state
        obs, rew, term, trunc, info = self._env.last()
        return self._flat(obs), rew, term, trunc, info  # 5. fixed method name
        
    def get_action_mask(self):
        return self.env.get_action_mask()