import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from til_environment.bomberman_env import Bomberman
from til_environment.config import default_config
from omegaconf import OmegaConf


cfg = default_config()

# Example: encourage exploration, penalise idling
cfg = OmegaConf.merge(cfg, {
    "rewards": {
        "step_penalty": -0.01,         # small cost per step
        "stationary_penalty": -0.05,   # discourage staying still
        "agent_collide_wall": -0.1,    # penalise bumping walls
        "collect_mission": 10.0,       # emphasise mission tiles
    }
})

# ── 1. Build the single-agent wrapper ───────────────────────────────
class BombermanEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg=None):
        self.cfg = cfg or default_config()
        self._env = Bomberman(self.cfg)
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
        mask = self._env.observe(self.controlled)["action_mask"]
        if not mask[action]: action = 4
        self._env.step(int(action))
        self._step_others()
        obs, rew, term, trunc, info = self._env.last()
        return self._flat(obs), rew, term, trunc, info

    def get_action_mask(self):
        obs = self._env.observe(self.controlled)
        return obs["action_mask"].astype(bool)

    def _skip_to_agent(self):
        for agent in self._env.agent_iter():
            if agent == self.controlled: return
            obs, _, term, trunc, _ = self._env.last()
            a = None if (term or trunc) else self._env.action_space(agent).sample()
            self._env.step(a)

    def _step_others(self):
        for agent in self._env.agents:
            if agent == self.controlled: continue
            obs, _, term, trunc, _ = self._env.last()
            a = None if (term or trunc) else self._env.action_space(agent).sample()
            self._env.step(a)

    def _flat(self, obs):
        sp = self._env.observation_space(self.controlled)
        return gym.spaces.utils.flatten(sp, obs).astype(np.float32)

    def close(self): self._env.close()

# ── 2. Validate the wrapper ─────────────────────────────────────────
check_env(BombermanEnv(), warn=True)

# ── 3. Wrap with ActionMasker for MaskablePPO ───────────────────────
def make_env():
    env = BombermanEnv()
    env = ActionMasker(env, lambda e: e.get_action_mask())
    return env

# ── 4. Create parallel envs (4 copies for faster data collection) ───
vec_env = make_vec_env(make_env, n_envs=4)
vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)


# ── 5. Create the MaskablePPO model ─────────────────────────────────
model = MaskablePPO(
    policy="MlpPolicy",
    env=vec_env,
    n_steps=2048,        # steps per env per update (8192 total)
    batch_size=256,       # larger batch for stability
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,        # small entropy bonus for exploration
    learning_rate=3e-4,
    policy_kwargs=dict(net_arch=[256, 256]),  # wider network for complex obs
    verbose=1,
    tensorboard_log="./ae_logs/"
)

# ── 6. Train ─────────────────────────────────────────────────────────
model.learn(total_timesteps=1_000_000)

# ── 7. Save ──────────────────────────────────────────────────────────
model.save("ppo_bomberman")