"""Manages the AE model."""
# from my_wrapper import BombermanEnv
from sb3_contrib import MaskablePPO
from til_environment.bomberman_env import Bomberman
import gymnasium as gym
from gymnasium.spaces import utils

class AEManager:

    def __init__(self):
        # This is where you can initialize your model and any static configurations.
        # TODO
        self.model = MaskablePPO.load("best_reward/best_reward_model")
        

    def ae(self, observation: dict[str, int | list[int]]) -> int:
        """Gets the next action for the agent, based on the observation.

        Args:
            observation: The observation from the environment. See
                `ae/README.md` for the format.

        Returns:
            An integer representing the action to take. See `ae/README.md` for
            the options.
        """

        # Your inference code goes here.
        # TOD:
        env = Bomberman()
        mask = observation["action_mask"]
        obs = utils.flatten(env.observation_space(), observation)
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)
