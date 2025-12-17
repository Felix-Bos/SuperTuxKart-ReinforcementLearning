from logger import Logger
import gym
from typing import List, Optional
import numpy as np
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm
import torch
from Buffers.types import *

class Trainer:


    """
    This API implements a generic trainer for RL agents.
    It handles the interaction between the agent and the environment,
    training loop,
    replay buffer,
    evaluation and 
    logging.
    """

    def __init__(
        self,
        env: gym.Env,
        eval_env: List[ gym.Env ],
        agent, 
        replay_buffer, 
        logger: Logger,
        eval_interval: int = 5_000,
        max_epochs: int = 100000,
        n_steps_per_epoch: int = 1000,
        nb_evals: int = 10,
        learning_starts = None ,
        save_best: bool = False,
        base_dir: Optional[str] = None,
        plot_agents: bool = False,
    ):
        self.env = env
        self.eval_env = eval_env
        self.agent = agent
        self.buffer = replay_buffer
        self.logger = logger
        self.eval_interval = eval_interval
        self.max_epochs = max_epochs
        self.n_steps_per_epoch = n_steps_per_epoch
        self.nb_evals = nb_evals
        self.learning_starts = learning_starts
        self.save_best = save_best
        self.base_dir = base_dir
        self.plot_agents = plot_agents

        def evaluate(self) -> np.array:
            """
            evaluate the agent on the eval_envs
            Returns:
                the rewards obtained in each step of the evaluation
            """
            rewards = np.zeros((len(self.eval_env), 1500))
            self.hidden = self.agent.algo.reset_hidden()
            for i, eval_env in enumerate(self.eval_env):
                obs, _ = eval_env.reset()  # Gymnasium format returns (obs, info)
                done = False
                step_count = 0
                while not done:
                    action, self.hidden = self.agent.algo.select_action(obs, self.hidden, eval_mode=True)
                    if hasattr(action, "detach"):
                        actions = {k: v.detach().cpu().numpy() for k, v in action.items()}
                    obs, reward, terminated, truncated, _ = eval_env.step(actions)
                    done = terminated or truncated  # Gymnasium format
                    rewards[i][step_count] = reward
                    step_count += 1

                return rewards

        def train(self):
            
            pbar = tqdm(total=self.max_expochs, desc=f"Agent Type: {self.agent.name}")
            
            obs, _ = self.env.reset()  # Gymnasium format returns (obs, info)

            # counters
            episode_reward = 0.0
            episode_counts = 0
            self.episode_step = 0

            # pre-allocation for Transition class
            self.log_prob = None
            self.hidden = self.agent.algo.reset_hidden()
            self.value = None
            self.next_hidden = None


            # Main training loop
            while episode_counts < self.max_epochs:

                # ---------------------------------------------------------
                # Interact with the environment
                # ---------------------------------------------------------
                # warmup phase to fill the replay buffer
                if self.buffer.size < self.learning_starts:
                    action = self.env.action_space.sample()
                else:
                    action, self.log_prob, self.value, self.next_hidden = self.agent.algo.select_action(obs, self.hidden, eval_mode=False)

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                
                # ---------------------------------------------------------
                # Store Transition in Replay Buffer 
                # ---------------------------------------------------------
                transition = {
                    "states": obs,
                    "action": action,
                    "reward": reward,
                    "next_states": next_obs,
                    "truncated": truncated,
                    "terminated": terminated,

                    "log_prob": self.log_prob,
                    "value": self.value,
                    "hidden": self.hidden,
                    "next_hidden": self.next_hidden,
                }
                t = Transition(**Transition)
                self.buffer.add(t)
                
                obs = next_obs
                self.episode_step += 1

                # ---------------------------------------------------------
                # Train the agent 
                # ---------------------------------------------------------
               
                if self.buffer.size >= self.learning_starts:
                    if self.local_step % self.agent.algo.config['update_interval'] == 0:
                        batch = self.buffer.sample(self.agent.algo.config['batch_size'], self.agent.algo.config.get('recent', False))
                        metrics = self.agent.algo.update(batch)

                # ---------------------------------------------------------
                # Handle logging
                # ---------------------------------------------------------
                # increment one global step
                self.logger.total_step += 1
                self.log_scalars(metrics)
                self.save_metrics(metrics)

                # ---------------------------------------------------------
                # handle Done 
                # ---------------------------------------------------------
                if terminated or truncated:
                    obs, _ = self.env.reset()  # Gymnasium format returns (obs, info)
                    episode_counts += 1
                    self.episode_step = 0

                
                 

                
                
            



        

    

