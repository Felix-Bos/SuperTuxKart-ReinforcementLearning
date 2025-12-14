"""
SuperTuxKart Gymnasium Utilities
================================

This module provides helper tools for creating and interacting with
SuperTuxKart environments backed by the `pystk2-gymnasium` package.

It simplifies environment configuration through a dataclass and provides
utility functions to:

- Instantiate a STK Gymnasium environment with the correct `AgentSpec`
- Run a quick random-policy rollout for debugging or benchmarking
- Launch the setup from the command line using an argument parser

The default behavior assumes flattened observations and a discrete
action space, making it suitable for reinforcement learning workflows.

Example (Python API)
--------------------

>>> from stk_env import SuperTuxKartEnvConfig, make_supertux_env, rollout_random_policy
>>> config = SuperTuxKartEnvConfig(env_id="supertuxkart/flattened_discrete-v0", render_mode="human")
>>> env = make_supertux_env(config)
>>> rewards = rollout_random_policy(env, num_steps=50)
>>> print("Mean reward:", sum(rewards)/len(rewards))

python env.py \
    --env-id supertuxkart/flattened_discrete-v0 \
    --render \
    --agent-name "FelixAI" \
    --laps 3 \
    --num-kart 8 \
    --difficulty 2 \
    --steps 500


Example (Command Line)
----------------------

$ python stk_env.py --env-id supertuxkart/flattened_discrete-v0 --render --steps 100
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional

import gymnasium as gym
from pystk2_gymnasium import AgentSpec
from pystk2_gymnasium.definitions import CameraMode


# ---------------------------------------------------------------------------
# Supported Environment Registry
# ---------------------------------------------------------------------------

SUPPORTED_ENV_IDS = {
    "supertuxkart/full-v0",
    "supertuxkart/simple-v0",
    "supertuxkart/flattened-v0",
    "supertuxkart/flattened_continuous_actions-v0",
    "supertuxkart/flattened_multidiscrete-v0",
    "supertuxkart/flattened_discrete-v0",
}


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------

@dataclass
class SuperTuxKartEnvConfig:
    """
    Configuration container for `make_supertux_env`.

    Parameters
    ----------
    env_id : str
        The Gymnasium registry ID for the desired SuperTuxKart environment.
        Must be one of `SUPPORTED_ENV_IDS`.

    render_mode : str or None
        Rendering strategy ("human", "rgb_array") or None for headless mode.

    track : str or None
        Name of the racing track. If None, a random one is selected.

    num_kart : int
        Number of karts spawned in the race.

    laps : int
        Number of laps required to finish the episode.

    difficulty : int
        Built-in AI difficulty level (0 = Easy, 2 = Hard).

    max_paths : int or None
        If supported by the environment, controls the number of forecasted path
        nodes used in observations.

    agent_rank_start : int or None
        Starting position of the controlled kart (None → random assignment).

    agent_name : str
        Display name for the agent.

    agent_use_ai : bool
        If `True`, the internal SuperTuxKart AI controls the kart instead of RL actions.

    agent_camera_mode : CameraMode
        Camera configuration for the controlled kart.

    Example
    -------
    >>> config = SuperTuxKartEnvConfig(render_mode="human", laps=2, agent_name="MyModel")
    >>> env = make_supertux_env(config)
    """

    env_id: str = "supertuxkart/flattened_discrete-v0"
    render_mode: Optional[str] = None
    track: Optional[str] = None
    num_kart: int = 4
    laps: int = 1
    difficulty: int = 2
    max_paths: Optional[int] = 60
    agent_rank_start: Optional[int] = None
    agent_name: str = "RL-Agent"
    agent_use_ai: bool = False
    agent_camera_mode: CameraMode = CameraMode.AUTO  # type: ignore
    with_graphics: bool = False

    def build_agent(self) -> AgentSpec:
        """Return a configured `AgentSpec` instance for environment creation."""
        return AgentSpec(
            rank_start=self.agent_rank_start,
            use_ai=self.agent_use_ai,
            name=self.agent_name,
            camera_mode=self.agent_camera_mode,
        )

    def to_make_kwargs(self) -> dict:
        """Return keyword arguments required by `gym.make()`."""
        kwargs = dict(
            render_mode=self.render_mode,
            track=self.track,
            num_kart=self.num_kart,
            laps=self.laps,
            difficulty=self.difficulty,
        )
        if self.max_paths is not None:
            kwargs["max_paths"] = self.max_paths
        return kwargs


# ---------------------------------------------------------------------------
# Environment Factory
# ---------------------------------------------------------------------------

def make_supertux_env(config: Optional[SuperTuxKartEnvConfig] = None) -> gym.Env:
    """
    Create and initialize a SuperTuxKart Gymnasium environment.

    Parameters
    ----------
    config : SuperTuxKartEnvConfig or None
        Optional custom configuration. If omitted, defaults are used.

    Returns
    -------
    gym.Env
        A ready-to-use Gymnasium environment instance.

    Raises
    ------
    ValueError
        If the provided `env_id` is not supported.

    Example
    -------
    >>> env = make_supertux_env()
    >>> obs, info = env.reset()
    """
    config = config or SuperTuxKartEnvConfig()

    if config.env_id not in SUPPORTED_ENV_IDS:
        raise ValueError(
            f"Unknown env_id '{config.env_id}'. "
            f"Valid values: {sorted(SUPPORTED_ENV_IDS)}"
        )

    env = gym.make(config.env_id, agent=config.build_agent(), **config.to_make_kwargs())
    return env


# ---------------------------------------------------------------------------
# Rollout Utility
# ---------------------------------------------------------------------------

def rollout_random_policy(
    env: gym.Env,
    *,
    num_steps: int = 1000,
    seed: Optional[int] = None,
) -> List[float]:
    """
    Execute a rollout using random actions and return per-step rewards.

    This function is useful to validate an environment or test performance.

    Parameters
    ----------
    env : gym.Env
        A Gymnasium environment instance.

    num_steps : int
        Maximum number of environment steps to execute.

    seed : int or None
        Optional seed to ensure deterministic behavior.

    Returns
    -------
    list of float
        Rewards obtained at each timestep.

    Example
    -------
    >>> from stk_env import make_supertux_env, rollout_random_policy
    >>> env = make_supertux_env()
    >>> rewards = rollout_random_policy(env, num_steps=200)
    >>> print("Mean reward:", sum(rewards) / len(rewards))
    """
    env.reset(seed=seed)
    rewards = []

    for _ in range(num_steps):
        action = env.action_space.sample() # sample an action randomly
        _, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        if terminated or truncated:
            env.reset()

    return rewards


# ---------------------------------------------------------------------------
# CLI Support
# ---------------------------------------------------------------------------

def _camera_mode_names() -> List[str]:
    """Return valid names for `CameraMode` enum, excluding private attributes."""
    return sorted(
        name for name in dir(CameraMode) if name.isupper() and not name.startswith("_")
    )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for environment testing."""
    parser = argparse.ArgumentParser(
        description="Launch and optionally test a SuperTuxKart RL environment."
    )
    parser.add_argument("--env-id", default="supertuxkart/flattened_discrete-v0", choices=sorted(SUPPORTED_ENV_IDS))
    parser.add_argument("--render", action="store_true", help="Enable live rendering (human mode).")
    parser.add_argument("--track", default=None)
    parser.add_argument("--num-kart", type=int, default=4)
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--max-paths", type=int, default=60)
    parser.add_argument("--agent-rank-start", type=int, default=None)
    parser.add_argument("--agent-name", default="RL-Agent")
    parser.add_argument("--agent-use-ai", action="store_true")
    parser.add_argument("--agent-camera-mode", default="AUTO", choices=_camera_mode_names())
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _camera_mode_from_name(name: str) -> CameraMode: # type: ignore
    """Convert given string to `CameraMode` enum member."""
    try:
        return getattr(CameraMode, name.upper())
    except AttributeError:
        raise ValueError(f"Invalid camera mode '{name}'. Valid: {', '.join(_camera_mode_names())}") from None


# ---------------------------------------------------------------------------
# Script Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for CLI execution."""
    args = _parse_args()

    config = SuperTuxKartEnvConfig(
        env_id=args.env_id,
        render_mode="human" if args.render else None,
        track=args.track,
        num_kart=args.num_kart,
        laps=args.laps,
        difficulty=args.difficulty,
        max_paths=args.max_paths,
        agent_rank_start=args.agent_rank_start,
        agent_name=args.agent_name,
        agent_use_ai=args.agent_use_ai,
        agent_camera_mode=_camera_mode_from_name(args.agent_camera_mode),
    )

    env = make_supertux_env(config)

    try:
        rewards = rollout_random_policy(env, num_steps=args.steps, seed=args.seed)
        print(
            f"\nRun complete → {len(rewards)} steps\n"
            f"Average reward: {sum(rewards) / len(rewards):.3f}\n"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
