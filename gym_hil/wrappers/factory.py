#!/usr/bin/env python

from typing import TypedDict

import gymnasium as gym

from gym_hil.envs.panda_arrange_boxes_gym_env import PandaArrangeBoxesGymEnv
from gym_hil.envs.panda_pick_gym_env import PandaPickCubeGymEnv
from gym_hil.wrappers.viewer_wrapper import PassiveViewerWrapper


class EEActionStepSize(TypedDict):
    x: float
    y: float
    z: float


def wrap_env(
    env: gym.Env,
    ee_step_size: EEActionStepSize | None = None,
    use_viewer: bool = False,
    use_gamepad: bool = False,
    use_gripper: bool = True,
    auto_reset: bool = False,
    show_ui: bool = True,
    gripper_penalty: float = -0.02,
    reset_delay_seconds: float = 1.0,
    controller_config_path: str = None,
) -> gym.Env:
    """Apply wrappers to an environment based on configuration.

    Args:
        env: The base environment to wrap
        ee_step_size: Step size for movement in meters
        use_viewer: Whether to add a passive viewer
        use_gamepad: Whether to use gamepad instead of keyboard controls
        use_gripper: Whether to enable gripper control
        auto_reset: Whether to automatically reset the environment when episode ends
        show_ui: Whether to show UI panels in the viewer
        gripper_penalty: Penalty for using the gripper
        reset_delay_seconds: The number of seconds to delay during reset
        controller_config_path: Path to the controller configuration JSON file

    Returns:
        The wrapped environment
    """
    # The wrappers that were here have been removed as they are no longer needed.
    # The environment now handles gamepad input programmatically.
    if use_viewer:
        env = PassiveViewerWrapper(env, show_left_ui=show_ui, show_right_ui=show_ui)

    return env


def make_env(
    env_id: str,
    ee_step_size: EEActionStepSize | None = None,
    use_viewer: bool = False,
    use_gamepad: bool = False,
    use_gripper: bool = True,
    auto_reset: bool = False,
    show_ui: bool = True,
    gripper_penalty: float = -0.02,
    reset_delay_seconds: float = 1.0,
    controller_config_path: str | None = None,
    **kwargs,
) -> gym.Env:
    """Create and wrap an environment in a single function.

    Args:
        env_id: The ID of the base environment to create
        ee_step_size: Step size for movement in meters
        use_viewer: Whether to add a passive viewer
        use_gamepad: Whether to use gamepad instead of keyboard controls
        use_gripper: Whether to enable gripper control
        auto_reset: Whether to automatically reset the environment when episode ends
        show_ui: Whether to show UI panels in the viewer
        gripper_penalty: Penalty for using the gripper
        reset_delay_seconds: The number of seconds to delay during reset
        controller_config_path: Path to the controller configuration JSON file
        **kwargs: Additional arguments to pass to the base environment

    Returns:
        The wrapped environment
    """
    # Create the base environment directly
    if env_id == "gym_hil/PandaPickCubeBase-v0":
        env = PandaPickCubeGymEnv(**kwargs)
    elif env_id == "gym_hil/PandaArrangeBoxesBase-v0":
        env = PandaArrangeBoxesGymEnv(**kwargs)
    else:
        raise ValueError(f"Environment ID {env_id} not supported")

    return wrap_env(
        env,
        ee_step_size=ee_step_size,
        use_viewer=use_viewer,
        use_gamepad=use_gamepad,
        use_gripper=use_gripper,
        auto_reset=auto_reset,
        show_ui=show_ui,
        gripper_penalty=gripper_penalty,
        reset_delay_seconds=reset_delay_seconds,
        controller_config_path=controller_config_path,
    )
