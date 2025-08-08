#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from gym_hil.mujoco_gym_env import PandaGymEnv, GymRenderingSpec

_PANDA_HOME = np.asarray((0, 0.195, 0, -2.43, 0, 2.62, 0.785))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.3, -0.15], [0.5, 0.15]])


class PandaPickCubeGymEnv(PandaGymEnv):
    """Environment for a Panda robot picking up a cube."""

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "sparse",
        random_block_position: bool = False,
    ):
        self.reward_type = reward_type

        # Load control configuration
        config_path = Path(__file__).parent.parent / "controller_config.json"
        with open(config_path) as f:
            self.control_config = json.load(f)

        xml_path = Path(__file__).parent.parent / "assets" / "scene.xml"
        super().__init__(
            xml_path=xml_path,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
            home_position=_PANDA_HOME,
            cartesian_bounds=_CARTESIAN_BOUNDS,
        )

        # Task-specific setup
        self._block_z = self._model.geom("block").size[2]
        self._random_block_position = random_block_position

        # Setup observation space properly to match what _compute_observation returns
        # Observation space design:
        #   - "state":  agent (robot) configuration as a single Box
        #   - "environment_state": block position in the world as a single Box
        #   - "pixels": (optional) dict of camera views if image observations are enabled

        self._setup_observation_space()

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        # Ensure gymnasium internal RNG is initialized when a seed is provided
        super().reset(seed=seed)

        mujoco.mj_resetData(self._model, self._data)

        # Reset the robot to home position
        self.reset_robot()

        # Sample a new block position
        if self._random_block_position:
            block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
            self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        else:
            block_xy = np.asarray([0.5, 0.0])
            self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        mujoco.mj_forward(self._model, self._data)

        # Cache the initial block height
        self._z_init = self._data.sensor("block_pos").data[2]
        self._z_success = self._z_init + 0.1

        obs = self._compute_observation()
        return obs, {}

    def step(
        self, action: np.ndarray, gamepad_state: Dict[str, Any] = None
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # If gamepad_state is provided, prioritize it over the agent's action.
        if gamepad_state:
            if gamepad_state.get("input_method") == "keyboard":
                action = self._keyboard_state_to_action(gamepad_state)
            else:
                action = self._gamepad_state_to_action(gamepad_state)

        # Apply the action to the robot
        self.apply_action(action)

        # Compute observation, reward and termination
        obs = self._compute_observation()
        rew = self._compute_reward()
        success = self._is_success()

        if self.reward_type == "sparse":
            success = rew == 1.0

        # Check if block is outside bounds
        block_pos = self._data.sensor("block_pos").data
        exceeded_bounds = np.any(block_pos[:2] < (_SAMPLING_BOUNDS[0] - 0.05)) or np.any(
            block_pos[:2] > (_SAMPLING_BOUNDS[1] + 0.05)
        )

        terminated = bool(success or exceeded_bounds)

        return obs, rew, terminated, False, {"succeed": success}

    def _compute_observation(self) -> dict:
        """Compute the current observation."""
        # Create the dictionary structure that matches our observation space
        observation = {}

        # Get robot state
        robot_state_dict = self.get_robot_state()

        # Assemble observation respecting the newly defined observation_space
        block_pos = self._data.sensor("block_pos").data.astype(np.float32)

        if self.image_obs:
            # Image observations - render_all_cameras() returns a list of frames.
            rendered_frames = self.render_all_cameras()
            front_view = rendered_frames[0] if len(rendered_frames) > 0 else np.zeros((self._render_specs.height, self._render_specs.width, 3), dtype=np.uint8)
            wrist_view = rendered_frames[1] if len(rendered_frames) > 1 else np.zeros((self._render_specs.height, self._render_specs.width, 3), dtype=np.uint8)
            observation = {
                "pixels": {"front": front_view, "wrist": wrist_view},
                "agent_pos": robot_state_dict,
            }
        else:
            # State-only observations
            block_pos = self._data.sensor("block_pos").data.astype(np.float32)
            block_quat = self._data.sensor("block_quat").data.astype(np.float32)
            environment_state = np.concatenate([block_pos, block_quat])

            observation = {
                "agent_pos": robot_state_dict,
                "environment_state": environment_state,
            }

        return observation

    def _compute_reward(self) -> float:
        """Compute reward based on current state."""
        block_pos = self._data.sensor("block_pos").data

        if self.reward_type == "dense":
            tcp_pos = self._data.sensor("2f85/pinch_pos").data.astype(np.float32)
            dist = np.linalg.norm(block_pos - tcp_pos)
            r_close = np.exp(-20 * dist)
            r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
            r_lift = np.clip(r_lift, 0.0, 1.0)
            return 0.3 * r_close + 0.7 * r_lift
        else:
            lift = block_pos[2] - self._z_init
            return float(lift > 0.1)

    def _is_success(self) -> bool:
        """Check if the task is successfully completed."""
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data.astype(np.float32)
        dist = np.linalg.norm(block_pos - tcp_pos)
        lift = block_pos[2] - self._z_init
        return dist < 0.05 and lift > 0.1


    def _setup_observation_space(self):
        """Setup the observation space for the Panda environment."""
        agent_pos_space = spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "gripper_pose": spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
                "qpos": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                "qvel": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            }
        )

        if self.image_obs:
            self.observation_space = spaces.Dict(
                {
                    "agent_pos": agent_pos_space,
                    "pixels": spaces.Dict(
                        {
                            "front": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self._render_specs.height, self._render_specs.width, 3),
                                dtype=np.uint8,
                            ),
                            "wrist": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self._render_specs.height, self._render_specs.width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "agent_pos": agent_pos_space,
                    "environment_state": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                }
            )

    def get_robot_state(self) -> Dict[str, np.ndarray]:
        """Get the current state of the robot as a dictionary."""
        # Get TCP pose (position and quaternion)
        tcp_pos = self._data.sensor("2f85/pinch_pos").data.astype(np.float32)

        # The 'pinch' site doesn't have a quaternion sensor, so we get it from the site's rotation matrix
        tcp_xmat = self._data.site("pinch").xmat
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, tcp_xmat)
        tcp_pose = np.concatenate([tcp_pos, quat]).astype(np.float32)

        # Get TCP velocity (linear and angular)
        tcp_vel = self._data.sensor("2f85/pinch_vel").data.astype(np.float32)

        # Get gripper pose
        gripper_pose = np.array([self._data.ctrl[self._gripper_ctrl_id]], dtype=np.float32)

        # Get joint state
        qpos = self.data.qpos[self._panda_dof_ids].astype(np.float32)
        qvel = self.data.qvel[self._panda_dof_ids].astype(np.float32)

        return {
            "tcp_pose": tcp_pose,
            "tcp_vel": tcp_vel,
            "gripper_pose": gripper_pose,
            "qpos": qpos,
            "qvel": qvel,
        }

    def _gamepad_state_to_action(self, gamepad_state: Dict[str, Any]) -> np.ndarray:
        """Translate a gamepad state dictionary into a 7D action vector using the loaded configuration."""
        action = np.zeros(7, dtype=np.float32)

        if "axes" not in gamepad_state or "buttons" not in gamepad_state:
            return action

        input_method = gamepad_state.get("input_method", "gamepad")
        
        # Gracefully handle the 'switch' event by returning a neutral action
        if input_method == 'switch':
            return action

        config = self.control_config["configurations"]["default_panda_pick"]
        mappings = config["mappings"][input_method]
        # Use sensitivity from gamepad_state if available, otherwise use config default
        sensitivity = gamepad_state.get("sensitivity", config["sensitivity"][input_method])

        # Get camera matrix for camera-relative control
        cam_id = self._model.camera("front").id
        cam_mat = self._data.cam_xmat[cam_id].reshape(3, 3)
        
        # Z-axis is the negative third column (camera looks along -Z)
        forward_vec = -cam_mat[:, 2]
        # Y-axis is the second column
        up_vec = cam_mat[:, 1]
        # X-axis is the first column
        right_vec = cam_mat[:, 0]

        # Movement vectors based on user input, scaled by sensitivity
        # These are now deltas in the world frame, but aligned with the camera
        move_vec = np.zeros(3)
        
        # Y-axis movement (Forward/Backward)
        y_input = 0
        if input_method == "gamepad":
            y_input = gamepad_state["axes"][mappings["y_axis"]["index"]]
        if input_method == "gamepad" and mappings["y_axis"]["inverted"]:
            y_input *= -1
        move_vec += forward_vec * y_input * sensitivity
        
        # X-axis movement (Left/Right)
        x_input = 0
        if input_method == "gamepad":
            x_input = gamepad_state["axes"][mappings["x_axis"]["index"]]
        if input_method == "gamepad" and mappings["x_axis"]["inverted"]:
            x_input *= -1
        move_vec += right_vec * x_input * sensitivity
        
        # Z-axis movement (Up/Down)
        z_input = 0
        if input_method == "gamepad":
            z_input = gamepad_state["axes"][mappings["z_axis"]["index"]]
        if input_method == "gamepad" and mappings["z_axis"]["inverted"]:
            z_input *= -1
        # Use world Z-axis for up/down movement to be intuitive
        move_vec[2] += z_input * sensitivity

        action[:3] = move_vec

        # Gripper control
        open_button = 0
        if input_method == "gamepad":
            open_button = gamepad_state["buttons"][mappings["gripper_open"]["index"]]
        close_button = 0
        if input_method == "gamepad":
            close_button = gamepad_state["buttons"][mappings["gripper_close"]["index"]]

        if open_button > 0.5:
            action[6] = 2.0  # Open
        elif close_button > 0.5:
            action[6] = 0.0  # Close
        else:
            action[6] = 1.0  # Hold

        return action


    def _keyboard_state_to_action(self, keyboard_state: Dict[str, Any]) -> np.ndarray:
        """Translate a keyboard state dictionary into a 7D action vector."""
        action = np.zeros(7, dtype=np.float32)

        if "axes" not in keyboard_state or "buttons" not in keyboard_state:
            return action

        config = self.control_config["configurations"]["default_panda_pick"]
        sensitivity = keyboard_state.get("sensitivity", config["sensitivity"]["keyboard"])

        # Get camera matrix for camera-relative control
        cam_id = self._model.camera("front").id
        cam_mat = self._data.cam_xmat[cam_id].reshape(3, 3)
        
        forward_vec = -cam_mat[:, 2]
        up_vec = cam_mat[:, 1]
        right_vec = cam_mat[:, 0]

        move_vec = np.zeros(3)
        
        # Y-axis movement (Forward/Backward)
        y_input = keyboard_state["axes"][1]
        move_vec += forward_vec * y_input * sensitivity
        
        # X-axis movement (Left/Right)
        x_input = keyboard_state["axes"][0]
        move_vec += right_vec * x_input * sensitivity
        
        # Z-axis movement (Up/Down)
        z_input = keyboard_state["axes"][3]
        move_vec[2] += z_input * sensitivity

        action[:3] = move_vec

        # Gripper control
        open_button = keyboard_state["buttons"][5]
        close_button = keyboard_state["buttons"][4]

        if open_button > 0.5:
            action[6] = 2.0  # Open
        elif close_button > 0.5:
            action[6] = 0.0  # Close
        else:
            action[6] = 1.0  # Hold

        return action


if __name__ == "__main__":
    from gym_hil import PassiveViewerWrapper

    env = PandaPickCubeGymEnv(render_mode="human")
    env = PassiveViewerWrapper(env)
    env.reset()
    for _ in range(100):
        env.step(np.random.uniform(-1, 1, 7))
    env.close()
