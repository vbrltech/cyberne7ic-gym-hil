#!/usr/bin/env python

import json
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from gym_hil.controllers import opspace
from gym_hil.mujoco_gym_env import MujocoRobotEnv, GymRenderingSpec

_SO101_HOME = np.asarray((0, 0, 0, 0, 0, 0))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.3, -0.15], [0.5, 0.15]])


class SO101PickCubeGymEnv(MujocoRobotEnv):
    """Environment for a SO-101 robot picking up a cube."""

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
        use_viewer: bool = False,
    ):
        self.reward_type = reward_type

        # Load control configuration
        config_path = Path(__file__).parent.parent / "controller_config.json"
        with open(config_path) as f:
            self.control_config = json.load(f)

        super().__init__(
            xml_path=Path(__file__).parent.parent / "assets" / "so101_pick_cube.xml",
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
            home_position=_SO101_HOME,
            cartesian_bounds=_CARTESIAN_BOUNDS,
        )

        # Task-specific setup
        self._block_z = self._model.geom("block").size[2]
        self._random_block_position = random_block_position

        # Cache robot IDs
        self._so101_dof_ids = np.asarray([self._model.joint(name).id for name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]])
        self._so101_ctrl_ids = np.asarray([self._model.actuator(name).id for name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]])
        self._gripper_ctrl_id = self._model.actuator("gripper").id
        self._pinch_site_id = self._model.site("gripperframe").id

        # Setup observation space properly to match what _compute_observation returns
        # Observation space design:
        #   - "state":  agent (robot) configuration as a single Box
        #   - "environment_state": block position in the world as a single Box
        #   - "pixels": (optional) dict of camera views if image observations are enabled

        self._setup_observation_space()
        self._setup_action_space()

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
        # Get robot state as a dictionary
        robot_state_dict = self.get_robot_state()

        if self.image_obs:
            # Image observations - render() now returns dual camera views as numpy array
            rendered_frames = self.render()
            front_view = rendered_frames[0]
            wrist_view = rendered_frames[1]
            observation = {
                "pixels": {"front": front_view, "wrist": wrist_view},
                "agent_pos": robot_state_dict,
            }
        else:
            # State-only observations
            block_pos = self._data.sensor("block_pos").data.astype(np.float64)
            block_quat = self._data.sensor("block_quat").data.astype(np.float64)
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
            tcp_pos = self._data.sensor("gripperframe").data
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
        tcp_pos = self._data.sensor("gripperframe").data
        dist = np.linalg.norm(block_pos - tcp_pos)
        lift = block_pos[2] - self._z_init
        return dist < 0.05 and lift > 0.1

    def _setup_observation_space(self):
        """Setup the observation space for the SO101 environment."""
        agent_pos_space = spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "gripper_pose": spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
                "qpos": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "qvel": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
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

    def _setup_action_space(self):
        """Setup the action space for the SO101 environment."""
        self.action_space = spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset_robot(self):
        """Reset the robot to home position."""
        self._data.qpos[self._so101_dof_ids] = self._home_position
        self._data.ctrl[self._so101_ctrl_ids] = 0.0
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position
        tcp_pos = self._data.sensor("gripperframe").data
        self._data.mocap_pos[0] = tcp_pos

    def apply_action(self, action):
        """Apply the action to the robot."""
        if len(action) == 7:
            x, y, z, rx, ry, rz, grasp_command = action
        elif len(action) == 4:
            x, y, z, grasp_command = action
            rx, ry, rz = 0, 0, 0
        else:
            raise ValueError(f"Action length must be 4 or 7, not {len(action)}")

        # Set the mocap position
        pos = self._data.mocap_pos[0].copy()
        dpos = np.asarray([x, y, z])
        npos = np.clip(pos + dpos, *self._cartesian_bounds)
        self._data.mocap_pos[0] = npos

        # Set gripper grasp
        # The grasp_command is an absolute state: 0.0 for close, 1.0 for hold, 2.0 for open.
        if grasp_command == 0.0:  # Close command
            # The Robotiq gripper controller expects a value from 0 to 255.
            # We send a high value to close the gripper.
            self._data.ctrl[self._gripper_ctrl_id] = 255.0
        elif grasp_command == 2.0:  # Open command
            self._data.ctrl[self._gripper_ctrl_id] = 0.0
        # If grasp_command is 1.0 (hold), we do nothing, maintaining the current state.

        # Apply operational space control
        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._pinch_site_id,
                dof_ids=self._so101_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=self._home_position,
                gravity_comp=True,
            )
            self._data.ctrl[self._so101_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

    def get_robot_state(self) -> Dict[str, np.ndarray]:
        """Get the current state of the robot as a dictionary."""
        # Get TCP pose (position and quaternion)
        tcp_pos = self._data.sensor("gripperframe").data.astype(np.float64)

        # The 'gripperframe' site doesn't have a quaternion sensor, so we get it from the site's rotation matrix
        tcp_xmat = self._data.site("gripperframe").xmat.astype(np.float64)
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, tcp_xmat)
        tcp_pose = np.concatenate([tcp_pos, quat.astype(np.float64)]).astype(np.float64)

        # Get TCP velocity (linear and angular)
        # These sensors are not defined in the XML, so we use placeholders
        tcp_vel = np.zeros(6, dtype=np.float64)

        # Get gripper pose
        gripper_pose = self.get_gripper_pose()

        # Get joint state
        qpos = self.data.qpos[self._so101_dof_ids].astype(np.float64)
        qvel = self.data.qvel[self._so101_dof_ids].astype(np.float64)

        return {
            "tcp_pose": tcp_pose,
            "tcp_vel": tcp_vel,
            "gripper_pose": gripper_pose,
            "qpos": qpos,
            "qvel": qvel,
        }

    def get_gripper_pose(self):
        """Get the current pose of the gripper."""
        return np.array([self._data.ctrl[self._gripper_ctrl_id]], dtype=np.float64)

    def _gamepad_state_to_action(self, gamepad_state: Dict[str, Any]) -> np.ndarray:
        """Translate a gamepad state dictionary into a 7D action vector using the loaded configuration."""
        action = np.zeros(7, dtype=np.float64)

        if "axes" not in gamepad_state or "buttons" not in gamepad_state:
            return action

        input_method = gamepad_state.get("input_method", "gamepad")
        
        # Gracefully handle the 'switch' event by returning a neutral action
        if input_method == 'switch':
            return action

        config = self.control_config["configurations"]["default_so101_pick"]
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

        config = self.control_config["configurations"]["default_so101_pick"]
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

    env = SO101PickCubeGymEnv(render_mode="human")
    env = PassiveViewerWrapper(env)
    env.reset()
    for _ in range(100):
        env.step(np.random.uniform(-1, 1, 7))
    env.close()