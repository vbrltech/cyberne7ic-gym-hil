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

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import hashlib
import time
import struct

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from gym_hil.controllers import opspace

MAX_GRIPPER_COMMAND = 255


@dataclass(frozen=True)
class GymRenderingSpec:
    height: int = 128
    width: int = 128
    camera_id: str | int = -1
    mode: Literal["rgb_array", "human"] = "rgb_array"


@dataclass
class RenderCache:
    """Cache for rendered frames and scene state."""
    frames: dict = None
    scene_hash: str = ""
    last_update_time: float = 0.0
    cache_ttl: float = 0.016  # 60fps cache timeout (16ms)
    
    def __post_init__(self):
        if self.frames is None:
            self.frames = {}


class MujocoGymEnv(gym.Env):
    """MujocoEnv with gym interface."""

    def __init__(
        self,
        xml_path: Path,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
    ):
        self._model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        self._model.vis.global_.offwidth = render_spec.width
        self._model.vis.global_.offheight = render_spec.height
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = physics_dt
        self._control_dt = control_dt
        self._n_substeps = int(control_dt // physics_dt)
        self._random = np.random.RandomState(seed)
        self._viewer: Optional[mujoco.Renderer] = None
        self._render_specs = render_spec
        
        # Render caching for performance optimization
        self._render_cache = RenderCache()
        self._scene_needs_update = True

    def _compute_scene_hash(self) -> str:
        """Compute a hash of the current simulation state for caching."""
        # Use key simulation state components for hashing
        state_components = [
            struct.pack('d', self._data.time),  # Convert float to bytes
            self._data.qpos.tobytes(),
            self._data.qvel.tobytes(),
            self._data.ctrl.tobytes(),
        ]
        
        # Include mocap data if available
        if self._data.mocap_pos is not None and self._data.mocap_pos.size > 0:
            state_components.append(self._data.mocap_pos.tobytes())
        if self._data.mocap_quat is not None and self._data.mocap_quat.size > 0:
            state_components.append(self._data.mocap_quat.tobytes())
            
        combined_state = b''.join(state_components)
        return hashlib.md5(combined_state).hexdigest()

    def _should_update_cache(self) -> bool:
        """Determine if the render cache should be updated."""
        current_time = time.time()
        current_hash = self._compute_scene_hash()
        
        # Check if scene state has changed or cache has expired
        if (current_hash != self._render_cache.scene_hash or
            current_time - self._render_cache.last_update_time > self._render_cache.cache_ttl):
            return True
        return False

    def _update_scene_if_needed(self, camera_id):
        """Update scene only if necessary for performance optimization."""
        if self._scene_needs_update or self._should_update_cache():
            self._viewer.update_scene(self._data, camera=camera_id)
            self._scene_needs_update = False

    def render(self):
        if self._viewer is None:
            self._viewer = mujoco.Renderer(
                model=self._model,
                height=self._render_specs.height,
                width=self._render_specs.width,
            )
            
        camera_id = self._render_specs.camera_id
        current_hash = self._compute_scene_hash()
        
        # Check cache first
        if (current_hash == self._render_cache.scene_hash and
            camera_id in self._render_cache.frames):
            return self._render_cache.frames[camera_id]
        
        # Update scene only if needed
        self._update_scene_if_needed(camera_id)
        frame = self._viewer.render()
        
        # Update cache
        if self._should_update_cache():
            self._render_cache.scene_hash = current_hash
            self._render_cache.last_update_time = time.time()
            self._render_cache.frames = {camera_id: frame}
        
        return frame

    def invalidate_render_cache(self):
        """Invalidate the render cache to force re-rendering."""
        self._render_cache.scene_hash = ""
        self._render_cache.frames.clear()
        self._scene_needs_update = True

    def close(self) -> None:
        """Release graphics resources if they exist.

        In MuJoCo < 2.3.0 `mujoco.Renderer` had no `close()` member.  Calling
        it unconditionally therefore raises `AttributeError`.  We check for
        the attribute first and fall back to a no-op, keeping compatibility
        across MuJoCo versions.
        """

        viewer = self._viewer
        if viewer is None:
            return

        if hasattr(viewer, "close") and callable(viewer.close):
            try:  # noqa: SIM105
                viewer.close()
            except Exception:
                # Ignore errors coming from already freed OpenGL contexts or
                # older MuJoCo builds.
                pass

        self._viewer = None

    # Accessors.

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def control_dt(self) -> float:
        return self._control_dt

    @property
    def physics_dt(self) -> float:
        return self._model.opt.timestep

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random


class MujocoRobotEnv(MujocoGymEnv):
    """Base class for Mujoco robot environments."""

    def __init__(
        self,
        xml_path: Path,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        home_position: np.ndarray = np.zeros(7),
        cartesian_bounds: np.ndarray = np.asarray([[-1, -1, -1], [1, 1, 1]]),
    ):
        super().__init__(
            xml_path=xml_path,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
        )

        self._home_position = home_position
        self._cartesian_bounds = cartesian_bounds

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.image_obs = image_obs

        # Setup cameras - with fallback for missing cameras
        camera_name_1 = "front"
        camera_name_2 = "handcam_rgb"
        
        try:
            camera_id_1 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_1)
        except Exception as e:
            print(f"Warning: Camera '{camera_name_1}' not found, using default camera")
            camera_id_1 = -1  # Default camera
            
        try:
            camera_id_2 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_2)
        except Exception as e:
            print(f"Warning: Camera '{camera_name_2}' not found, using same as primary camera")
            camera_id_2 = camera_id_1  # Use same camera as fallback
            
        self.camera_id = (camera_id_1, camera_id_2)

        # Setup observation and action spaces
        self._setup_observation_space()
        self._setup_action_space()

        # Initialize renderer
        self._viewer = mujoco.Renderer(self.model, height=render_spec.height, width=render_spec.width)
        self._viewer.render()
        
        # Initialize multi-camera render cache
        self._multi_camera_cache = RenderCache()

    def _setup_observation_space(self):
        raise NotImplementedError

    def _setup_action_space(self):
        raise NotImplementedError

    def reset_robot(self):
        raise NotImplementedError

    def apply_action(self, action):
        raise NotImplementedError

    def get_robot_state(self):
        raise NotImplementedError

    def get_gripper_pose(self):
        raise NotImplementedError

    def render(self) -> np.ndarray:
        """Renders the environment from the default camera for gymnasium compatibility."""
        camera_id_to_render = self.camera_id[0]
        current_hash = self._compute_scene_hash()
        
        # Check single camera cache
        cache_key = f"single_{camera_id_to_render}"
        if (current_hash == self._render_cache.scene_hash and
            cache_key in self._render_cache.frames):
            return self._render_cache.frames[cache_key]
        
        # Update scene and render
        self._update_scene_if_needed(camera_id_to_render)
        frame = self._viewer.render()
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        
        # Update cache
        if self._should_update_cache():
            self._render_cache.scene_hash = current_hash
            self._render_cache.last_update_time = time.time()
            self._render_cache.frames[cache_key] = frame
        
        return frame

    def render_all_cameras(self) -> list[np.ndarray]:
        """Renders the environment from all available cameras with optimized caching."""
        current_hash = self._compute_scene_hash()
        cache_key = "all_cameras"
        
        # Check multi-camera cache
        if (current_hash == self._multi_camera_cache.scene_hash and
            cache_key in self._multi_camera_cache.frames):
            return self._multi_camera_cache.frames[cache_key]
        
        # Render all cameras with single scene update
        frames = []
        scene_updated = False
        
        for i, camera_id in enumerate(self.camera_id):
            try:
                # Only update scene once for all cameras
                if not scene_updated:
                    self._update_scene_if_needed(camera_id)
                    scene_updated = True
                else:
                    # Just switch camera without full scene update
                    self._viewer.update_scene(self.data, camera=camera_id)
                
                frame = self._viewer.render()
                if not isinstance(frame, np.ndarray):
                    frame = np.array(frame)
                frames.append(frame)
            except Exception as e:
                print(f"Warning: Failed to render camera {i} (id={camera_id}): {e}")
                # Create a black frame as fallback
                fallback_frame = np.zeros((self._render_specs.height, self._render_specs.width, 3), dtype=np.uint8)
                frames.append(fallback_frame)
        
        # Update multi-camera cache
        if self._should_update_cache():
            self._multi_camera_cache.scene_hash = current_hash
            self._multi_camera_cache.last_update_time = time.time()
            self._multi_camera_cache.frames[cache_key] = frames
        
        return frames

    def step(self, action):
        """Override step to invalidate cache when simulation state changes."""
        result = super().step(action) if hasattr(super(), 'step') else None
        self.invalidate_render_cache()
        return result

    def reset(self, **kwargs):
        """Override reset to invalidate cache when environment resets."""
        result = super().reset(**kwargs) if hasattr(super(), 'reset') else None
        self.invalidate_render_cache()
        return result


class PandaGymEnv(MujocoRobotEnv):
    """Franka Panda robot environment."""

    def __init__(
        self,
        xml_path: Path | None = None,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        home_position: np.ndarray = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4)),  # noqa: B008
        cartesian_bounds: np.ndarray = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]]),  # noqa: B008
    ):
        if xml_path is None:
            xml_path = Path(__file__).parent.parent / "assets" / "scene.xml"

        super().__init__(
            xml_path=xml_path,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
            home_position=home_position,
            cartesian_bounds=cartesian_bounds,
        )

        # Cache robot IDs
        self._panda_dof_ids = np.asarray([self._model.joint(f"joint{i}").id for i in range(1, 8)])
        self._panda_ctrl_ids = np.asarray([self._model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self._pinch_site_id = self._model.site("pinch").id

    def _setup_observation_space(self):
        """Setup the observation space for the Franka environment."""
        base_obs_space = {
            "agent_pos": spaces.Dict(
                {
                    "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                    "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                    "gripper_pose": spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
                }
            )
        }

        self.observation_space = spaces.Dict(base_obs_space)

        if self.image_obs:
            self.observation_space = spaces.Dict(
                {
                    **base_obs_space,
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

    def _setup_action_space(self):
        """Setup the action space for the Franka environment."""
        self.action_space = spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset_robot(self):
        """Reset the robot to home position."""
        self._data.qpos[self._panda_dof_ids] = self._home_position
        self._data.ctrl[self._panda_ctrl_ids] = 0.0
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
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
                dof_ids=self._panda_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=self._home_position,
                gravity_comp=True,
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)
        
        # Invalidate render cache after physics step
        self.invalidate_render_cache()

    def get_robot_state(self):
        """Get the current state of the robot."""
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        # tcp_quat = self._data.sensor("2f85/pinch_quat").data
        # tcp_vel = self._data.sensor("2f85/pinch_vel").data
        # tcp_angvel = self._data.sensor("2f85/pinch_angvel").data
        qpos = self.data.qpos[self._panda_dof_ids].astype(np.float32)
        qvel = self.data.qvel[self._panda_dof_ids].astype(np.float32)
        gripper_pose = self.get_gripper_pose()

        return np.concatenate([qpos, qvel, gripper_pose, tcp_pos])

    def get_gripper_pose(self):
        """Get the current pose of the gripper."""
        return np.array([self._data.ctrl[self._gripper_ctrl_id]], dtype=np.float32)
