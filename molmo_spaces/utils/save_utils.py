import json
import logging
import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import cv2
import h5py
import imageio
import numpy as np
import torch

from molmo_spaces.env.abstract_sensors import SensorSuite

log = logging.getLogger(__name__)

COMPR = "lzf"  # default H5 compression to use.
SAVED_VIDEO_SENSORS_KEY = "_saved_video_sensors"


def is_camera_sensor(sensor_name: str, sensor_suite: SensorSuite | None = None) -> bool:
    """
    Determine if a sensor corresponds to a camera (RGB or depth) that produces image data.

    Uses sensor type metadata when available (preferred), falls back to naming heuristics
    for backward compatibility when sensor_suite is not provided.

    Args:
        sensor_name: Name of the sensor to check
        sensor_suite: Optional SensorSuite to query for sensor type metadata

    Returns:
        True if the sensor is a camera that produces image data (RGB or depth), False otherwise.
        Returns False for camera parameter sensors (CameraParameterSensor) which contain
        metadata but not image data.
    """
    # Import here to avoid circular dependency
    from molmo_spaces.env.sensors_cameras import CameraParameterSensor, CameraSensor, DepthSensor

    # Preferred approach: Check sensor type directly via SensorSuite
    if sensor_suite is not None and sensor_name in sensor_suite.sensors:
        sensor = sensor_suite.sensors[sensor_name]
        # Check if it's a CameraSensor or DepthSensor (but not CameraParameterSensor)
        return isinstance(sensor, CameraSensor | DepthSensor) and not isinstance(
            sensor, CameraParameterSensor
        )


def byte_array_to_string(bytes_to_decode: np.ndarray):
    return bytes(bytes_to_decode).rstrip(b"\x00").decode("utf-8")


def dict_to_byte_array(target_dict, sensor_name: str, str_max_len: int):
    data_string = json.dumps(target_dict, sort_keys=True, separators=(",", ":"))
    if len(data_string) > str_max_len:
        log.warning(
            f"Warning: Truncated JSON string to {str_max_len} characters for {sensor_name}. Data values may be missing."
        )
    byte_array = np.zeros(str_max_len, dtype=np.uint8)
    encoded = data_string.encode("utf-8")[:str_max_len]
    byte_array[: len(encoded)] = list(encoded)
    return byte_array


def _debug_structure(data, max_depth=2, current_depth=0):
    """Helper function to show the structure of nested dictionaries for debugging."""
    if current_depth >= max_depth:
        return "..."

    if isinstance(data, dict):
        return {
            k: _debug_structure(v, max_depth, current_depth + 1) for k, v in list(data.items())[:3]
        }
    elif isinstance(data, list):
        return f"list[{len(data)}]({type(data[0]).__name__ if data else 'empty'})"
    elif hasattr(data, "shape"):  # tensor or numpy array
        return f"{type(data).__name__}({data.shape})"
    else:
        return f"{type(data).__name__}"


def safe_to_tensor(data):
    """Safely convert data to tensor, handling different dimensionalities."""
    if isinstance(data, np.ndarray):
        # For 1D arrays (sensor data like poses, joint positions), convert directly to tensor
        if data.ndim == 1:
            return torch.from_numpy(data.copy())
        # For 2D/3D arrays, convert directly without torchvision transforms
        elif data.ndim >= 2:
            # Always preserve original format for images and other multi-dimensional data
            # This avoids torchvision's to_tensor which changes HWC->CHW and uint8->float32
            return torch.from_numpy(data.copy())
        else:
            return torch.from_numpy(data.copy())
    else:
        # Try to convert to numpy first, then to tensor
        try:
            return torch.from_numpy(np.array(data))
        except ValueError:
            # Last resort: convert to numpy array and then to tensor
            return torch.from_numpy(np.array(data, dtype=np.float32))


def convert_to_arr(observations: list[dict], sensor_suite: SensorSuite) -> list[dict]:
    for observation in observations:
        for sensor in observation:
            if sensor_suite.sensors[sensor].is_dict:
                observation[sensor] = dict_to_byte_array(
                    observation[sensor], sensor, sensor_suite.sensors[sensor].str_max_len
                )
    return observations


def batch_observations(
    observations: list[dict], sensor_suite: SensorSuite, device: torch.device | None = None
) -> dict[str, dict | torch.Tensor]:
    """Transpose a batch of observation dicts to a dict of batched
    observations.

    # Arguments

    observations :  List of dicts of observations.
    device : The torch.device to put the resulting tensors on.
        Will not move the tensors if None.

    # Returns

    Transposed dict of lists of observations.
    """

    def collect_arrays(observation: dict[str, Any]) -> dict[str, dict | list]:
        """Collect raw numpy arrays/data without converting to tensors yet."""
        if not isinstance(observation, dict):
            raise TypeError(f"Expected dict observation, got {type(observation)}: {observation}")

        batch_dict: defaultdict = defaultdict(list)

        for sensor in observation:
            if isinstance(observation[sensor], dict):
                # For nested dicts, recurse
                batch_dict[sensor] = collect_arrays(observation[sensor])
            else:
                # For leaf values, just add the raw data (don't convert to tensor yet)
                batch_dict[sensor].append(observation[sensor])

        return dict(batch_dict)

    def fill_arrays(input_batch: Any, observation: dict[str, Any]) -> None:
        """Fill batch structure with raw arrays."""
        for sensor in observation:
            if isinstance(observation[sensor], dict):
                fill_arrays(input_batch[sensor], observation[sensor])
            else:
                input_batch[sensor].append(observation[sensor])

    def stack_and_tensorize(input_batch: Any) -> None:
        """Stack numpy arrays first, then convert to tensor once (more efficient)."""
        for sensor in input_batch:
            if isinstance(input_batch[sensor], dict):
                stack_and_tensorize(input_batch[sensor])
            else:
                # Stack numpy arrays first
                data_list = input_batch[sensor]
                try:
                    # Try to stack as numpy first (much faster than stacking tensors)
                    if isinstance(data_list[0], np.ndarray):
                        stacked_numpy = np.stack(data_list, axis=0)
                        input_batch[sensor] = torch.from_numpy(stacked_numpy)
                    else:
                        # Fallback: convert each to numpy, then stack
                        numpy_arrays = [
                            np.array(d) if not isinstance(d, np.ndarray) else d for d in data_list
                        ]
                        stacked_numpy = np.stack(numpy_arrays, axis=0)
                        input_batch[sensor] = torch.from_numpy(stacked_numpy)

                    # Move to device if specified
                    if device is not None:
                        input_batch[sensor] = input_batch[sensor].to(device=device)
                except Exception as e:
                    # Fallback to old method if numpy stacking fails
                    log.warning(
                        f"Fast stacking failed for {sensor}, falling back to tensor stack: {e}"
                    )
                    input_batch[sensor] = torch.stack(
                        [
                            safe_to_tensor(batch).to(device=device)
                            if device
                            else safe_to_tensor(batch)
                            for batch in data_list
                        ],
                        dim=0,
                    )

    if len(observations) == 0:
        return cast(dict[str, dict | torch.Tensor], observations)

    observations = convert_to_arr(observations, sensor_suite)
    batch = collect_arrays(observations[0])

    for obs in observations[1:]:
        fill_arrays(batch, obs)

    stack_and_tensorize(batch)

    return cast(dict[str, dict | torch.Tensor], batch)


def prepare_episode_for_saving(
    history: dict,
    sensor_suite: SensorSuite,
    fps: float,
    save_dir: str | None = None,
    episode_idx: int = 0,
    save_file_suffix: str = "",
    remove_sensors_if_save_dir: bool = True,
) -> dict[str, torch.Tensor] | None:
    """
    Transform raw episode history into batched format ready for save_trajectories().

    Takes the output of task.get_history() and produces a single dict with all data
    batched along the time dimension.

    Args:
        history: Dict from task.get_history() containing:
            - "observations": List[List[Dict]] - [timestep][batch_idx][sensor_name]
            - "rewards": List[List[float]]
            - "terminals": List[List[bool]]
            - "truncateds": List[List[bool]]
            - "actions": List[...] (optional, currently unused)
            - "obs_scene": Dict (optional)
        sensor_suite: SensorSuite for observation processing
        save_dir: Optional directory to save videos immediately (before batching)
        episode_idx: Episode index for video filenames
        save_file_suffix: Optional suffix for video filenames
        remove_sensors_if_save_dir: remove camera-related sensors if video saved

    Returns:
        Dict[str, Tensor] with all data batched along time dimension, or None if no data
        Structure:
        {
            # Sensor observations (from batch_observations)
            "qpos": Tensor(T, ...),
            "tcp_pose": Tensor(T, 7),
            ...
            # Episode metadata (camera data removed if videos saved)
            "rewards": Tensor(T,),
            "terminals": Tensor(T,),
            "truncateds": Tensor(T,),
            "successes": Tensor(T,),
            "obs_scene": str (JSON),
        }
    """
    import gc

    observations_list = history.get("observations", [])

    if not observations_list or len(observations_list) == 0:
        log.info("No observation history to save")
        return None

    # Flatten batch dimension (extract first environment since batch_size=1)
    flattened_obs = [timestep_obs[0] for timestep_obs in observations_list]

    if not flattened_obs:
        log.info("No flattened observations to save")
        return None

    log.info(f"Preparing episode data: {len(flattened_obs)} timesteps")

    # Delete original observations_list to free memory immediately
    # This removes one complete copy of all episode data
    del observations_list
    history.pop("observations", None)
    gc.collect()

    # MEMORY OPTIMIZATION: Save videos BEFORE batching to avoid massive memory spike
    # Camera images are ~80% of episode memory. By saving videos now and removing
    # images from observations, we avoid creating giant tensor copies during batching.
    if save_dir is not None:
        log.debug(f"Saving videos before batching for episode {episode_idx}")
        # Use existing function to extract and save videos from raw observations
        save_videos_from_raw_observations(
            flattened_obs,
            save_dir,
            fps,
            episode_idx=episode_idx,
            save_file_suffix=save_file_suffix,
            sensor_suite=sensor_suite,
        )

        saved_video_sensors = {}
        for sensor_name in flattened_obs[0]:
            if is_camera_sensor(sensor_name, sensor_suite) and not sensor_name.endswith("_seg"):
                saved_video_sensors[sensor_name] = (
                    "depth" if sensor_name.endswith("_depth") else "rgb"
                )

        if remove_sensors_if_save_dir:
            # CRITICAL: Delete camera data (RGB and depth) from observations to avoid batching it
            # This is where the massive memory savings come from
            removed_sensors = set()
            for obs in flattened_obs:
                sensors_to_remove = []
                for sensor_name in obs:
                    # Check if this is a camera sensor (RGB or depth)
                    # Skip segmentation sensors as they're not videos
                    if is_camera_sensor(sensor_name, sensor_suite) and not sensor_name.endswith(
                        "_seg"
                    ):
                        sensors_to_remove.append(sensor_name)

                # Remove camera data
                for sensor_name in sensors_to_remove:
                    obs.pop(sensor_name, None)
                    removed_sensors.add(sensor_name)

            if removed_sensors:
                log.debug(
                    f"Removed camera sensors from observations before batching: {removed_sensors}"
                )

        gc.collect()

    # Batch observations: List[Dict] -> Dict[str, Tensor(T, ...)]
    # Note: Camera images already removed if save_dir was provided, so this is much smaller
    batched_data = batch_observations(flattened_obs, sensor_suite)
    if save_dir is not None:
        batched_data[SAVED_VIDEO_SENSORS_KEY] = saved_video_sensors

    # Delete flattened_obs after batching to free memory
    # This removes the intermediate flattened copy
    del flattened_obs
    gc.collect()

    # Add rewards if present
    if "rewards" in history:
        rewards_list = history["rewards"]
        # Flatten batch dimension directly into numpy array (avoid intermediate list)
        rewards_array = np.array(
            [timestep_reward[0] for timestep_reward in rewards_list], dtype=np.float32
        )
        batched_data["rewards"] = torch.from_numpy(rewards_array)
        del rewards_array  # Free numpy array after conversion
        history.pop("rewards", None)

    # Add terminals if present
    if "terminals" in history:
        terminals_list = history["terminals"]
        # Flatten batch dimension directly into numpy array (avoid intermediate list)
        terminals_array = np.array(
            [timestep_terminal[0] for timestep_terminal in terminals_list], dtype=bool
        )
        batched_data["terminals"] = torch.from_numpy(terminals_array)
        del terminals_array
        history.pop("terminals", None)

    # Add truncateds if present
    if "truncateds" in history:
        truncateds_list = history["truncateds"]
        # Flatten batch dimension directly into numpy array (avoid intermediate list)
        truncateds_array = np.array(
            [timestep_truncated[0] for timestep_truncated in truncateds_list], dtype=bool
        )
        batched_data["truncateds"] = torch.from_numpy(truncateds_array)
        del truncateds_array
        history.pop("truncateds", None)

    # Add successes if present
    if "successes" in history:
        successes_list = history["successes"]
        # Flatten batch dimension directly into numpy array (avoid intermediate list)
        successes_array = np.array(
            [timestep_success[0] for timestep_success in successes_list], dtype=bool
        )
        batched_data["successes"] = torch.from_numpy(successes_array)
        del successes_array
        history.pop("successes", None)

    # Add obs_scene if present
    if "obs_scene" in history:
        batched_data["obs_scene"] = json.dumps(history["obs_scene"])

    # Final GC to clean up any remaining references
    gc.collect()

    return batched_data


def _save_sensor_video(
    sensor_name: str,
    sensor_type: str,
    frames: np.ndarray | list[np.ndarray],
    video_path: str,
    fps: float,
    logger: logging.Logger | None = None,
) -> None:
    """
    Save a single sensor's video (RGB or depth).

    Single source of truth for all sensor video saving logic.
    Handles shape validation, dtype conversion, and routing to appropriate save function.

    Args:
        sensor_name: Name of the sensor (for error messages)
        sensor_type: Either "rgb" or "depth"
        frames: Video frames as numpy array or list of arrays
        video_path: Path to save the video
        fps: Frames per second
        logger: Optional logger instance
    """
    from molmo_spaces.utils.depth_utils import save_depth_video

    logger = logger or log

    # Stack frames if needed
    if isinstance(frames, list):
        frames = np.array(frames)

    if sensor_type == "depth":
        # Handle depth sensors
        if frames.dtype != np.float32:
            logger.warning(f"{sensor_name} has dtype {frames.dtype}, expected float32. Converting.")
            frames = frames.astype(np.float32)

        logger.debug(
            f"{sensor_name} (depth): shape={frames.shape}, "
            f"dtype={frames.dtype}, range=[{frames.min():.3f}, {frames.max():.3f}]m"
        )

        # Handle shape: should be (T, H, W) or (T, H, W, 1)
        if frames.ndim == 4 and frames.shape[-1] == 1:
            frames = frames.squeeze(-1)
            logger.debug(f"Squeezed {sensor_name} from 4D to 3D: {frames.shape}")
        elif frames.ndim != 3:
            raise ValueError(
                f"{sensor_name} has unexpected shape {frames.shape} for depth data. "
                f"Expected (T, H, W) or (T, H, W, 1). "
                f"Check the DepthSensor observation_space and data format."
            )

        # Save depth video
        save_depth_video(frames, video_path, fps=fps, logger=logger)

    elif sensor_type == "rgb":
        # Handle RGB sensors
        logger.debug(
            f"{sensor_name} (RGB): shape={frames.shape}, "
            f"dtype={frames.dtype}, range=[{frames.min():.3f}, {frames.max():.3f}]"
        )

        # Validate shape: should be (T, H, W, C)
        if frames.ndim != 4:
            raise ValueError(
                f"{sensor_name} has unexpected shape {frames.shape} for RGB data. "
                f"Expected (T, H, W, C)."
            )

        # Check if we have CHW format instead of HWC
        if frames.shape[-1] not in [1, 3, 4] and frames.shape[1] in [1, 3, 4]:
            # Convert from (T, C, H, W) to (T, H, W, C)
            frames = np.transpose(frames, (0, 2, 3, 1))
            logger.debug(f"Converted {sensor_name} from CHW to HWC: {frames.shape}")

        # Validate channel count
        if frames.shape[-1] not in [1, 3, 4]:
            raise ValueError(
                f"{sensor_name} has invalid channel count: {frames.shape[-1]}. "
                f"Expected 1, 3, or 4 channels."
            )

        # Convert to uint8 if needed
        if frames.dtype == np.float32 or frames.dtype == np.float64:
            # Check range - should be [0,1] for RGB
            min_val = float(frames.min())
            max_val = float(frames.max())
            # Allow small tolerance for floating point errors
            if min_val < -1e-6 or max_val > 1.0 + 1e-6:
                raise ValueError(
                    f"{sensor_name}: RGB values outside valid [0,1] range: [{min_val:.6f}, {max_val:.6f}]. "
                    f"This likely indicates depth data being treated as RGB. "
                    f"Check if sensor name ends with '_depth' and is being routed correctly."
                )
            # Clip to [0, 1] and convert to uint8
            frames = np.clip(frames, 0.0, 1.0)
            frames = (frames * 255).astype(np.uint8)
            logger.debug(f"Converted {sensor_name} from float32 [0,1] to uint8 [0,255]")
        elif frames.dtype != np.uint8:
            frames = frames.astype(np.uint8)
            logger.debug(f"Converted {sensor_name} to uint8")

        # Save RGB video
        save_frames_to_mp4(frames, video_path, fps=fps)
    else:
        raise ValueError(f"Unknown sensor_type: {sensor_type}. Expected 'rgb' or 'depth'.")


def save_frames_to_mp4(
    frames: Sequence[np.ndarray],
    file_path: str,
    fps: float,
    extra_kwargs: dict[str, Any] | None = None,
) -> None:
    """Save RGB frames to MP4 video file.

    Low-level function that assumes frames are already validated and in uint8 format.
    Use _save_sensor_video() for high-level saving with validation.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    if not isinstance(frames, np.ndarray):
        frames = np.array(frames)

    # Frames should already be uint8 if coming from _save_sensor_video()
    if frames.dtype != np.uint8:
        log.warning(f"save_frames_to_mp4: Expected uint8, got {frames.dtype}. Converting...")
        if frames.dtype in [np.float32, np.float64]:
            frames = np.clip(frames, 0.0, 1.0)
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)

    # Ensure file path has .mp4 extension
    file_path = Path(file_path)
    if file_path.suffix != ".mp4":
        file_path = file_path.with_suffix(".mp4")

    kwargs = {
        "fps": fps,
        "quality": 5,
        **(extra_kwargs if extra_kwargs is not None else {}),
    }

    # Explicitly use ffmpeg plugin to avoid imageio selecting wrong plugin (e.g., tifffile)
    try:
        # Try using ffmpeg plugin explicitly
        writer = imageio.get_writer(
            file_path, format="ffmpeg", fps=fps, quality=5, **(extra_kwargs if extra_kwargs else {})
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    except (ImportError, OSError, ValueError, RuntimeError) as e:
        # Fallback to mimwrite if explicit writer fails
        # Common exceptions:
        # - ImportError: ffmpeg plugin not installed
        # - OSError: file system or ffmpeg execution issues
        # - ValueError: invalid parameters
        # - RuntimeError: ffmpeg runtime errors
        log.debug(f"FFmpeg writer failed ({type(e).__name__}: {e}), falling back to mimwrite")
        # Don't pass macro_block_size parameter as it's not supported in newer imageio versions
        imageio.mimwrite(file_path, frames, format="mp4", **kwargs)


def save_trajectories(
    episodes_data: list[dict[str, torch.Tensor]],
    save_dir: str,
    fps: float,
    save_file_suffix: str = "",
    save_mp4s: bool = True,
    logger: logging.Logger | None = None,
) -> Path:
    """
    Save trajectories in the expected hierarchical HDF5 format.

    Args:
        episodes_data: List of batched observations (output of batch_observations())
                      Each episode is a Dict[str, torch.Tensor] where tensors have shape (T, ...)
        save_dir: Directory to save files
        fps: Frames per second of episode data
        save_file_suffix: Optional suffix for filenames
        save_mp4s: Whether to save MP4 videos
        logger: Optional logger to use (defaults to module logger)

    Expected structure:
    traj_N/
    ├── obs/
    │   ├── agent/
    │   │   ├── qpos (T,str_max_len)
    │   │   └── qvel (T,str_max_len)
    │   ├── extra/
    │   │   ├── obj_start (T,7)
    │   │   ├── obj_end (T,7)
    │   │   ├── tcp_pose (T,7)
    │   │   ├── grasp_pose (T,7)
    │   │   ├── robot_base_pose (T,7)
    │   │   └── door_state (T,str_max_len)
    │   │       ├── joint_angle
    │   │       ├── opening_percentage
    │   │       ├── handle_position
    │   │       ├── handle_extents
    │   │       ├── door_position
    │   │       └── is_open
    │   ├── sensor_param/
    │   │   └── render_camera/
    │   │       ├── extrinsic_cv (T,3,4)
    │   │       ├── cam2world_gl (T,4,4)
    │   │       └── intrinsic_cv (T,3,3)
    │   └── sensor_data/
    │       └── render_camera/
    │           ├── rgb (T,str_max_len) - video path
    │           ├── depth (T,str_max_len) - video path
    │           └── segmentation (T,str_max_len) - video path
    ├── actions (T,str_max_len) - flattened
    ├── extra/ - original formats for reference
    └── episode metadata...
    """
    logger = logger or log

    os.makedirs(save_dir, exist_ok=True)

    # Save HDF5 file
    hdf5_path = os.path.join(save_dir, f"trajectories{save_file_suffix}.h5")

    with h5py.File(hdf5_path, "w") as hdf5_file:
        for episode_idx, episode_data in enumerate(episodes_data):
            episode_group = hdf5_file.create_group(f"traj_{episode_idx}")

            logger.debug(
                f"\n[SAVE_UTILS DEBUG] Processing episode {episode_idx} with batched observations"
            )
            logger.debug(f"[SAVE_UTILS DEBUG] Available sensors: {list(episode_data.keys())}")

            # Create main obs group with expected structure
            obs_group = episode_group.create_group("obs")

            # Save agent data (qpos, qvel)
            _save_agent_data_from_batched(obs_group, episode_data)

            # Save extra data (pose sensors)
            _save_extra_data_from_batched(obs_group, episode_data)

            # Save sensor parameters (camera params)
            _save_sensor_params_from_batched(obs_group, episode_data)

            # Save sensor data (camera structure)
            _save_sensor_data_from_batched(
                obs_group, episode_data, episode_idx, save_dir, save_file_suffix, logger
            )

            # Save actions
            _save_actions_from_batched(episode_group, episode_data, logger)

            # Save environment states
            _save_env_states_from_batched(episode_group, episode_data)

            # Create placeholder episode metadata (since we don't have this in batched format)
            num_timesteps = len(episode_data["qpos"])

            if "terminateds" in episode_data:
                terminated_array = episode_data["terminateds"]
            else:
                logger.warning("No terminated recorded, assuming episode ended")
                # Create dummy terminated/truncated arrays - assume last step is terminal
                terminated_array = np.zeros(num_timesteps, dtype=bool)
                terminated_array[-1] = True  # Assume episode ended
            episode_group.create_dataset("terminated", data=terminated_array, compression=COMPR)

            if "truncateds" in episode_data:
                truncated_array = episode_data["truncateds"]
            else:
                logger.warning("No truncateds recorded, assuming untruncated")
                truncated_array = np.zeros(num_timesteps, dtype=bool)
            episode_group.create_dataset("truncated", data=truncated_array, compression=COMPR)

            # Create dummy rewards (could be made configurable)
            if "rewards" in episode_data:
                rewards_array = episode_data["rewards"]
            else:
                logger.warning("No rewards recorded, assuming success")
                rewards_array = np.zeros(num_timesteps, dtype=np.float32)
                rewards_array[-1] = 1.0  # Reward at end
            episode_group.create_dataset("rewards", data=rewards_array, compression=COMPR)

            # Save obs_scene
            if "obs_scene" in episode_data:
                obs_scene = episode_data["obs_scene"]
            else:
                logger.warning("No obs_scene recorded, using default")
                obs_scene = json.dumps({})
            episode_group.create_dataset(
                "obs_scene", data=obs_scene
            )  # don't compress scalar dataset

            # Success and fail arrays
            if "successes" in episode_data:
                success_array = episode_data["successes"]
            else:
                logger.warning("No successes recorded, assuming success at end")
                success_array = np.zeros(num_timesteps, dtype=bool)
                success_array[-1] = True  # Assume success

            fail_array = ~success_array
            episode_group.create_dataset("success", data=success_array, compression=COMPR)
            episode_group.create_dataset("fail", data=fail_array, compression=COMPR)

    logger.info(f"Saved {len(episodes_data)} episodes to: {os.path.abspath(save_dir)}")
    # logger.info(f"  HDF5 file: {hdf5_path}")

    # Videos should have been saved during prepare_episode_for_saving() before batching
    # This is required for memory optimization - camera data is removed before batching
    if save_mp4s:
        if len(episodes_data) > 0:
            # Verify that camera data was removed (indicates videos were already saved)
            first_episode = episodes_data[0]
            camera_sensors_in_batch = [s for s in first_episode if is_camera_sensor(s)]

            if camera_sensors_in_batch:
                raise RuntimeError(
                    f"Camera data still present in batched episodes: {camera_sensors_in_batch}. "
                    f"Videos must be saved via save_videos_from_raw_observations() before batching. "
                    f"Pass save_dir to prepare_episode_for_saving() to enable this."
                )

        logger.debug("Videos were saved during prepare_episode_for_saving() (before batching)")
    return Path(hdf5_path)


def _save_agent_data_from_batched(obs_group, episode_data) -> None:
    """Save agent qpos/qvel data from batched observations."""
    agent_group = obs_group.create_group("agent")

    if "qpos" in episode_data:
        if "qpos" in episode_data:
            qpos_tensor = episode_data["qpos"]
            qpos_numpy = qpos_tensor.detach().cpu().numpy()
            agent_group.create_dataset("qpos", data=qpos_numpy, compression=COMPR)
    else:
        log.warning("No qpos found in episode_data, cannot save qpos!")

    if "qvel" in episode_data or "qvel_dict" in episode_data:
        if "qvel" in episode_data:
            qvel_tensor = episode_data["qvel"]
            qvel_numpy = qvel_tensor.detach().cpu().numpy()
            agent_group.create_dataset("qvel", data=qvel_numpy, compression=COMPR)
    else:
        log.warning("No qvel found in episode_data, cannot save qvel!")


def _save_extra_data_from_batched(obs_group, episode_data) -> None:
    """Save extra task data (pose sensors) from batched observations."""
    extra_group = obs_group.create_group("extra")

    # TODO(max): why do we have this???
    extra_sensor_mapping = {
        # Standard object pose sensors
        "obj_start_pose": "obj_start",
        "obj_end_pose": "obj_end",
        "grasp_state_pickup_obj": "grasp_state_pickup_obj",
        "grasp_state_place_receptacle": "grasp_state_place_receptacle",
        # Task info sensor
        "task_info": "task_info",
        # RBY1 door opening pose sensors
        "door_start_pose": "obj_start",
        "door_end_pose": "obj_end",
        # RBY1 door state sensors
        "door_state": "door_state",
        "door_state_dict": "door_state_dict",
        # Single arm TCP sensors
        "tcp_pose": "tcp_pose",
        "grasp_pose": "grasp_pose",
        "release_pose": "release_pose",
        "phase_target_pose": "phase_target_pose",
        # RBY1 dual-arm TCP sensors
        "left_tcp_pose": "left_tcp_pose",
        "right_tcp_pose": "right_tcp_pose",
        # RBY1 grasp state sensors
        "rby1_left_grasp_state": "rby1_left_grasp_state",
        "rby1_right_grasp_state": "rby1_right_grasp_state",
        # Base pose sensor
        "robot_base_pose": "robot_base_pose",
        # Policy sensors
        "object_contacts": "object_contacts",
        "pickup_obj_pose": "pickup_obj_pose",
        "curobo_plan": "curobo_plan",
        "curobo_spheres": "curobo_spheres",
        "curobo_obstacles": "curobo_obstacles",
        "policy_phase": "policy_phase",
        "policy_num_retries": "policy_num_retries",
        "grasp_plan_diagnostics": "grasp_plan_diagnostics",
        # Object tracking sensors
        "object_image_points": "object_image_points",
    }

    def _save_nested_data(data, group, name_prefix=""):
        """Recursively save nested dictionary data until hitting tensors."""
        if isinstance(data, dict):
            # Create subgroup and recurse
            if name_prefix:
                subgroup = group.create_group(name_prefix)
            else:
                subgroup = group
            for key, value in data.items():
                _save_nested_data(value, subgroup, key)
        elif isinstance(data, torch.Tensor):
            data_numpy = data.detach().cpu().numpy()
            group.create_dataset(name_prefix, data=data_numpy, compression=COMPR)
        else:
            # Handle other data types (lists, primitives, etc.)
            if isinstance(data, np.ndarray):
                group.create_dataset(name_prefix, data=data, compression=COMPR)
            else:
                # Convert to numpy if possible
                try:
                    data_numpy = np.array(data)
                    group.create_dataset(name_prefix, data=data_numpy, compression=COMPR)
                except Exception as e:
                    log.warning(f"Could not save data for {name_prefix}: {type(data)}, error: {e}")

    for sensor_name, target_name in extra_sensor_mapping.items():
        if sensor_name in episode_data:
            # Use recursive loop for all sensors - handles both simple tensors and nested dicts
            sensor_data = episode_data[sensor_name]
            _save_nested_data(sensor_data, extra_group, target_name)


def _save_sensor_params_from_batched(obs_group, episode_data) -> None:
    """Save sensor parameters (camera params) from batched observations."""
    sensor_param_group = None

    for sensor_name, param_sensors in episode_data.items():
        if not sensor_name.startswith("sensor_param_"):
            continue

        if sensor_param_group is None:
            sensor_param_group = obs_group.create_group("sensor_param")

        camera_name = sensor_name.replace("sensor_param_", "")
        camera_param_group = sensor_param_group.create_group(camera_name)
        for param_type, sensor_tensor in param_sensors.items():
            sensor_numpy = sensor_tensor.detach().cpu().numpy()
            camera_param_group.create_dataset(param_type, data=sensor_numpy, compression=COMPR)


def _save_sensor_data_from_batched(
    obs_group, episode_data, episode_idx=None, save_dir=None, save_file_suffix="", logger=None
) -> None:
    """Save sensor data groups (camera structure) from batched observations.

    Note: RGB data is intentionally skipped to keep HDF5 file sizes manageable.
    Only depth and segmentation data are saved if present.

    Args:
        obs_group: HDF5 group to save sensor data to
        episode_data: Batched episode data
        episode_idx: Episode index for generating video paths
        save_dir: Directory where videos are saved
        save_file_suffix: Suffix for video filenames
        logger: Logger instance
    """
    logger = logger or log
    sensor_data_group = obs_group.create_group("sensor_data")

    # Find camera sensors (depth, segmentation) - SKIP RGB to reduce file size
    camera_sensors = dict(episode_data.get(SAVED_VIDEO_SENSORS_KEY, {}))
    for sensor_name in episode_data:
        logger.debug(f"Sensor name: {sensor_name}")
        # Skip RGB sensors to reduce HDF5 file size
        # RGB sensors: exo_camera_X, wrist_camera
        # if (sensor_name.startswith("exo_camera") and not ("_depth" in sensor_name or "_seg" in sensor_name)) or \
        #    sensor_name.startswith("wrist_camera"):
        #     camera_sensors[sensor_name] = "rgb"

        # Depth sensors
        if sensor_name.endswith("_depth"):
            camera_sensors[sensor_name] = "depth"
        # Segmentation sensors
        elif sensor_name.endswith("_seg"):
            camera_sensors[sensor_name] = "segmentation"
        elif (
            "camera" in sensor_name and "sensor_param" not in sensor_name
        ):  # TODO: this is a hack to get the rgb camera
            camera_sensors[sensor_name] = "rgb"

    # Group cameras by base name
    camera_groups = {}
    for sensor_name, sensor_type in camera_sensors.items():
        # Extract camera name from sensor name
        if sensor_name.endswith("_depth"):
            camera_name = sensor_name[:-6]  # Remove '_depth' suffix
        elif sensor_name.endswith("_seg"):
            camera_name = sensor_name[:-4]  # Remove '_seg' suffix
        else:
            camera_name = sensor_name  # Use as-is for other sensors

        if camera_name not in camera_groups:
            camera_groups[camera_name] = {}
        camera_groups[camera_name][sensor_type] = sensor_name
    # Create groups for each camera and save the actual frame data
    for camera_name, sensors in camera_groups.items():
        # Save frame data for each sensor type
        for sensor_type, sensor_name in sensors.items():
            if sensor_type == "rgb":
                # RGB: save video filename reference
                video_filename = f"episode_{episode_idx:08d}_{camera_name}{save_file_suffix}.mp4"
                # Convert string to byte array
                filename_bytes = video_filename.encode("utf-8")
                byte_array = np.zeros(100, dtype=np.uint8)  # Fixed size byte array
                byte_array[: len(filename_bytes)] = list(filename_bytes)
                sensor_data_group.create_dataset(
                    camera_name, data=byte_array, dtype=np.uint8, compression=COMPR
                )
            elif sensor_type == "depth":
                # Depth: save video filename reference (depth videos are now saved like RGB)
                video_filename = f"episode_{episode_idx:08d}_{sensor_name}{save_file_suffix}.mp4"
                filename_bytes = video_filename.encode("utf-8")
                byte_array = np.zeros(100, dtype=np.uint8)
                byte_array[: len(filename_bytes)] = list(filename_bytes)
                # Use full sensor name (e.g., "exo_camera_1_depth") as dataset name
                sensor_data_group.create_dataset(
                    sensor_name, data=byte_array, dtype=np.uint8, compression=COMPR
                )
            else:
                # Segmentation or other: save as numpy array
                sensor_tensor = episode_data[sensor_name]
                sensor_numpy = sensor_tensor.detach().cpu().numpy()
                sensor_data_group.create_dataset(sensor_name, data=sensor_numpy, compression=COMPR)


def _save_actions_from_batched(episode_group, episode_data, logger=None) -> None:
    """Save actions from batched observations."""
    logger = logger or log

    action_keys: list[str] = [k for k in episode_data if k.startswith("actions/")]

    actions_group = episode_group.create_group("actions")

    if action_keys:
        for action_key in action_keys:
            actions_tensor = episode_data[action_key]
            action_key_bn = action_key.replace("actions/", "")
            actions_numpy = actions_tensor.detach().cpu().numpy()
            actions_group.create_dataset(action_key_bn, data=actions_numpy, compression=COMPR)
            logger.debug(f"[SAVE_UTILS] Successfully saved {action_key_bn} from batched data")
    else:
        logger.warning("[SAVE_UTILS] No actions found in batched data, leaving actions group empty")


def _save_env_states_from_batched(episode_group, episode_data) -> None:
    """Save environment states from batched observations."""
    env_states_group = episode_group.create_group("env_states")

    if "env_states" not in episode_data:
        # Create empty groups
        env_states_group.create_group("actors")
        articulations_group = env_states_group.create_group("articulations")
        num_timesteps = len(episode_data[list(episode_data.keys())[0]])
        dummy_articulation_state = np.zeros((num_timesteps, 31), dtype=np.float32)
        articulations_group.create_dataset(
            "panda", data=dummy_articulation_state, compression=COMPR
        )
        return

    # Handle env_states tensor - need to decode JSON from each timestep
    env_states_tensor = episode_data["env_states"]
    env_states_numpy = env_states_tensor.detach().cpu().numpy()

    # Decode JSON data from each timestep
    env_states_data = []
    for timestep_data in env_states_numpy:
        # Find the end of the actual data (before null padding)
        actual_length = np.where(timestep_data == 0)[0]
        if len(actual_length) > 0:
            timestep_data = timestep_data[: actual_length[0]]

        json_string = bytes(timestep_data.astype(np.uint8)).decode("utf-8")
        parsed_data = json.loads(json_string)
        env_states_data.append(parsed_data)

    # Create actors group
    actors_group = env_states_group.create_group("actors")

    # Collect all unique actor names across all timesteps
    all_actor_names = set()
    for env_data in env_states_data:
        all_actor_names.update(env_data.get("actors", {}).keys())

    # Create datasets for each discovered actor
    for actor_name in sorted(all_actor_names):
        actor_data = []
        for env_data in env_states_data:
            if actor_name in env_data.get("actors", {}):
                actor_data.append(env_data["actors"][actor_name])
            else:
                actor_data.append([0.0] * 13)  # Fallback zeros if missing

        actors_group.create_dataset(
            actor_name, data=np.array(actor_data, dtype=np.float32), compression=COMPR
        )

    # Create articulations group
    articulations_group = env_states_group.create_group("articulations")
    articulation_data = []
    for env_data in env_states_data:
        if "panda" in env_data.get("articulations", {}):
            articulation_data.append(env_data["articulations"]["panda"])
        else:
            articulation_data.append([0.0] * 31)  # Fallback zeros

    articulations_group.create_dataset(
        "panda", data=np.array(articulation_data, dtype=np.float32), compression=COMPR
    )


def save_videos_from_raw_observations(
    observations_list,
    save_dir,
    fps,
    episode_idx=0,
    save_file_suffix="",
    sensor_suite: SensorSuite | None = None,
) -> None:
    """
    Save videos immediately from raw observations before batch processing.
    This avoids the corruption that happens during batch_observations tensor conversion.

    Args:
        observations_list: List of raw observation dicts from episode steps
        save_dir: Directory to save videos
        fps: Frames per second of episode data
        episode_idx: Episode index for naming
        save_file_suffix: Optional suffix for filenames
        sensor_suite: Optional SensorSuite for proper sensor type detection
    """
    os.makedirs(save_dir, exist_ok=True)

    if not observations_list:
        log.warning("No observations to save videos from")
        return

    # Find all camera sensors in the first observation
    camera_sensors = {}
    for sensor_name in observations_list[0]:
        if not is_camera_sensor(sensor_name, sensor_suite):
            continue

        # Categorize by type
        # Check depth FIRST (before RGB) to correctly identify depth cameras
        # e.g., "wrist_camera_depth" should be "depth" not "rgb"
        if sensor_name.endswith("_depth"):
            camera_sensors[sensor_name] = "depth"
        # Skip segmentation sensors (not saved as videos)
        elif sensor_name.endswith("_seg"):
            continue
        else:
            # All other camera sensors are RGB
            camera_sensors[sensor_name] = "rgb"

    log.debug(
        f"INFO: Saving videos for episode {episode_idx} with {len(camera_sensors)} cameras "
        f"({sum(1 for t in camera_sensors.values() if t == 'rgb')} RGB, "
        f"{sum(1 for t in camera_sensors.values() if t == 'depth')} depth) "
        f"and {len(observations_list)} frames"
    )
    log.debug(f"Camera sensors detected: {camera_sensors}")

    # Extract and save video for each camera
    for sensor_name, sensor_type in camera_sensors.items():
        # Extract frames from all observations for this camera
        frames = []
        for obs in observations_list:
            if sensor_name in obs:
                frame_data = obs[sensor_name]
                if isinstance(frame_data, np.ndarray):
                    # Remove batch dimension if present
                    if frame_data.ndim == 4 and frame_data.shape[0] == 1:
                        frame_data = frame_data[0]
                    frames.append(frame_data)

        if frames:
            # Generate video path
            video_path = os.path.join(
                save_dir, f"episode_{episode_idx:08d}_{sensor_name}{save_file_suffix}.mp4"
            )

            # Use unified video saving function (handles all validation and conversion)
            _save_sensor_video(
                sensor_name=sensor_name,
                sensor_type=sensor_type,
                frames=frames,
                video_path=video_path,
                fps=fps,
                logger=log,
            )

            if sensor_type == "rgb":
                annotated_frames = _annotate_grasp_frames(
                    frames,
                    observations_list,
                    sensor_name,
                )
                if annotated_frames is not None:
                    annotated_path = os.path.join(
                        save_dir,
                        f"episode_{episode_idx:08d}_{sensor_name}{save_file_suffix}_annotated.mp4",
                    )
                    _save_sensor_video(
                        sensor_name=sensor_name,
                        sensor_type="rgb",
                        frames=annotated_frames,
                        video_path=annotated_path,
                        fps=fps,
                        logger=log,
                    )

                waypoint_frames = _annotate_waypoint_frames(
                    frames,
                    observations_list,
                    sensor_name,
                )
                if waypoint_frames is not None:
                    waypoint_path = os.path.join(
                        save_dir,
                        f"episode_{episode_idx:08d}_{sensor_name}{save_file_suffix}_waypoints.mp4",
                    )
                    _save_sensor_video(
                        sensor_name=sensor_name,
                        sensor_type="rgb",
                        frames=waypoint_frames,
                        video_path=waypoint_path,
                        fps=fps,
                        logger=log,
                    )

            log.debug(f"SUCCESS: Saved {sensor_type} video: {video_path} ({len(frames)} frames)")
        else:
            log.warning(f"WARNING: No frames found for camera {sensor_name}")


def _decode_json_sensor(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, np.ndarray):
        value = bytes(np.asarray(value, dtype=np.uint8)).rstrip(b"\x00")
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return decoded if isinstance(decoded, dict) else None
    return None


def _grasp_diagnostic_lines(diagnostic: dict | None) -> list[str]:
    """Format grasp filtering details for a persistent video overlay."""
    if not diagnostic:
        return ["selector details: unavailable (legacy recording)"]

    status = str(diagnostic.get("status", "unknown")).replace("_", " ").upper()
    lines = [f"selector status: {status}"]
    if "candidate_count" in diagnostic:
        lines.append(
            "candidates: "
            f"raw={diagnostic.get('candidate_count', 0)}  "
            f"ranked={diagnostic.get('ranked_count', 0)}  "
            f"collision-free={diagnostic.get('collision_free_count', 0)}  "
            f"IK-checked={diagnostic.get('ik_checked_count', 0)}"
        )

    phase_counts = diagnostic.get("phase_feasible_counts", {})
    if isinstance(phase_counts, dict) and phase_counts:
        checked = int(diagnostic.get("ik_checked_count", 0))
        lines.append("per phase: individually IK-feasible; cumulative=same candidate survives")
        phase_order = ["grasp", "pregrasp", "lift", "preplace", "place"]
        phase_order.extend(name for name in phase_counts if name not in phase_order)
        for phase in phase_order:
            counts = phase_counts.get(phase)
            if not isinstance(counts, dict):
                continue
            lines.append(
                f"{phase}: IK {int(counts.get('feasible', 0))}/{checked}; "
                f"cumulative {int(counts.get('cumulative', 0))}/{checked}"
            )

    fallback = diagnostic.get("fallback")
    if isinstance(fallback, dict):
        fallback_phases = fallback.get("phase_feasible_counts", {})
        fallback_name = (
            "grasp+pregrasp"
            if isinstance(fallback_phases, dict) and "pregrasp" in fallback_phases
            else "grasp only"
        )
        lines.append(
            f"fallback ({fallback_name}): "
            f"{fallback.get('status', 'unknown')}; "
            f"jointly feasible={fallback.get('jointly_feasible_count', 0)}"
        )
    lift_fallback = diagnostic.get("lift_fallback")
    if isinstance(lift_fallback, dict):
        lines.append(
            "lift best effort: "
            f"{100 * lift_fallback.get('selected_height', 0):.1f} cm "
            f"(requested {100 * lift_fallback.get('requested_height', 0):.1f} cm)"
        )
    placement_search = diagnostic.get("placement_orientation_search")
    if isinstance(placement_search, dict):
        lines.append(
            "placement orientations: "
            f"feasible={placement_search.get('num_feasible', 0)}/"
            f"{placement_search.get('num_orientations', 0)}"
        )
        if str(placement_search.get("status", "")).startswith("selected"):
            lines.append(
                "placement selected: "
                f"X={placement_search.get('selected_x_degrees', 0):g} deg, "
                f"Y={placement_search.get('selected_y_degrees', 0):g} deg"
            )
            if "selected_preplace_clearance" in placement_search:
                lines.append(
                    "preplace clearance: "
                    f"{100 * placement_search['selected_preplace_clearance']:.1f} cm "
                    f"(requested {100 * placement_search['requested_preplace_clearance']:.1f} cm)"
                )
            if placement_search.get("selected_place_exact") is False:
                selected_attempt = next(
                    (
                        attempt
                        for attempt in placement_search.get("attempts", [])
                        if attempt.get("x_degrees") == placement_search.get("selected_x_degrees")
                        and attempt.get("y_degrees") == placement_search.get("selected_y_degrees")
                    ),
                    {},
                )
                lines.append(
                    "place best effort: "
                    f"{1000 * selected_attempt.get('place_position_error', 0):.1f} mm, "
                    f"{np.degrees(selected_attempt.get('place_rotation_error', 0)):.1f} deg"
                )
        else:
            failures = {}
            for attempt in placement_search.get("attempts", []):
                phase = attempt.get("failure_phase", "unknown")
                failures[phase] = failures.get(phase, 0) + 1
            lines.append(
                "placement failures: "
                + ", ".join(f"{phase}={count}" for phase, count in sorted(failures.items()))
            )
    note = diagnostic.get("selection_note")
    if note:
        lines.append(f"note: {note}")
    cause = diagnostic.get("failure_cause")
    if cause:
        failure_phase = diagnostic.get("failure_phase")
        prefix = f"cause ({failure_phase})" if failure_phase else "cause"
        lines.append(f"{prefix}: {cause}")
    elif status.startswith("SELECTED"):
        lines.append("cause: none; a candidate was selected")
    return lines


def _wrap_video_text(text: str, max_width: int, font_scale: float) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


_BOX_CORNERS = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
# Corner pairs differing in exactly one coordinate are the twelve edges of the box.
_BOX_EDGES = [
    (a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1
]


def _annotate_waypoint_frames(
    frames: list[np.ndarray],
    observations_list: list[dict],
    camera_name: str,
) -> list[np.ndarray] | None:
    """Overlay the planner's per-phase waypoints and the grasp point on the gripper.

    Complements :func:`_annotate_grasp_frames`, which shows what the planner was AIMING at. This
    shows what the end effector actually DID: without it a failure reads as "the arm misbehaved"
    with no way to tell a mis-aimed grasp from an unreachable one. Drawn per frame:

    * the motion planner's path for the CURRENT phase, when one was published (``curobo_plan``),
      so a planned transit is distinguishable from an unplanned straight line at a glance
    * the grasp point (``tcp_pose``, the end effector's declared TCP site) plus a fading trail
    * each phase's IK target (``phase_target_pose``), revealed when the planner chooses it and
      filled once that phase is passed
    * a connecting line and millimetre readout, so the outstanding error is legible frame by frame

    ``tcp_pose`` is published relative to the robot base group, so it is composed with
    ``robot_base_pose`` before projection -- used raw it lands behind the camera and draws nothing.
    """
    required = {"tcp_pose", "phase_target_pose", "robot_base_pose"}
    if not observations_list or not required <= set(observations_list[0]):
        return None
    parameter_name = f"sensor_param_{camera_name}"
    if parameter_name not in observations_list[0]:
        return None

    phase_names = {2: "PREGRASP", 3: "GRASP", 5: "LIFT", 7: "PLACE"}
    phase_colours = {2: (255, 200, 80), 3: (120, 255, 60), 5: (60, 190, 255), 7: (200, 160, 255)}

    def to_world(point, base_pose):
        base = np.asarray(base_pose, float).reshape(-1)
        translation, (qw, qx, qy, qz) = base[:3], base[3:7]
        rotation = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ])
        return translation + rotation @ np.asarray(point, float).reshape(-1)[:3]

    def pixel(point, extrinsic, intrinsic):
        if not np.all(np.isfinite(point)):
            return None
        camera_point = extrinsic @ np.append(point[:3], 1.0)
        if camera_point[2] <= 1e-6:
            return None
        projected = intrinsic @ camera_point
        return tuple(np.rint(projected[:2] / projected[2]).astype(int))

    annotated, waypoints, trail = [], {}, []
    for source_frame, observation in zip(frames, observations_list):
        frame = np.asarray(source_frame).copy()
        height, width = frame.shape[:2]
        camera_params = observation[parameter_name]
        extrinsic = np.asarray(camera_params["extrinsic_cv"])
        intrinsic = np.asarray(camera_params["intrinsic_cv"])
        phase = int(np.asarray(observation.get("policy_phase", 0)).reshape(-1)[0])

        # The planner's path for this phase, if the policy published one. Points arrive in the
        # base frame, like tcp_pose, so they go through the same composition.
        plan = observation.get("curobo_plan")
        if isinstance(plan, dict) and (plan.get("points") or plan.get("status") == "plan_failed"):
            rejected = plan.get("status") == "plan_failed"
            # A rejected trajectory is drawn too, in red: it is what the optimiser converged to
            # before refusing it, and seeing it is the difference between "cuRobo did nothing" and
            # "cuRobo tried hard and the route is genuinely blocked".
            colour = (60, 60, 235) if rejected else (255, 140, 60)
            base_pose = observation["robot_base_pose"]
            previous_pixel = None
            for point in plan.get("points") or []:
                projected = pixel(to_world(point, base_pose), extrinsic, intrinsic)
                if projected is None:
                    previous_pixel = None
                    continue
                if previous_pixel is not None:
                    cv2.line(frame, previous_pixel, projected, colour, 1, cv2.LINE_AA)
                cv2.circle(frame, projected, 2, colour, -1, cv2.LINE_AA)
                previous_pixel = projected
            if rejected:
                caption = (
                    f"planning FAILED - red: rejected trajectory ({len(plan['points'])} pts), "
                    f"executing straight line"
                    if plan.get("points")
                    else "planning FAILED - straight-line fallback"
                )
            else:
                caption = f"planned: {len(plan['points'])} waypoints"
            cv2.putText(frame, caption, (6, height - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, colour, 1, cv2.LINE_AA)

        # The world the planner was given for this phase. ``to_world`` applies a 7D pose to a
        # point, so a box corner goes local -> base -> world by calling it twice.
        obstacles = observation.get("curobo_obstacles")
        if isinstance(obstacles, dict) and obstacles.get("boxes"):
            base_pose = observation["robot_base_pose"]
            for box in obstacles["boxes"]:
                pose, half = box[:7], np.asarray(box[7:], dtype=float) / 2.0
                corners = [
                    pixel(
                        to_world(to_world(np.asarray(sign, dtype=float) * half, pose), base_pose),
                        extrinsic, intrinsic,
                    )
                    for sign in _BOX_CORNERS
                ]
                for first, second in _BOX_EDGES:
                    if corners[first] is not None and corners[second] is not None:
                        cv2.line(frame, corners[first], corners[second], (90, 200, 250), 1,
                                 cv2.LINE_AA)
            cv2.putText(
                frame, f"colliders: {len(obstacles['boxes'])} boxes", (6, height - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (90, 200, 250), 1, cv2.LINE_AA,
            )

        # cuRobo's own collision model at this step: what the planner routes, as opposed to what
        # the renderer draws. The payload rides on the TCP, so it composes through tcp_pose first.
        spheres = observation.get("curobo_spheres")
        if isinstance(spheres, dict) and (spheres.get("spheres") or spheres.get("payload")):
            base_pose = observation["robot_base_pose"]
            tcp = np.asarray(observation["tcp_pose"]).reshape(-1)
            focal = float(intrinsic[0][0])
            body = [(s, (140, 210, 140)) for s in spheres.get("spheres", [])]
            payload = [(s, (80, 120, 255)) for s in spheres.get("payload", [])]
            for sphere, colour in body + payload:
                is_payload = colour == (80, 120, 255)
                centre = np.asarray(sphere[:3], dtype=float)
                if is_payload:
                    centre = to_world(centre, tcp)  # TCP frame -> base frame
                point = to_world(centre, base_pose)
                projected = pixel(point, extrinsic, intrinsic)
                if projected is None:
                    continue
                depth = (extrinsic @ np.append(point, 1.0))[2]
                radius = int(round(focal * float(sphere[3]) / max(depth, 1e-6)))
                if radius >= max(height, width):
                    continue
                if is_payload:
                    # Most payload spheres are cuRobo's 1 mm surface samples, which project to a
                    # fifth of a pixel and rounded away to nothing. Drawn filled at a floor of two
                    # pixels: the point is where the planner thinks the payload is, and a marker
                    # too small to see answers that question no better than drawing nothing.
                    cv2.circle(frame, projected, max(radius, 2), colour, -1, cv2.LINE_AA)
                elif radius > 0:
                    cv2.circle(frame, projected, radius, colour, 1, cv2.LINE_AA)
            if payload:
                cv2.putText(
                    frame, f"payload: {len(payload)} spheres", (6, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (80, 120, 255), 1, cv2.LINE_AA,
                )

        target = np.asarray(observation["phase_target_pose"]).reshape(-1)
        if phase in phase_names and phase not in waypoints and np.all(np.isfinite(target[:3])):
            waypoints[phase] = target[:3].copy()

        for waypoint_phase, position in waypoints.items():
            spot = pixel(position, extrinsic, intrinsic)
            if spot is None:
                continue
            colour = phase_colours[waypoint_phase]
            cv2.circle(frame, spot, 9, colour, -1 if phase > waypoint_phase else 2, cv2.LINE_AA)
            cv2.putText(frame, phase_names[waypoint_phase], (spot[0] + 12, spot[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)

        trail.append(to_world(np.asarray(observation["tcp_pose"]).reshape(-1),
                              observation["robot_base_pose"]))
        for index, point in enumerate(trail[:-1]):
            spot = pixel(point, extrinsic, intrinsic)
            if spot is not None:
                fade = 0.25 + 0.75 * (index / max(1, len(trail) - 1))
                cv2.circle(frame, spot, 2, (int(220 * fade), int(120 * fade), int(255 * fade)),
                           -1, cv2.LINE_AA)
        spot = pixel(trail[-1], extrinsic, intrinsic)
        if spot is not None and 0 <= spot[0] < width and 0 <= spot[1] < height:
            cv2.circle(frame, spot, 7, (200, 60, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, spot, 7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "GRASP POINT", (spot[0] + 11, spot[1] + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (225, 150, 255), 1, cv2.LINE_AA)
            if phase in waypoints:
                goal = pixel(waypoints[phase], extrinsic, intrinsic)
                if goal is not None:
                    cv2.line(frame, spot, goal, (255, 255, 255), 1, cv2.LINE_AA)
                    error = float(np.linalg.norm(trail[-1] - waypoints[phase])) * 1000.0
                    cv2.putText(frame, f"{error:.0f} mm",
                                ((spot[0] + goal[0]) // 2 + 6, (spot[1] + goal[1]) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        panel = frame.copy()
        cv2.rectangle(panel, (0, 0), (width, 20), (0, 0, 0), -1)
        cv2.addWeighted(panel, 0.45, frame, 0.55, 0, frame)
        cv2.putText(frame, "magenta: grasp point on the gripper + trail | rings: phase IK targets",
                    (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (235, 235, 235), 1, cv2.LINE_AA)
        annotated.append(frame)
    return annotated


def _annotate_grasp_frames(
    frames: list[np.ndarray],
    observations_list: list[dict],
    camera_name: str,
) -> list[np.ndarray] | None:
    """Overlay pickup, planned grasp, and post-pickup release sites on RGB frames."""
    if not observations_list or "grasp_pose" not in observations_list[0]:
        return None
    parameter_name = f"sensor_param_{camera_name}"
    if parameter_name not in observations_list[0]:
        return None

    phase_names = {
        0: "unknown",
        1: "gripper-open",
        2: "pregrasp",
        3: "grasp",
        4: "gripper-close",
        5: "lift",
        6: "preplace",
        7: "place",
        8: "retreat",
        9: "go-home",
    }
    annotated = []
    pickup_completed = False
    previous_phase = None
    for frame_index, (source_frame, observation) in enumerate(zip(frames, observations_list)):
        frame = np.asarray(source_frame).copy()
        height, width = frame.shape[:2]

        grasp_pose = np.asarray(observation["grasp_pose"]).reshape(-1)
        camera_params = observation[parameter_name]
        extrinsic = np.asarray(camera_params["extrinsic_cv"])
        intrinsic = np.asarray(camera_params["intrinsic_cv"])
        point_camera = (
            None
            if not np.all(np.isfinite(grasp_pose))
            else extrinsic @ np.append(grasp_pose[:3], 1.0)
        )
        phase = int(np.asarray(observation.get("policy_phase", 0)).reshape(-1)[0])
        if phase in {2, 3, 4} or (phase == 1 and previous_phase not in {None, 1, 7}):
            pickup_completed = False
        elif phase in {5, 6, 7}:
            pickup_completed = True
        release_pose = np.asarray(observation.get("release_pose", np.full(7, np.nan))).reshape(-1)
        show_release = pickup_completed and np.all(np.isfinite(release_pose))
        release_point_camera = (
            extrinsic @ np.append(release_pose[:3], 1.0) if show_release else None
        )
        phase_target_pose = np.asarray(
            observation.get("phase_target_pose", np.full(7, np.nan))
        ).reshape(-1)
        phase_target_camera = (
            None
            if not np.all(np.isfinite(phase_target_pose))
            else extrinsic @ np.append(phase_target_pose[:3], 1.0)
        )
        header = (
            "yellow: pickup object | magenta: exact planned grasp site"
            if point_camera is not None
            else "NO VALID GRASP PLAN | yellow: pickup object"
        )
        if release_point_camera is not None:
            header += " | cyan: exact planned release site"
        if phase_target_camera is not None:
            header += " | green: active phase IK target"
        diagnostic = _decode_json_sensor(observation.get("grasp_plan_diagnostics"))
        raw_lines = [header, *_grasp_diagnostic_lines(diagnostic)]
        raw_lines.append(f"phase: {phase_names.get(phase, 'unknown')} | frame: {frame_index}")
        font_scale = 0.38
        text_lines = [
            wrapped
            for line in raw_lines
            for wrapped in _wrap_video_text(line, width - 24, font_scale)
        ]
        line_height = 16
        panel_height = min(height, 10 + line_height * len(text_lines))
        panel = frame.copy()
        cv2.rectangle(panel, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(panel, 0.42, frame, 0.58, 0, frame)
        for line_index, line in enumerate(text_lines):
            cv2.putText(
                frame,
                line,
                (12, 16 + line_index * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255) if line_index == 0 else (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

        object_points = observation.get("object_image_points", {})
        pickup_camera = object_points.get("pickup_obj", {}).get(camera_name)
        if pickup_camera is not None:
            count = int(np.asarray(pickup_camera["num_points"]).reshape(-1)[0])
            points = np.asarray(pickup_camera["points"])[:count]
            if len(points):
                pixels = points * np.array([width, height])
                lower = np.maximum(np.floor(pixels.min(axis=0) - 8), [0, 0]).astype(int)
                upper = np.minimum(
                    np.ceil(pixels.max(axis=0) + 8),
                    [width - 1, height - 1],
                ).astype(int)
                cv2.rectangle(frame, tuple(lower), tuple(upper), (255, 190, 0), 2)
                cv2.putText(
                    frame,
                    "PICKUP OBJECT",
                    (int(lower[0]), max(int(lower[1]) - 6, panel_height + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 190, 0),
                    2,
                    cv2.LINE_AA,
                )

        if point_camera is not None and point_camera[2] > 0:
            point_pixel = intrinsic @ point_camera
            point_pixel = point_pixel[:2] / point_pixel[2]
            x, y = np.rint(point_pixel).astype(int)
            if 0 <= x < width and 0 <= y < height:
                cv2.drawMarker(
                    frame,
                    (x, y),
                    (255, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=24,
                    thickness=3,
                )
                cv2.circle(frame, (x, y), 10, (255, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    "PLANNED GRASP SITE",
                    (min(x + 12, width - 190), max(y - 10, panel_height + 34)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        if release_point_camera is not None and release_point_camera[2] > 0:
            release_pixel = intrinsic @ release_point_camera
            release_pixel = release_pixel[:2] / release_pixel[2]
            release_x, release_y = np.rint(release_pixel).astype(int)
            if 0 <= release_x < width and 0 <= release_y < height:
                release_color = (0, 220, 255)
                cv2.drawMarker(
                    frame,
                    (release_x, release_y),
                    release_color,
                    markerType=cv2.MARKER_DIAMOND,
                    markerSize=24,
                    thickness=3,
                )
                cv2.circle(
                    frame,
                    (release_x, release_y),
                    10,
                    release_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "PLANNED RELEASE SITE",
                    (
                        min(release_x + 12, width - 200),
                        max(release_y - 10, panel_height + 52),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    release_color,
                    2,
                    cv2.LINE_AA,
                )

        if phase_target_camera is not None and phase_target_camera[2] > 0:
            target_pixel = intrinsic @ phase_target_camera
            target_pixel = target_pixel[:2] / target_pixel[2]
            target_x, target_y = np.rint(target_pixel).astype(int)
            if 0 <= target_x < width and 0 <= target_y < height:
                target_color = (80, 255, 80)
                cv2.drawMarker(
                    frame,
                    (target_x, target_y),
                    target_color,
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=32,
                    thickness=2,
                )
                cv2.circle(
                    frame,
                    (target_x, target_y),
                    14,
                    target_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"IK TARGET: {phase_names.get(phase, 'unknown').upper()}",
                    (
                        min(target_x + 14, width - 180),
                        max(target_y - 12, panel_height + 70),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    target_color,
                    2,
                    cv2.LINE_AA,
                )

        annotated.append(frame)
        previous_phase = phase
    return annotated
