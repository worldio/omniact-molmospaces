import gymnasium.spaces as gyms
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from molmo_spaces.env.abstract_sensors import Sensor
from molmo_spaces.utils.pose import GL_FROM_CV


class CameraSensor(Sensor):
    """Sensor for RGB camera images from MuJoCo."""

    def __init__(
        self,
        camera_name: str = "camera",
        img_resolution: tuple[int, int] = (480, 480),
        uuid: str | None = None,
    ) -> None:
        self.camera_name = camera_name
        self.img_resolution = img_resolution

        if uuid is None:
            uuid = f"camera_{camera_name}"

        # Define observation space for RGB images
        width, height = img_resolution
        observation_space = gyms.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get camera image from environment rendering."""

        render_override = None
        policy = task._registered_policy
        if policy is not None and batch_index == env.current_batch_index:
            render_override = policy.get_render_override()
        if render_override is None:
            frame = env.render_rgb_frame(self.camera_name)
        else:
            frame = env.render_rgb_frame(
                self.camera_name,
                data=render_override.data,
                highlight_geom_ids=render_override.highlight_geom_ids,
                highlight_rgba=render_override.highlight_rgba,
            )

        if policy is not None:
            caption = policy.get_render_caption()
            if caption:
                frame = self._draw_caption(frame, caption)

        if frame is not None:
            return frame

        # Return black image if no rendering available
        width, height = self.img_resolution
        return np.zeros((height, width, 3), dtype=np.uint8)

    @staticmethod
    def _draw_caption(frame: np.ndarray, lines: tuple[str, ...]) -> np.ndarray:
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        font_size = max(12, image.width // 35)
        try:
            bold_font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
            regular_font = ImageFont.truetype("DejaVuSans.ttf", max(11, font_size - 3))
        except OSError:
            bold_font = regular_font = ImageFont.load_default()
        line_height = font_size + 5
        box_height = min(image.height, 10 + line_height * len(lines))
        draw.rectangle((0, 0, image.width, box_height), fill=(0, 0, 0))
        for index, line in enumerate(lines):
            font = bold_font if index == 0 else regular_font
            draw.text((12, 5 + index * line_height), line, fill=(255, 255, 255), font=font)
        return np.asarray(image)


class DepthSensor(Sensor):
    """Sensor for depth images from MuJoCo.

    Returns raw metric depth in meters as float32. Encoding to RGB for video storage
    happens at save time. See molmo_spaces.utils.depth_utils for encoding/decoding functions.
    """

    def __init__(
        self,
        camera_name: str = "camera",
        img_resolution: tuple[int, int] = (480, 480),
        uuid: str | None = None,
    ) -> None:
        self.camera_name = camera_name
        self.img_resolution = img_resolution

        if uuid is None:
            uuid = f"depth_{camera_name}"

        # Define observation space for raw depth (float32 in meters)
        width, height = img_resolution
        observation_space = gyms.Box(low=0.0, high=10.0, shape=(height, width), dtype=np.float32)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get depth image from environment rendering."""
        # Use camera-specific frame access for multi-camera support
        if hasattr(env, "render_depth_frame") and callable(env.render_depth_frame):
            frame = env.render_depth_frame(self.camera_name)
            if frame is not None:
                return frame

        # Fallback to default camera for backward compatibility
        if hasattr(env, "depth_frame") and env.depth_frame is not None:
            return env.depth_frame

        # Return zero depth if no rendering available
        width, height = self.img_resolution
        return np.zeros((height, width), dtype=np.float32)


class SegmentationSensor(Sensor):
    """Sensor for segmentation images from MuJoCo, outputs video-compatible arrays."""

    def __init__(
        self,
        camera_name: str = "camera",
        img_resolution: tuple[int, int] = (480, 480),
        uuid: str | None = None,
    ) -> None:
        self.camera_name = camera_name
        self.img_resolution = img_resolution

        if uuid is None:
            uuid = f"segmentation_{camera_name}"

        # Define observation space for uint8 images with channel dimension
        width, height = img_resolution
        observation_space = gyms.Box(low=0, high=255, shape=(height, width, 1), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get segmentation image from environment rendering."""
        # Use camera-specific frame access for multi-camera support
        if hasattr(env, "segmentation_frame") and callable(env.segmentation_frame):
            frame = env.segmentation_frame(self.camera_name)
            if frame is not None:
                return frame

        # Fallback to default camera for backward compatibility
        if hasattr(env, "segmentation_frame") and env.segmentation_frame is not None:
            return env.segmentation_frame

        # Return zero segmentation if no rendering available
        width, height = self.img_resolution
        return np.zeros((height, width, 1), dtype=np.uint8)


class CameraParameterSensor(Sensor):
    """Sensor for camera parameters (intrinsics and extrinsics)."""

    def __init__(
        self,
        camera_name: str,
        img_resolution: tuple[int, int],
        uuid: str | None = None,
    ) -> None:
        self.img_resolution = img_resolution
        self.camera_name = camera_name

        if uuid is None:
            uuid = f"camera_params_{camera_name}"

        observation_space = gyms.Dict(
            {
                "extrinsic_cv": gyms.Box(low=-np.inf, high=np.inf, shape=(3, 4), dtype=np.float32),
                "cam2world_gl": gyms.Box(low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float32),
                "intrinsic_cv": gyms.Box(low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float32),
            }
        )
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> dict:
        """Get camera parameters for a specific environment."""
        camera = env.camera_manager.registry[self.camera_name]
        pose = camera.get_pose()  # CV-convention cam-to-world: columns (right, -up, forward)
        extrinsic_cv = np.linalg.inv(pose)[:3, :]  # 3x4 matrix
        # cam2world_gl must actually be GL -- columns (right, up, -forward); GL_FROM_CV is the flip.
        cam2world_gl = pose @ GL_FROM_CV

        width, height = self.img_resolution
        fovy_degrees = camera.fov

        # Convert field of view to focal length
        focal_length = (height / 2.0) / np.tan(np.radians(fovy_degrees / 2.0))

        # Create intrinsic matrix (assuming square pixels and centered principal point)
        intrinsic_cv = np.array(
            [[focal_length, 0, width / 2.0], [0, focal_length, height / 2.0], [0, 0, 1]],
            dtype=np.float32,
        )

        # Ensure consistent structure and ordering
        data = {
            "cam2world_gl": cam2world_gl.tolist(),
            "extrinsic_cv": extrinsic_cv.tolist(),
            "intrinsic_cv": intrinsic_cv.tolist(),
        }
        return data
