"""Load RoboMIND HDF5 trajectories and extract subsampled video frames."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Embodiments that store BGR images (per RoboMIND README)
_BGR_EMBODIMENTS = {"h5_franka_3rgb", "h5_franka_1rgb", "h5_ur_1rgb", "h5_franka_fr3_dual"}


@dataclass
class TrajectoryData:
    """Decoded trajectory from a single HDF5 file."""

    frames: List[np.ndarray]
    """RGB uint8 images, shape (H, W, 3) each."""

    language_instruction: str
    """Natural-language task description."""

    frame_indices: List[int]
    """Original HDF5 indices of the selected frames."""

    source_path: str = ""
    total_frames: int = 0
    camera: str = ""


@dataclass
class MultiCameraTrajectoryData:
    """Decoded trajectory with multiple camera views per time step."""

    frames: List[List[np.ndarray]]
    """Per-time-step list of RGB images, one per camera."""

    language_instruction: str
    frame_indices: List[int]
    cameras: List[str] = field(default_factory=list)
    source_path: str = ""
    total_frames: int = 0


def _decode_jpeg(buf: bytes) -> np.ndarray:
    """Decode JPEG bytes → RGB numpy array (H, W, 3)."""
    from PIL import Image

    return np.array(Image.open(io.BytesIO(buf)).convert("RGB"))


def load_trajectory(
    path: str | Path,
    camera: str = "camera_front",
    target_fps: float = 2.0,
    source_fps: float = 10.0,
    embodiment: Optional[str] = None,
) -> TrajectoryData:
    """Load an HDF5 trajectory and subsample frames.

    Parameters
    ----------
    path
        Path to a RoboMIND ``.hdf5`` file.
    camera
        Camera key under ``observations/rgb_images/``.
    target_fps
        Desired extraction rate (Hz).
    source_fps
        Native recording rate of the trajectory (Hz).
    embodiment
        If set (e.g. ``"h5_franka_3rgb"``), applies BGR→RGB correction.
    """
    import h5py

    path = Path(path)
    step = max(1, int(round(source_fps / target_fps)))

    with h5py.File(path, "r") as f:
        lang_raw = f["language_instruction"][()]
        if isinstance(lang_raw, bytes):
            lang_raw = lang_raw.decode("utf-8")
        language = str(lang_raw)

        rgb_group = f["observations"]["rgb_images"][camera]
        total = rgb_group.shape[0]
        indices = list(range(0, total, step))

        frames: List[np.ndarray] = []
        for idx in indices:
            raw = rgb_group[idx]
            if isinstance(raw, bytes):
                img = _decode_jpeg(raw)
            elif isinstance(raw, np.ndarray) and raw.ndim == 1:
                img = _decode_jpeg(raw.tobytes())
            elif isinstance(raw, np.ndarray) and raw.ndim == 3 and raw.dtype == np.uint8:
                img = raw
            else:
                img = _decode_jpeg(bytes(raw))

            if embodiment and embodiment in _BGR_EMBODIMENTS:
                img = img[:, :, ::-1].copy()

            frames.append(img)

    logger.info(
        "Loaded %d/%d frames from %s (camera=%s, step=%d)",
        len(frames), total, path.name, camera, step,
    )

    return TrajectoryData(
        frames=frames,
        language_instruction=language,
        frame_indices=indices,
        source_path=str(path),
        total_frames=total,
        camera=camera,
    )


def list_cameras(path: str | Path) -> List[str]:
    """Return all camera keys available in an HDF5 trajectory."""
    import h5py

    with h5py.File(Path(path), "r") as f:
        return sorted(f["observations"]["rgb_images"].keys())


def load_trajectory_multicam(
    path: str | Path,
    cameras: List[str] | None = None,
    target_fps: float = 2.0,
    source_fps: float = 10.0,
    embodiment: Optional[str] = None,
) -> MultiCameraTrajectoryData:
    """Load an HDF5 trajectory with all (or selected) camera views.

    Parameters
    ----------
    cameras
        Camera keys to load. If *None*, all available cameras are used.
    """
    import h5py

    path = Path(path)
    step = max(1, int(round(source_fps / target_fps)))

    with h5py.File(path, "r") as f:
        lang_raw = f["language_instruction"][()]
        if isinstance(lang_raw, bytes):
            lang_raw = lang_raw.decode("utf-8")
        language = str(lang_raw)

        rgb_images = f["observations"]["rgb_images"]
        if cameras is None:
            cameras = sorted(rgb_images.keys())

        first_cam = rgb_images[cameras[0]]
        total = first_cam.shape[0]
        indices = list(range(0, total, step))

        multi_frames: List[List[np.ndarray]] = []
        for idx in indices:
            views: List[np.ndarray] = []
            for cam in cameras:
                raw = rgb_images[cam][idx]
                if isinstance(raw, bytes):
                    img = _decode_jpeg(raw)
                elif isinstance(raw, np.ndarray) and raw.ndim == 1:
                    img = _decode_jpeg(raw.tobytes())
                elif isinstance(raw, np.ndarray) and raw.ndim == 3 and raw.dtype == np.uint8:
                    img = raw
                else:
                    img = _decode_jpeg(bytes(raw))

                if embodiment and embodiment in _BGR_EMBODIMENTS:
                    img = img[:, :, ::-1].copy()
                views.append(img)
            multi_frames.append(views)

    logger.info(
        "Loaded %d/%d frames from %s (cameras=%s, step=%d)",
        len(multi_frames), total, path.name, cameras, step,
    )

    return MultiCameraTrajectoryData(
        frames=multi_frames,
        language_instruction=language,
        frame_indices=indices,
        cameras=cameras,
        source_path=str(path),
        total_frames=total,
    )
