#!/usr/bin/env python3
"""
Pure Python VIO-SLAM pipeline extracted from the verified notebook workflow.

This module is now the primary runtime for the VIO-SLAM project. It keeps the
same EuRoC `mav0` assumptions used in the notebook while exporting
self-awareness metrics and TUM trajectories for downstream reliability
prediction.
"""

from __future__ import annotations

import csv
import glob
import json
import logging
import os
import pickle
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from tqdm import tqdm

logger = logging.getLogger(__name__)


DEFAULT_CAMERA_MATRIX = np.array(
    [
        [458.654, 0.0, 367.215],
        [0.0, 457.296, 248.375],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass
class PairDiagnostics:
    """Frame-to-frame tracking diagnostics."""

    num_keypoints_prev: int = 0
    num_keypoints_curr: int = 0
    num_matches: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    mean_epipolar_error: float = 0.0
    tracking_state: int = 0
    lost_tracking: int = 1


@dataclass
class MetricsRecord:
    """Window-level metrics exported for self-awareness inference."""

    timestamp: float
    frame_id: int
    window_start_timestamp: float
    window_end_timestamp: float
    num_keypoints: int
    feature_count: int
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    feature_tracking_ratio: float
    tracked_map_points: int
    tracking_state: int
    lost_tracking: int
    mean_epipolar_error: float
    pose_optimization_residual: float
    imu_delta_translation: float
    trajectory_increment_norm: float
    processing_time_ms: float
    window_size: int


@dataclass
class OnlinePredictionRecord:
    """Streaming self-awareness prediction aligned to one VIO output row."""

    timestamp: float
    frame_id: int
    failure_probability: float
    confidence_score: float
    predicted_pose_error: float
    predicted_localization_reliability: float
    predicted_failure: int
    window_size: int
    frames_seen: int


class OnlineSelfAwareBridge:
    """Maintain a streaming self-aware predictor in a sidecar Python process."""

    def __init__(self,
                 python_executable: str,
                 inference_script: str,
                 config_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 dataset_stats_path: Optional[str] = None,
                 verbose: bool = False):
        self.verbose = verbose
        command = [
            python_executable,
            "-u",
            inference_script,
            "--stream",
        ]
        if config_path:
            command.extend(["--config", config_path])
        if checkpoint_path:
            command.extend(["--checkpoint", checkpoint_path])
        if dataset_stats_path:
            command.extend(["--dataset-stats", dataset_stats_path])

        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def predict(self, metrics_record: MetricsRecord) -> Optional[OnlinePredictionRecord]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("Online self-aware bridge failed to start correctly")

        self.process.stdin.write(json.dumps(asdict(metrics_record)) + "\n")
        self.process.stdin.flush()
        response_line = self.process.stdout.readline()
        if not response_line:
            stderr_output = ""
            if self.process.stderr is not None:
                stderr_output = self.process.stderr.read().strip()
            raise RuntimeError(f"Online self-aware predictor stopped unexpectedly. {stderr_output}")

        payload = json.loads(response_line)
        status = payload.get("status")
        if status == "warmup":
            if self.verbose:
                logger.info(
                    "Online predictor warming up: %s/%s rows",
                    payload.get("frames_seen", 0),
                    payload.get("required_window_size", 0),
                )
            return None
        if status != "prediction":
            raise RuntimeError(f"Unexpected online predictor response: {payload}")

        return OnlinePredictionRecord(
            timestamp=float(payload["timestamp"]),
            frame_id=int(payload["frame_id"]),
            failure_probability=float(payload["failure_probability"]),
            confidence_score=float(payload["confidence_score"]),
            predicted_pose_error=float(payload["predicted_pose_error"]),
            predicted_localization_reliability=float(payload["predicted_localization_reliability"]),
            predicted_failure=int(payload["predicted_failure"]),
            window_size=int(payload.get("window_size", 0)),
            frames_seen=int(payload.get("frames_seen", 0)),
        )

    def close(self):
        if self.process.poll() is not None:
            return

        try:
            if self.process.stdin is not None:
                self.process.stdin.write(json.dumps({"command": "close"}) + "\n")
                self.process.stdin.flush()
        except BrokenPipeError:
            pass
        finally:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


class EuRoCDatasetLoader:
    """EuRoC `mav0` loader matching the notebook layout."""

    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self.cam_dirs = {
            "cam0": os.path.join(self.root_dir, "cam0", "data"),
            "cam1": os.path.join(self.root_dir, "cam1", "data"),
        }
        self.cam_configs = {
            "cam0": os.path.join(self.root_dir, "cam0", "sensor.yaml"),
            "cam1": os.path.join(self.root_dir, "cam1", "sensor.yaml"),
        }
        self.imu_file = os.path.join(self.root_dir, "imu0", "data.csv")

    def validate(self) -> bool:
        return (
            os.path.isdir(self.cam_dirs["cam0"])
            and os.path.isfile(self.imu_file)
        )

    def load_images(self, camera: str = "cam0") -> Tuple[np.ndarray, List[str]]:
        cam_dir = self.cam_dirs.get(camera)
        if not cam_dir or not os.path.isdir(cam_dir):
            raise ValueError(f"Camera directory not found: {cam_dir}")

        image_paths = sorted(glob.glob(os.path.join(cam_dir, "*.png")))
        if not image_paths:
            raise ValueError(f"No PNG images found in {cam_dir}")

        timestamps = np.array(
            [int(os.path.splitext(os.path.basename(path))[0]) for path in image_paths],
            dtype=np.int64,
        )
        return timestamps, image_paths

    def load_imu(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not os.path.isfile(self.imu_file):
            raise FileNotFoundError(f"IMU file not found: {self.imu_file}")

        data = np.loadtxt(self.imu_file, delimiter=",", comments="#")
        timestamps = data[:, 0].astype(np.int64)
        gyro = data[:, 1:4]
        accel = data[:, 4:7]
        return timestamps, gyro, accel

    def get_camera_matrix(self, camera: str = "cam0") -> np.ndarray:
        sensor_path = self.cam_configs.get(camera)
        if not sensor_path or not os.path.isfile(sensor_path):
            return DEFAULT_CAMERA_MATRIX.copy()

        try:
            with open(sensor_path, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
            intrinsics = config.get("intrinsics", [])
            if len(intrinsics) >= 4:
                fx, fy, cx, cy = intrinsics[:4]
                return np.array(
                    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to read camera intrinsics from %s: %s", sensor_path, exc)
        return DEFAULT_CAMERA_MATRIX.copy()


class IMUPreintegrator:
    """Minimal IMU preintegrator copied from the validated notebook."""

    def __init__(self, gravity: Optional[np.ndarray] = None):
        self.gravity = gravity if gravity is not None else np.array([0.0, 0.0, -9.81])
        self.reset()

    def reset(self):
        self.delta_t = 0.0
        self.delta_R = np.eye(3)
        self.delta_v = np.zeros(3)
        self.delta_p = np.zeros(3)

    def integrate(
        self,
        dt_arr: np.ndarray,
        omega_arr: np.ndarray,
        acc_arr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.reset()
        for dt, omega, accel in zip(dt_arr, omega_arr, acc_arr):
            dR = Rotation.from_rotvec(omega * dt).as_matrix()
            self.delta_R = self.delta_R @ dR
            accel_world = self.delta_R @ accel + self.gravity
            self.delta_v += accel_world * dt
            self.delta_p += self.delta_v * dt + 0.5 * accel_world * dt**2
            self.delta_t += dt

        return self.delta_p.copy(), self.delta_v.copy(), self.delta_R.copy()


def load_config(config_path: Optional[str]) -> Dict:
    """Load YAML config or return notebook-like defaults."""
    config = {
        "dataset": {
            "camera": "cam0",
            "downsample_factor": 10,
            "default_data_path": "data/mav0",
        },
        "slam": {
            "window_size": 3,
            "orb_features": 1000,
            "max_matches": 500,
            "ransac_threshold": 1.0,
            "gravity": [0.0, 0.0, -9.81],
        },
        "visualization": {
            "show_trajectory": False,
            "save_plot": False,
        },
    }

    if config_path and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        _deep_update(config, loaded)
    return config


def _deep_update(base: Dict, updates: Dict):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def resolve_data_path(explicit_path: Optional[str] = None) -> Optional[Path]:
    """Resolve a EuRoC mav0 directory using notebook-style defaults."""
    root = Path(__file__).resolve().parent
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    env_path = os.environ.get("VIO_SLAM_DATA_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            root / "data" / "mav0",
            Path.cwd() / "data" / "mav0",
            root / "mav0",
            Path.cwd() / "mav0",
        ]
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "cam0" / "data").is_dir() and (candidate / "imu0" / "data.csv").is_file():
            return candidate
    return None


def _safe_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return np.zeros_like(vector)
    return vector / norm


def _compute_mean_epipolar_error(E: Optional[np.ndarray], pts1: np.ndarray, pts2: np.ndarray) -> float:
    if E is None or len(pts1) == 0 or len(pts2) == 0:
        return 0.0

    ones = np.ones((len(pts1), 1))
    pts1_h = np.hstack([pts1, ones])
    pts2_h = np.hstack([pts2, ones])
    residuals = np.abs(np.sum((pts2_h @ E) * pts1_h, axis=1))
    return float(np.mean(residuals)) if len(residuals) else 0.0


def _yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    quat_xyzw = Rotation.from_euler("z", yaw).as_quat()
    return tuple(float(v) for v in quat_xyzw)


def compute_visual_measurements(
    image_paths: List[str],
    window_size: int,
    camera_matrix: np.ndarray,
    orb_features: int,
    max_matches: int,
    ransac_threshold: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[PairDiagnostics]]:
    """Compute pairwise visual measurements for a sliding window."""
    orb = cv2.ORB_create(nfeatures=orb_features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    rotations: List[np.ndarray] = []
    translations: List[np.ndarray] = []
    diagnostics: List[PairDiagnostics] = []

    for idx in range(window_size):
        img1 = cv2.imread(image_paths[idx], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_paths[idx + 1], cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            raise FileNotFoundError(f"Failed to read image pair: {image_paths[idx]}, {image_paths[idx + 1]}")

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        diag = PairDiagnostics(
            num_keypoints_prev=len(kp1),
            num_keypoints_curr=len(kp2),
        )

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            rotations.append(np.eye(3))
            translations.append(np.zeros(3))
            diagnostics.append(diag)
            continue

        matches = sorted(matcher.match(des1, des2), key=lambda match: match.distance)[:max_matches]
        diag.num_matches = len(matches)
        if len(matches) < 8:
            rotations.append(np.eye(3))
            translations.append(np.zeros(3))
            diagnostics.append(diag)
            continue

        pts1 = np.array([kp1[match.queryIdx].pt for match in matches], dtype=np.float32)
        pts2 = np.array([kp2[match.trainIdx].pt for match in matches], dtype=np.float32)

        E, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=ransac_threshold,
        )

        if E is None or mask is None:
            rotations.append(np.eye(3))
            translations.append(np.zeros(3))
            diagnostics.append(diag)
            continue

        inlier_mask = mask.ravel().astype(bool)
        inlier_pts1 = pts1[inlier_mask]
        inlier_pts2 = pts2[inlier_mask]
        diag.num_inliers = int(np.count_nonzero(inlier_mask))
        diag.inlier_ratio = float(diag.num_inliers / diag.num_matches) if diag.num_matches else 0.0
        diag.mean_epipolar_error = _compute_mean_epipolar_error(E, inlier_pts1, inlier_pts2)

        if diag.num_inliers < 5:
            rotations.append(np.eye(3))
            translations.append(np.zeros(3))
            diagnostics.append(diag)
            continue

        _, rotation, translation, pose_mask = cv2.recoverPose(E, pts1, pts2, camera_matrix)
        if pose_mask is not None:
            pose_inliers = int(np.count_nonzero(pose_mask))
            diag.num_inliers = min(diag.num_inliers, pose_inliers) if diag.num_inliers else pose_inliers
            diag.inlier_ratio = float(diag.num_inliers / diag.num_matches) if diag.num_matches else 0.0

        diag.tracking_state = 1
        diag.lost_tracking = 0
        rotations.append(rotation)
        translations.append(translation.flatten())
        diagnostics.append(diag)

    return rotations, translations, diagnostics


def slide_window_vio(
    ts_img: np.ndarray,
    ts_imu: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
    image_paths: List[str],
    camera_matrix: np.ndarray,
    window_size: int,
    orb_features: int,
    max_matches: int,
    ransac_threshold: float,
    gravity: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Run one sliding-window optimization step."""
    preintegrator = IMUPreintegrator(gravity=gravity)
    delta_positions: List[np.ndarray] = []
    delta_rotations: List[np.ndarray] = []

    for idx in range(1, window_size + 1):
        t0, t1 = ts_img[idx - 1], ts_img[idx]
        mask = (ts_imu >= t0) & (ts_imu < t1)
        ts_sec = ts_imu[mask].astype(np.float64) * 1e-9
        omega_arr = gyro[mask]
        acc_arr = accel[mask]

        if len(ts_sec) < 2:
            delta_positions.append(np.zeros(3))
            delta_rotations.append(np.eye(3))
            continue

        dt_arr = np.diff(ts_sec)
        dp, _dv, dR = preintegrator.integrate(dt_arr, omega_arr[:-1], acc_arr[:-1])
        delta_positions.append(dp)
        delta_rotations.append(dR)

    rotations_vis, translations_vis, pair_diagnostics = compute_visual_measurements(
        image_paths=image_paths,
        window_size=window_size,
        camera_matrix=camera_matrix,
        orb_features=orb_features,
        max_matches=max_matches,
        ransac_threshold=ransac_threshold,
    )

    translation_dirs = [_safe_normalize(t) for t in translations_vis]
    p_init = np.zeros((window_size + 1, 3))
    theta_init = np.zeros(window_size + 1)

    for idx in range(1, window_size + 1):
        p_init[idx] = p_init[idx - 1] + delta_positions[idx - 1]
        yaw = np.arctan2(delta_rotations[idx - 1][1, 0], delta_rotations[idx - 1][0, 0])
        theta_init[idx] = theta_init[idx - 1] + yaw

    x0 = np.hstack([p_init.flatten(), theta_init])

    def residual(x: np.ndarray) -> np.ndarray:
        positions = x[: 3 * (window_size + 1)].reshape((window_size + 1, 3))
        theta = x[3 * (window_size + 1) :]
        residuals = []

        for idx in range(1, window_size + 1):
            residuals.extend((positions[idx] - positions[idx - 1] - delta_positions[idx - 1]).tolist())
            imu_yaw = np.arctan2(delta_rotations[idx - 1][1, 0], delta_rotations[idx - 1][0, 0])
            residuals.append((theta[idx] - theta[idx - 1]) - imu_yaw)

            if pair_diagnostics[idx - 1].tracking_state:
                delta_est = positions[idx] - positions[idx - 1]
                dir_est = _safe_normalize(delta_est)
                residuals.extend((dir_est - translation_dirs[idx - 1]).tolist())

        return np.array(residuals, dtype=np.float64)

    solution = least_squares(residual, x0, verbose=0)
    positions_opt = solution.x[: 3 * (window_size + 1)].reshape((window_size + 1, 3))
    theta_opt = solution.x[3 * (window_size + 1) :]
    residual_value = float(np.mean(np.abs(solution.fun))) if len(solution.fun) else 0.0

    diagnostics = {
        "pairs": pair_diagnostics,
        "final_pair": pair_diagnostics[-1] if pair_diagnostics else PairDiagnostics(),
        "pose_optimization_residual": residual_value,
        "imu_delta_translation": float(np.linalg.norm(delta_positions[-1])) if delta_positions else 0.0,
        "trajectory_increment_norm": float(np.linalg.norm(positions_opt[-1])),
        "yaw_increment": float(theta_opt[-1]),
    }
    return p_init, positions_opt, theta_opt, diagnostics


class NotebookDerivedVIOPipeline:
    """Primary VIO-SLAM pipeline extracted from the validated notebook."""

    def __init__(self, config: Dict, online_predictor: Optional[OnlineSelfAwareBridge] = None):
        self.config = config
        self.online_predictor = online_predictor
        self.metrics_records: List[MetricsRecord] = []
        self.online_prediction_records: List[OnlinePredictionRecord] = []
        self.trajectory_positions = np.zeros((0, 3), dtype=np.float64)
        self.trajectory_yaws = np.zeros((0,), dtype=np.float64)
        self.trajectory_timestamps = np.zeros((0,), dtype=np.float64)

    def run(self, data_path: str) -> np.ndarray:
        loader = EuRoCDatasetLoader(data_path)
        if not loader.validate():
            raise RuntimeError(f"Invalid EuRoC mav0 directory: {data_path}")

        camera = self.config["dataset"]["camera"]
        downsample = int(self.config["dataset"]["downsample_factor"])
        window_size = int(self.config["slam"]["window_size"])
        orb_features = int(self.config["slam"]["orb_features"])
        max_matches = int(self.config["slam"]["max_matches"])
        ransac_threshold = float(self.config["slam"]["ransac_threshold"])
        gravity = np.array(self.config["slam"]["gravity"], dtype=np.float64)

        ts_img, image_paths = loader.load_images(camera)
        ts_imu, gyro, accel = loader.load_imu()
        camera_matrix = loader.get_camera_matrix(camera)

        num_windows = len(ts_img) - window_size
        starts = list(range(0, max(num_windows, 0), downsample))
        logger.info("Running notebook-derived VIO pipeline on %s windows", len(starts))

        trajectory_positions: List[np.ndarray] = []
        trajectory_yaws: List[float] = []
        trajectory_timestamps: List[float] = []
        metrics_records: List[MetricsRecord] = []
        online_prediction_records: List[OnlinePredictionRecord] = []
        cumulative_position = np.zeros(3, dtype=np.float64)
        cumulative_yaw = 0.0

        try:
            for start_idx in tqdm(starts, desc="VIO Sliding-Window", ncols=80):
                step_start = time.time()
                t_win = ts_img[start_idx : start_idx + window_size + 1]
                img_win = image_paths[start_idx : start_idx + window_size + 1]

                _, positions_opt, theta_opt, diagnostics = slide_window_vio(
                    ts_img=t_win,
                    ts_imu=ts_imu,
                    gyro=gyro,
                    accel=accel,
                    image_paths=img_win,
                    camera_matrix=camera_matrix,
                    window_size=window_size,
                    orb_features=orb_features,
                    max_matches=max_matches,
                    ransac_threshold=ransac_threshold,
                    gravity=gravity,
                )

                increment = positions_opt[-1]
                cumulative_position = cumulative_position + increment
                cumulative_yaw = cumulative_yaw + float(theta_opt[-1])

                trajectory_positions.append(cumulative_position.copy())
                trajectory_yaws.append(cumulative_yaw)
                trajectory_timestamps.append(float(t_win[-1]) / 1e9)

                final_pair = diagnostics["final_pair"]
                metrics_record = MetricsRecord(
                    timestamp=float(t_win[-1]) / 1e9,
                    frame_id=int(start_idx + window_size),
                    window_start_timestamp=float(t_win[0]) / 1e9,
                    window_end_timestamp=float(t_win[-1]) / 1e9,
                    num_keypoints=int(final_pair.num_keypoints_curr),
                    feature_count=int(final_pair.num_keypoints_curr),
                    num_matches=int(final_pair.num_matches),
                    num_inliers=int(final_pair.num_inliers),
                    inlier_ratio=float(final_pair.inlier_ratio),
                    feature_tracking_ratio=float(final_pair.inlier_ratio),
                    tracked_map_points=int(final_pair.num_inliers),
                    tracking_state=int(final_pair.tracking_state),
                    lost_tracking=int(final_pair.lost_tracking),
                    mean_epipolar_error=float(final_pair.mean_epipolar_error),
                    pose_optimization_residual=float(diagnostics["pose_optimization_residual"]),
                    imu_delta_translation=float(diagnostics["imu_delta_translation"]),
                    trajectory_increment_norm=float(diagnostics["trajectory_increment_norm"]),
                    processing_time_ms=(time.time() - step_start) * 1000.0,
                    window_size=window_size,
                )
                metrics_records.append(metrics_record)

                if self.online_predictor is not None:
                    prediction = self.online_predictor.predict(metrics_record)
                    if prediction is not None:
                        online_prediction_records.append(prediction)
                        logger.info(
                            "Online self-aware prediction frame=%s failure_probability=%.3f confidence=%.3f",
                            prediction.frame_id,
                            prediction.failure_probability,
                            prediction.confidence_score,
                        )
        finally:
            if self.online_predictor is not None:
                self.online_predictor.close()

        self.metrics_records = metrics_records
        self.online_prediction_records = online_prediction_records
        self.trajectory_positions = np.asarray(trajectory_positions, dtype=np.float64)
        self.trajectory_yaws = np.asarray(trajectory_yaws, dtype=np.float64)
        self.trajectory_timestamps = np.asarray(trajectory_timestamps, dtype=np.float64)
        return self.trajectory_positions

    def save_metrics_csv(self, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(self.metrics_records[0]).keys()) if self.metrics_records else list(MetricsRecord.__annotations__.keys()))
            writer.writeheader()
            for record in self.metrics_records:
                writer.writerow(asdict(record))
        logger.info("Saved metrics CSV to %s", output_path)

    def save_tum_trajectory(self, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            for timestamp, position, yaw in zip(self.trajectory_timestamps, self.trajectory_positions, self.trajectory_yaws):
                qx, qy, qz, qw = _yaw_to_quaternion(float(yaw))
                handle.write(
                    f"{timestamp:.9f} {position[0]:.6f} {position[1]:.6f} {position[2]:.6f} "
                    f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
                )
        logger.info("Saved TUM trajectory to %s", output_path)

    def save_online_predictions_csv(self, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(asdict(self.online_prediction_records[0]).keys())
                if self.online_prediction_records
                else list(OnlinePredictionRecord.__annotations__.keys()),
            )
            writer.writeheader()
            for record in self.online_prediction_records:
                writer.writerow(asdict(record))
        logger.info("Saved online self-aware predictions to %s", output_path)

    def save_trajectory_text(self, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        header = "timestamp x y z yaw"
        data = np.column_stack([self.trajectory_timestamps, self.trajectory_positions, self.trajectory_yaws]) if len(self.trajectory_positions) else np.empty((0, 5))
        np.savetxt(output_path, data, fmt="%.6f", header=header, comments="# ")
        logger.info("Saved trajectory text to %s", output_path)

    def save_trajectory_pickle(self, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as handle:
            pickle.dump(
                {
                    "timestamps": self.trajectory_timestamps,
                    "positions": self.trajectory_positions,
                    "yaws": self.trajectory_yaws,
                },
                handle,
            )
        logger.info("Saved trajectory pickle to %s", output_path)

    def save_plot(self, output_path: str, show_plot: bool = False):
        import matplotlib.pyplot as plt

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 6))
        if len(self.trajectory_positions):
            plt.plot(self.trajectory_positions[:, 0], self.trajectory_positions[:, 1], "b.-", label="VIO-Optimized")
        plt.title("Global Trajectory (XY Plane)")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        if show_plot:
            plt.show()
        plt.close()
        logger.info("Saved trajectory plot to %s", output_path)

    def get_statistics(self) -> Dict[str, float]:
        total_distance = 0.0
        if len(self.trajectory_positions) > 1:
            total_distance = float(np.sum(np.linalg.norm(np.diff(self.trajectory_positions, axis=0), axis=1)))
        tracking_ratio = float(np.mean([record.tracking_state for record in self.metrics_records])) if self.metrics_records else 0.0
        mean_inlier_ratio = float(np.mean([record.inlier_ratio for record in self.metrics_records])) if self.metrics_records else 0.0
        return {
            "num_poses": int(len(self.trajectory_positions)),
            "trajectory_length_m": total_distance,
            "tracking_success_ratio": tracking_ratio,
            "mean_inlier_ratio": mean_inlier_ratio,
            "online_predictions": int(len(self.online_prediction_records)),
            "online_failure_probability_mean": float(np.mean([record.failure_probability for record in self.online_prediction_records])) if self.online_prediction_records else 0.0,
        }
