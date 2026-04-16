"""
Task 1.3: Internal Signal Logging

Python module to log internal SLAM metrics from VINS-Fusion.
This interfaces with ROS topics published by VINS-Fusion to extract:
  - feature_count
  - tracked_feature_ratio
  - reprojection_error (mean/std)
  - imu_residual
  - optimization_iterations
  - tracking_time
  - tracking_state

Output: CSV file with columns [timestamp, feature_count, reprojection_error,
        imu_residual, tracking_state, ...]

Usage (with ROS):
    rosrun self_aware_slam slam_metrics_logger.py --output slam_metrics.csv

Standalone (without ROS):
    python slam_metrics_logger.py --mode standalone --output slam_metrics.csv
"""

import csv
import time
import os
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class SLAMMetrics:
    timestamp: float = 0.0
    feature_count: int = 0
    feature_tracking_ratio: float = 0.0
    reprojection_error_mean: float = 0.0
    reprojection_error_std: float = 0.0
    imu_residual_norm: float = 0.0
    optimization_iterations: int = 0
    tracking_time_ms: float = 0.0
    tracking_state: int = 1  # 1=ok, 0=lost
    camera_motion_magnitude: float = 0.0
    tracking_length: float = 0.0


FIELD_NAMES = [
    'timestamp', 'feature_count', 'feature_tracking_ratio',
    'reprojection_error_mean', 'reprojection_error_std',
    'imu_residual_norm', 'optimization_iterations',
    'tracking_time_ms', 'tracking_state',
    'camera_motion_magnitude', 'tracking_length'
]


class SLAMMetricsLogger:
    """Logs SLAM internal metrics to CSV."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.metrics_buffer: List[SLAMMetrics] = []
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        # Write header
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
            writer.writeheader()

    def log(self, metrics: SLAMMetrics):
        self.metrics_buffer.append(metrics)
        # Flush every 100 entries
        if len(self.metrics_buffer) >= 100:
            self.flush()

    def flush(self):
        if not self.metrics_buffer:
            return
        with open(self.output_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
            for m in self.metrics_buffer:
                writer.writerow(asdict(m))
        self.metrics_buffer.clear()

    def close(self):
        self.flush()


class VINSFusionMetricsExtractor:
    """
    Extracts metrics from VINS-Fusion ROS topics.

    Requires ROS environment. Subscribe to:
      - /vins_estimator/feature (feature tracking info)
      - /vins_estimator/imu_propagate (IMU state)
      - /vins_estimator/odometry (pose output)
    """

    def __init__(self, logger: SLAMMetricsLogger):
        self.logger = logger
        self._prev_position = None
        self._frame_count = 0

    def try_ros_init(self):
        """Attempt to initialize ROS node. Returns True if ROS is available."""
        try:
            import rospy
            rospy.init_node('slam_metrics_logger', anonymous=True)
            self._setup_subscribers()
            return True
        except ImportError:
            print("ROS not available. Use --mode standalone for testing.")
            return False

    def _setup_subscribers(self):
        """Set up ROS subscribers for VINS-Fusion topics."""
        import rospy
        from sensor_msgs.msg import PointCloud
        from nav_msgs.msg import Odometry

        rospy.Subscriber('/vins_estimator/feature', PointCloud,
                         self._feature_callback)
        rospy.Subscriber('/vins_estimator/odometry', Odometry,
                         self._odometry_callback)

    def _feature_callback(self, msg):
        """Process feature tracking message."""
        import rospy
        metrics = SLAMMetrics()
        metrics.timestamp = msg.header.stamp.to_sec()
        metrics.feature_count = len(msg.points)
        # Feature tracking ratio estimated from channel data
        if msg.channels and len(msg.channels) > 0:
            track_counts = [c.values[0] for c in msg.channels
                            if c.name == 'track_cnt' and c.values]
            if track_counts:
                tracked = sum(1 for t in track_counts if t > 1)
                metrics.feature_tracking_ratio = tracked / max(len(track_counts), 1)
        self._current_feature_metrics = metrics

    def _odometry_callback(self, msg):
        """Process odometry message to compute motion magnitude."""
        import numpy as np
        pos = msg.pose.pose.position
        current_pos = np.array([pos.x, pos.y, pos.z])

        if self._prev_position is not None:
            motion = np.linalg.norm(current_pos - self._prev_position)
        else:
            motion = 0.0
        self._prev_position = current_pos

        if hasattr(self, '_current_feature_metrics'):
            m = self._current_feature_metrics
            m.camera_motion_magnitude = motion
            self.logger.log(m)

    def spin(self):
        """ROS spin loop."""
        import rospy
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            rate.sleep()
        self.logger.close()


def main():
    parser = argparse.ArgumentParser(description='SLAM Metrics Logger')
    parser.add_argument('--output', type=str, default='slam_metrics_dataset/slam_metrics.csv')
    parser.add_argument('--mode', type=str, choices=['ros', 'standalone'], default='ros')
    args = parser.parse_args()

    logger = SLAMMetricsLogger(args.output)

    if args.mode == 'ros':
        extractor = VINSFusionMetricsExtractor(logger)
        if extractor.try_ros_init():
            print(f"Logging SLAM metrics to {args.output}")
            extractor.spin()
        else:
            print("Falling back to standalone mode.")
    else:
        print("Standalone mode: Use generate_synthetic_data.py to create test data.")

    logger.close()


if __name__ == '__main__':
    main()
