#!/bin/bash
#
# Run VINS-Fusion on EuRoC bag files and capture output.
#
# Prerequisites:
#   - ROS (melodic/noetic) or ROS2 (humble) installed
#   - VINS-Fusion built: https://github.com/HKUST-Aerial-Robotics/VINS-Fusion
#   - EuRoC bags downloaded (see download instructions below)
#
# Usage:
#   ./scripts/run_vins_euroc.sh /path/to/euroc_bags /path/to/vins_output
#
#   Example:
#   ./scripts/run_vins_euroc.sh ~/data/euroc ~/data/vins_output
#
# EuRoC bag download:
#   wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.bag
#   (repeat for MH_02_easy through MH_05_difficult)

set -e

EUROC_BAG_DIR="${1:?Usage: $0 <euroc_bag_dir> <output_dir>}"
OUTPUT_DIR="${2:?Usage: $0 <euroc_bag_dir> <output_dir>}"

# VINS-Fusion config file for EuRoC (adjust path to your VINS-Fusion install)
VINS_CONFIG="${VINS_CONFIG:-$(rospack find vins 2>/dev/null || echo '/opt/ros/vins_ws/src/VINS-Fusion')/config/euroc/euroc_stereo_imu_config.yaml}"

# Sequences to process
SEQUENCES=(
    "MH_01_easy"
    "MH_02_easy"
    "MH_03_medium"
    "MH_04_difficult"
    "MH_05_difficult"
)

echo "=================================================="
echo "VINS-Fusion EuRoC Batch Runner"
echo "=================================================="
echo "Bag dir:    $EUROC_BAG_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Config:     $VINS_CONFIG"
echo ""

if [ ! -f "$VINS_CONFIG" ]; then
    echo "WARNING: VINS config not found at $VINS_CONFIG"
    echo "Set VINS_CONFIG env var to your euroc config yaml."
    echo ""
fi

for SEQ in "${SEQUENCES[@]}"; do
    BAG_FILE="$EUROC_BAG_DIR/${SEQ}.bag"
    SEQ_OUTPUT="$OUTPUT_DIR/$SEQ"

    if [ ! -f "$BAG_FILE" ]; then
        echo "[$SEQ] Bag not found: $BAG_FILE - skipping"
        continue
    fi

    echo ""
    echo "=================================================="
    echo "Processing: $SEQ"
    echo "=================================================="

    mkdir -p "$SEQ_OUTPUT"

    # Launch VINS-Fusion in background
    echo "Starting VINS-Fusion..."

    # For ROS1:
    if command -v roslaunch &> /dev/null; then
        # Start roscore if not running
        if ! pgrep -x roscore > /dev/null; then
            roscore &
            sleep 2
        fi

        roslaunch vins vins_rviz.launch \
            config_path:="$VINS_CONFIG" \
            2>&1 | tee "$SEQ_OUTPUT/vins_log.txt" &
        VINS_PID=$!
        sleep 5

        # Play bag
        echo "Playing bag: $BAG_FILE"
        rosbag play "$BAG_FILE" --clock -r 0.5
        sleep 3

        # Kill VINS
        kill $VINS_PID 2>/dev/null || true
        wait $VINS_PID 2>/dev/null || true

    # For ROS2:
    elif command -v ros2 &> /dev/null; then
        ros2 launch vins vins_launch.py \
            config_path:="$VINS_CONFIG" \
            2>&1 | tee "$SEQ_OUTPUT/vins_log.txt" &
        VINS_PID=$!
        sleep 5

        echo "Playing bag: $BAG_FILE"
        ros2 bag play "$BAG_FILE" --clock -r 0.5
        sleep 3

        kill $VINS_PID 2>/dev/null || true
        wait $VINS_PID 2>/dev/null || true
    else
        echo "ERROR: Neither ROS1 nor ROS2 found. Install ROS first."
        exit 1
    fi

    # Copy VINS output trajectory
    # VINS-Fusion writes to ~/.ros/ or current directory
    for CANDIDATE in \
        "$HOME/.ros/vins_result_no_loop.csv" \
        "$HOME/.ros/vins_result_loop.csv" \
        "./vins_result_no_loop.csv" \
        "./vins_result_loop.csv"; do
        if [ -f "$CANDIDATE" ]; then
            cp "$CANDIDATE" "$SEQ_OUTPUT/"
            echo "Copied: $CANDIDATE -> $SEQ_OUTPUT/"
        fi
    done

    echo "[$SEQ] Done."
done

echo ""
echo "=================================================="
echo "All sequences processed."
echo "Output in: $OUTPUT_DIR"
echo ""
echo "Next step: process results into dataset format:"
echo "  python scripts/process_euroc_results.py \\"
echo "    --euroc-dir /path/to/euroc_extracted \\"
echo "    --vins-dir $OUTPUT_DIR \\"
echo "    --output-dir slam_metrics_dataset"
echo "=================================================="
