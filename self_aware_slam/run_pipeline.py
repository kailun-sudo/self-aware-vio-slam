"""
Self-Aware SLAM: Complete Pipeline Runner

Runs the full pipeline:
  1. Generate synthetic data (or use real SLAM data)
  2. Build reliability dataset (Milestone 2)
  3. Train and evaluate all three models (Milestone 3)
  4. Compare results

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-datagen  # if data already exists
    python run_pipeline.py --model transformer  # train single model
"""

import argparse
import os
import sys
import json

from src.utils.config_loader import load_config


def run_data_generation(config):
    print("\n" + "=" * 60)
    print("Step 1: Generating Synthetic SLAM Data")
    print("=" * 60)
    from scripts.generate_synthetic_data import main as gen_main
    gen_main()


def run_real_data_processing(config):
    """Process real EuRoC + VINS-Fusion data into dataset format."""
    print("\n" + "=" * 60)
    print("Step 1: Processing Real EuRoC + VINS-Fusion Data")
    print("=" * 60)
    from src.euroc.process_sequence import process_sequence, auto_detect_sequence_name

    euroc_dir = config.get('euroc', {}).get('dataset_dir', '')
    vins_dir = config.get('euroc', {}).get('vins_output_dir', '')
    output_dir = config['paths']['slam_metrics_dir']

    if not euroc_dir or not vins_dir:
        print("ERROR: Set euroc.dataset_dir and euroc.vins_output_dir in config.yaml")
        print("  Or pass --euroc-dir and --vins-dir on the command line.")
        sys.exit(1)

    sequences = config['dataset']['euroc_sequences']

    # Try to find matching directories
    for seq in sequences:
        # Try common EuRoC directory name patterns
        euroc_candidates = [
            os.path.join(euroc_dir, seq),
            os.path.join(euroc_dir, f"{seq}_easy"),
            os.path.join(euroc_dir, f"{seq}_medium"),
            os.path.join(euroc_dir, f"{seq}_difficult"),
        ]
        euroc_path = None
        for c in euroc_candidates:
            if os.path.isdir(c):
                euroc_path = c
                break

        vins_candidates = [
            os.path.join(vins_dir, seq),
            os.path.join(vins_dir, f"{seq}_easy"),
            os.path.join(vins_dir, f"{seq}_medium"),
            os.path.join(vins_dir, f"{seq}_difficult"),
        ]
        vins_path = None
        for c in vins_candidates:
            if os.path.isdir(c):
                vins_path = c
                break

        if not euroc_path:
            print(f"  Warning: EuRoC dir not found for {seq}, skipping")
            continue
        if not vins_path:
            print(f"  Warning: VINS output not found for {seq}, skipping")
            continue

        # Find trajectory file
        from scripts.process_euroc_results import find_vins_trajectory, find_vins_log
        try:
            traj_file = find_vins_trajectory(vins_path)
            log_file = find_vins_log(vins_path)
            process_sequence(
                euroc_seq_dir=euroc_path,
                vins_trajectory_path=traj_file,
                sequence_name=seq,
                output_base_dir=output_dir,
                vins_log_path=log_file,
            )
        except Exception as e:
            print(f"  ERROR processing {seq}: {e}")


def run_dataset_building(config):
    print("\n" + "=" * 60)
    print("Step 2: Building Reliability Dataset (Milestone 2)")
    print("=" * 60)
    from src.data.dataset_builder import build_dataset, save_dataset
    dataset = build_dataset(config)
    output_path = config['paths'].get(
        'train_dataset_path',
        os.path.join(config['paths']['results_dir'], 'train_dataset_v2.pkl'),
    )
    save_dataset(dataset, output_path)
    return dataset


def run_training(config, model_types=None):
    print("\n" + "=" * 60)
    print("Step 3: Training Failure Prediction Models (Milestone 3)")
    print("=" * 60)
    from src.models.train import train

    if model_types is None:
        model_types = ['mlp', 'lstm', 'transformer']

    all_results = {}
    for model_type in model_types:
        print(f"\n{'─' * 40}")
        print(f"Training {model_type.upper()} model...")
        print(f"{'─' * 40}")
        cfg = config.copy()
        cfg['model'] = config['model'].copy()
        cfg['model']['type'] = model_type
        _, results = train(cfg)
        all_results[model_type] = results

    return all_results


def compare_models(all_results):
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(f"\n{'Model':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
          f"{'F1':>10} {'AUC':>10} {'MAE':>10}")
    print("─" * 75)
    for name, res in all_results.items():
        m = res['test_metrics']
        print(f"{name.upper():<15} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>10.4f} {m['f1']:>10.4f} {m['roc_auc']:>10.4f} "
              f"{res['test_mae']:>10.4f}")

    # Find best model
    best = max(all_results.items(), key=lambda x: x[1]['test_metrics']['f1'])
    print(f"\nBest model by F1: {best[0].upper()} (F1={best[1]['test_metrics']['f1']:.4f})")


def main():
    parser = argparse.ArgumentParser(description='Self-Aware SLAM Pipeline')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--skip-datagen', action='store_true',
                        help='Skip synthetic data generation')
    parser.add_argument('--skip-dataset', action='store_true',
                        help='Skip dataset building (use existing)')
    parser.add_argument('--model', type=str, choices=['mlp', 'lstm', 'transformer'],
                        default=None, help='Train only one model type')
    parser.add_argument('--real-data', action='store_true',
                        help='Use real EuRoC + VINS-Fusion data instead of synthetic')
    parser.add_argument('--euroc-dir', type=str, default=None,
                        help='Override euroc.dataset_dir from config')
    parser.add_argument('--vins-dir', type=str, default=None,
                        help='Override euroc.vins_output_dir from config')
    args = parser.parse_args()

    config = load_config(args.config)

    # Override euroc paths from CLI if provided
    if args.euroc_dir:
        config.setdefault('euroc', {})['dataset_dir'] = args.euroc_dir
    if args.vins_dir:
        config.setdefault('euroc', {})['vins_output_dir'] = args.vins_dir

    # Step 1: Data generation
    if not args.skip_datagen:
        if args.real_data:
            run_real_data_processing(config)
        else:
            run_data_generation(config)

    # Step 2: Dataset building
    if not args.skip_dataset:
        run_dataset_building(config)

    # Step 3: Training
    model_types = [args.model] if args.model else None
    all_results = run_training(config, model_types)

    # Step 4: Comparison
    if len(all_results) > 1:
        compare_models(all_results)

    print("\nPipeline complete!")


if __name__ == '__main__':
    main()
