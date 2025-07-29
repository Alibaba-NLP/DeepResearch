#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning Chain Reconstruction Script

This script reconstructs verbose expert LRM trajectories into concise and effective reasoning chains,
removing redundant information while preserving key reasoning steps, suitable for model training.

Usage:
    python scripts/reconstruct_reasoning.py --config configs/default_config.yaml --input data/trajectories.json --output data/reconstructed_trajectories.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from websailor.utils.config import Config
from websailor.utils.logger import get_logger

logger = get_logger(__name__)


def load_trajectories(file_path: str) -> List[Dict[str, Any]]:
    """Load trajectory data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_quality_metrics(original_trajectory: Dict[str, Any], 
                            reconstructed_trajectory: Dict[str, Any]) -> Dict[str, float]:
    """Calculate reconstruction quality metrics"""
    original_steps = len(original_trajectory.get("trajectory", []))
    reconstructed_steps = len(reconstructed_trajectory.get("reconstructed_trajectory", []))
    
    compression_ratio = 1 - (reconstructed_steps / original_steps) if original_steps > 0 else 0
    
    return {
        "original_steps": original_steps,
        "reconstructed_steps": reconstructed_steps,
        "compression_ratio": compression_ratio
    }


def main():
    parser = argparse.ArgumentParser(description="Reconstruct reasoning chains")
    parser.add_argument("--config", type=str, required=True, help="Configuration file path")
    parser.add_argument("--input", type=str, required=True, help="Input trajectory file path")
    parser.add_argument("--output", type=str, required=True, help="Output reconstructed trajectory file path")
    parser.add_argument("--instruction_model", type=str, default="gpt-4o", help="Instruction model for reconstruction")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch processing size")
    parser.add_argument("--quality_threshold", type=float, default=0.8, help="Quality filtering threshold")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    logger.info(f"Starting reasoning chain reconstruction using instruction model: {args.instruction_model}")
    
    # Load trajectory data
    trajectories = load_trajectories(args.input)
    logger.info(f"Loaded {len(trajectories)} trajectories")
    
    # Initialize reconstructor and processor
    reconstructor = TrajectoryReconstructor(
        instruction_model_name=args.instruction_model,
        model_config=config.instruction_model
    )
    
    processor = TrajectoryProcessor(config.trajectory_processing)
    
    # Reconstruct trajectories
    reconstructed_trajectories = []
    quality_stats = {
        "total_processed": 0,
        "successful_reconstructions": 0,
        "failed_reconstructions": 0,
        "average_compression_ratio": 0.0,
        "quality_filtered": 0
    }
    
    for i, trajectory in enumerate(trajectories):
        try:
            logger.info(f"Reconstructing trajectory {i+1}/{len(trajectories)}")
            
            # Preprocess trajectory
            processed_trajectory = processor.preprocess_trajectory(trajectory)
            
            # Execute reconstruction
            reconstructed = reconstructor.reconstruct_trajectory(processed_trajectory)
            
            # Calculate quality metrics
            quality_metrics = calculate_quality_metrics(trajectory, reconstructed)
            
            # Quality filtering
            if quality_metrics["compression_ratio"] < 0.1:  # Compression ratio too low
                logger.warning(f"Trajectory {i+1} has low compression ratio, skipping")
                quality_stats["quality_filtered"] += 1
                continue
            
            # Post-processing
            final_trajectory = processor.postprocess_trajectory(reconstructed)
            
            # Add metadata
            final_trajectory.update({
                "id": f"reconstructed_{i:06d}",
                "original_trajectory_id": trajectory.get("id"),
                "reconstruction_model": args.instruction_model,
                "quality_metrics": quality_metrics,
                "reconstruction_metadata": {
                    "compression_ratio": quality_metrics["compression_ratio"],
                    "original_steps": quality_metrics["original_steps"],
                    "reconstructed_steps": quality_metrics["reconstructed_steps"]
                }
            })
            
            reconstructed_trajectories.append(final_trajectory)
            quality_stats["successful_reconstructions"] += 1
            quality_stats["average_compression_ratio"] += quality_metrics["compression_ratio"]
            
            # Periodically save progress
            if (i + 1) % args.batch_size == 0:
                logger.info(f"Completed {i+1} trajectory reconstructions, saving intermediate results...")
                save_trajectories(reconstructed_trajectories, args.output)
            
        except Exception as e:
            logger.error(f"Error reconstructing trajectory {i+1}: {e}")
            quality_stats["failed_reconstructions"] += 1
            continue
        
        quality_stats["total_processed"] += 1
    
    # Calculate final statistics
    if quality_stats["successful_reconstructions"] > 0:
        quality_stats["average_compression_ratio"] /= quality_stats["successful_reconstructions"]
    
    # Save final results
    save_trajectories(reconstructed_trajectories, args.output)
    
    # Save quality statistics
    stats_file = Path(args.output).with_suffix('.stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(quality_stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Reconstruction completed!")
    logger.info(f"Successful reconstructions: {quality_stats['successful_reconstructions']}")
    logger.info(f"Failed reconstructions: {quality_stats['failed_reconstructions']}")
    logger.info(f"Quality filtered: {quality_stats['quality_filtered']}")
    logger.info(f"Average compression ratio: {quality_stats['average_compression_ratio']:.2%}")
    logger.info(f"Results saved to: {args.output}")


def save_trajectories(trajectories: List[Dict[str, Any]], output_path: str):
    """Save reconstructed trajectory data"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(trajectories, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main() 