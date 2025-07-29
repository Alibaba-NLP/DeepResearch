#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph Expansion Main Script

This script handles batch processing of rare entities to generate expanded knowledge graphs.
According to paper requirements, it builds complex, non-linear graph structures starting from rare entities.

Usage:
    python scripts/expand_graphs.py --entities_file data/generated_entities/entities.json
    python scripts/expand_graphs.py --entities_file data/generated_entities/entities.json --config configs/default_config.yaml

Author: Evan Zuo
Date: January 2025
"""

import sys
import argparse
from pathlib import Path

# Add project root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from websailor.utils.config import Config
from websailor.utils.logger import get_logger
from websailor.data_synthesis.graph_expander import expand_entities_from_file


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Batch expand rare entities to generate knowledge graphs")
    
    parser.add_argument(
        "--entities_file", 
        type=str, 
        default="data/generated_entities/entities.json",
        help="Rare entities file path"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="../configs/default_config.yaml",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="../data/expanded_graphs",
        help="Output directory"
    )
    
    return parser.parse_args()


def validate_inputs(args):
    """Validate input parameters"""
    entities_path = Path(args.entities_file)
    if not entities_path.exists():
        raise FileNotFoundError(f"Entities file does not exist: {entities_path}")
    
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")


def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Validate input
    validate_inputs(args)
    
    # Load configuration
    config = Config(args.config)

    logger = get_logger(__name__, config.get("graph_expander.log_path"))
    
    # Print configuration parameters and save to log
    config.print_config()
    
    # Start expansion
    logger.info("Starting batch knowledge graph expansion...")
    
    # Execute graph expansion
    saved_graphs = expand_entities_from_file(args.entities_file, config, args.output_dir)
    
    # Output result statistics
    logger.info(f"Graph expansion completed!")
    logger.info(f"Successfully generated graphs: {len(saved_graphs)}")
    logger.info(f"Graph files saved to directory: {args.output_dir}")


if __name__ == "__main__":
    main() 