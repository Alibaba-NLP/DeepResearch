#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Rare Entities Script

This script uses large models to generate rare entities, which will be used to build knowledge graphs and generate Q&A pairs.
Supports batch generation and multi-domain entities.

Usage:
    python scripts/generate_entities.py --config configs/default_config.yaml --output data/generated_entities/entities.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from websailor.data_synthesis.entity_generator import EntityGenerator
from websailor.utils.config import Config
from websailor.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate rare entities")
    parser.add_argument("--config", type=str, required=True, help="Configuration file path")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--num_entities", type=int, default=200, help="Number of entities to generate")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch processing size")
    parser.add_argument("--entity_types", type=str, nargs="+", help="List of entity types")
    parser.add_argument("--domains", type=str, nargs="+", help="List of domains")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Set output path
    if args.output:
        config.set('entity_generator.storage_path', str(Path(args.output)))
    
    logger.info(f"Starting to generate rare entities, target count: {args.num_entities}")
    
    # Initialize entity generator
    entity_generator = EntityGenerator(config)
    
    # Generate entities
    generated_entities = entity_generator.generate_rare_entities(
        num_entities=args.num_entities,
        entity_types=args.entity_types,
        domains=args.domains,
        batch_size=args.batch_size
    )
    
    # Output statistics
    entity_types_count = {}
    domain_count = {}
    
    for entity in generated_entities:
        entity_types_count[entity.entity_type] = entity_types_count.get(entity.entity_type, 0) + 1
        domain_count[entity.domain] = domain_count.get(entity.domain, 0) + 1
    
    logger.info("\n=== Generation Statistics ===")
    logger.info(f"Total generated entities: {len(generated_entities)}")
    
    logger.info("\nEntity type distribution:")
    for entity_type, count in entity_types_count.items():
        logger.info(f"  - {entity_type}: {count}")
    
    logger.info("\nDomain distribution:")
    for domain, count in domain_count.items():
        logger.info(f"  - {domain}: {count}")
    
    logger.info("\nRarity distribution:")
    rarity_ranges = {
        "Very High (0.8-1.0)": len([e for e in generated_entities if 0.8 <= e.rarity_score <= 1.0]),
        "High (0.6-0.8)": len([e for e in generated_entities if 0.6 <= e.rarity_score < 0.8]),
        "Medium (0.4-0.6)": len([e for e in generated_entities if 0.4 <= e.rarity_score < 0.6]),
        "Low (0.2-0.4)": len([e for e in generated_entities if 0.2 <= e.rarity_score < 0.4]),
        "Very Low (0.0-0.2)": len([e for e in generated_entities if 0.0 <= e.rarity_score < 0.2])
    }
    
    for range_name, count in rarity_ranges.items():
        logger.info(f"  - {range_name}: {count}")
    
    logger.info("\nEntity generation completed!")


if __name__ == "__main__":
    main() 