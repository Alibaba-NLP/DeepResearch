#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SailorFog-QA Dataset Generation Script

This script is used to generate high-quality question-answer pairs from sampled subgraphs to build the SailorFog-QA dataset.
Supports batch processing, QA generation at different difficulty levels, and serialized saving of results.

Author: Evan Zuo
Date: January 2025
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

from websailor.utils.config import Config
from websailor.data_synthesis.qa_generator import QAGenerator
from websailor.data_synthesis.subgraph_sampler import SubgraphData
from websailor.utils.logger import get_logger


def get_all_directories_path(path):
    # 获取所有条目
    all_entries = os.listdir(path)
    # 过滤出文件夹
    directories = [os.path.join(path, entry) for entry in all_entries if os.path.isdir(os.path.join(path, entry))]
    return directories


def contains_dir(path, dir_name):
    dir_path = os.path.join(path, dir_name)
    return os.path.exists(dir_path) and os.path.isdir(dir_path)


def load_subgraph_from_file(file_path: str) -> SubgraphData:
    """Load subgraph data from file
    
    Args:
        file_path: Subgraph file path
        
    Returns:
        Subgraph data object
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return SubgraphData.from_dict(data)


def generate_qa_from_subgraphs(config: Config, input_dir: str, output_dir: str, 
                              num_qa_per_subgraph: int, file_pattern: str = "*.json"):
    """Generate QA pairs from subgraph directory
    
    Args:
        config: Configuration object
        input_dir: Input subgraph directory
        output_dir: Output QA pairs directory
        num_qa_per_subgraph: Number of QA pairs to generate per subgraph
        file_pattern: File matching pattern
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize QA generator
    qa_generator = QAGenerator(config)
    
    # Find subgraph files
    input_path = Path(input_dir)
    subgraph_files = list(input_path.glob(file_pattern))
    
    if not subgraph_files:
        logger.error(f"No matching subgraph files found in directory {input_dir}")
        return
    
    logger.info(f"Found {len(subgraph_files)} subgraph files")
    
    all_qa_pairs = []
    successful_files = 0
    
    for i, file_path in enumerate(subgraph_files):
        logger.info(f"Processing file {i+1}/{len(subgraph_files)}: {file_path.name}")
        
        try:
            # Load from sampled subgraph directory
            subgraph = load_subgraph_from_file(str(file_path))
            
            # Generate QA pairs
            qa_pairs = qa_generator.generate_qa_pairs(subgraph, num_qa_per_subgraph)
            
            if qa_pairs:
                all_qa_pairs.extend(qa_pairs)
                successful_files += 1
                
                # Save QA pairs for individual file
                qa_output_file = Path(output_dir) / f"{file_path.stem}_qa.json"
                with open(qa_output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'subgraph_id': subgraph.id,
                        'source_file': str(file_path),
                        'qa_pairs': qa_pairs,
                        'metadata': {
                            'num_qa_pairs': len(qa_pairs),
                            'strategy': subgraph.strategy.value,
                            'complexity_score': subgraph.complexity_score,
                            'topology_features': subgraph.topology_features
                        }
                    }, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Generated {len(qa_pairs)} QA pairs for {file_path.name}")
            else:
                logger.warning(f"No QA pairs generated for file {file_path.name}")
                
        except Exception as e:
            logger.error(f"Failed to process file {file_path.name}: {e}")
            continue
    
    # Save summary results
    summary_file = Path(output_dir) / "sailorfog_qa_dataset.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'dataset_info': {
                'name': 'SailorFog-QA',
                'description': 'High-difficulty QA dataset generated from complex knowledge graph subgraphs',
                'total_qa_pairs': len(all_qa_pairs),
                'successful_files': successful_files,
                'total_files': len(subgraph_files),
                'generation_config': {
                    'qa_pairs_per_subgraph': num_qa_per_subgraph,
                    'difficulty_levels': config.get('data_synthesis.qa_generator.difficulty_levels', [1, 2, 3])
                }
            },
            'qa_pairs': all_qa_pairs
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"QA generation completed!")
    logger.info(f"Successfully processed files: {successful_files}/{len(subgraph_files)}")
    logger.info(f"Total QA pairs generated: {len(all_qa_pairs)}")
    logger.info(f"Summary file saved to: {summary_file}")


def generate_qa_from_single_file(config: Config, input_file: str, output_file: str, 
                                num_qa_pairs: int):
    """Generate QA pairs from a single subgraph file
    
    Args:
        config: Configuration object
        input_file: Input subgraph file
        output_file: Output QA file
        num_qa_pairs: Number of QA pairs to generate
    """
    # Initialize QA generator
    qa_generator = QAGenerator(config)
    
    try:
        # Load subgraph
        subgraph = load_subgraph_from_file(input_file)
        
        logger.info(f"Loaded subgraph: {subgraph.id}")
        logger.info(f"Subgraph statistics: {subgraph.topology_features}")
        
        # Generate QA pairs
        qa_pairs = qa_generator.generate_qa_pairs(subgraph, num_qa_pairs)
        
        # Save results
        result = {
            'subgraph_info': {
                'id': subgraph.id,
                'strategy': subgraph.strategy.value,
                'complexity_score': subgraph.complexity_score,
                'topology_features': subgraph.topology_features,
                'source_file': input_file
            },
            'qa_pairs': qa_pairs,
            'metadata': {
                'num_qa_pairs': len(qa_pairs),
                'generation_timestamp': str(Path().cwd())
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Successfully generated {len(qa_pairs)} QA pairs")
        logger.info(f"Results saved to: {output_file}")
        
        # Print some example QA pairs
        if qa_pairs:
            logger.info("Generated QA pair examples:")
            for i, qa in enumerate(qa_pairs[:3]):  # Show first 3
                logger.info(f"QA {i+1} (difficulty {qa.get('difficulty', 'N/A')}):")
                logger.info(f"  Question: {qa.get('question', 'N/A')}")
                logger.info(f"  Answer: {qa.get('answer', 'N/A')[:100]}...")
                logger.info("")
        
    except Exception as e:
        logger.error(f"Failed to process file: {e}")


def analyze_qa_dataset(qa_file: str):
    """Analyze statistical information of QA dataset
    
    Args:
        qa_file: QA dataset file path
    """
    try:
        with open(qa_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        qa_pairs = data.get('qa_pairs', [])
        
        if not qa_pairs:
            logger.warning("No QA pairs in the dataset")
            return
        
        # Statistical information
        difficulty_stats = {}
        question_types = {}
        
        for qa in qa_pairs:
            # Difficulty statistics
            difficulty = qa.get('difficulty', 'unknown')
            difficulty_stats[difficulty] = difficulty_stats.get(difficulty, 0) + 1
            
            # Question type statistics (if available)
            question_type = qa.get('question_type', 'general')
            question_types[question_type] = question_types.get(question_type, 0) + 1
        
        logger.info("=== QA Dataset Analysis ===")
        logger.info(f"Total QA pairs: {len(qa_pairs)}")
        logger.info(f"Difficulty distribution: {difficulty_stats}")
        logger.info(f"Question type distribution: {question_types}")
        
        # Analyze data quality
        valid_qa = 0
        for qa in qa_pairs:
            if qa.get('question') and qa.get('answer'):
                valid_qa += 1
        
        logger.info(f"Valid QA pairs: {valid_qa}/{len(qa_pairs)} ({valid_qa/len(qa_pairs)*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"Failed to analyze dataset: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate SailorFog-QA dataset")
    parser.add_argument("--config", default="configs/default_config.yaml", 
                    help="Configuration file path")
    parser.add_argument("--input-dir", default="data/expanded_graphs",
                    help="Input subgraph directory path")
    parser.add_argument("--output-dir", default="data/sailorfog_qa",
                    help="Output QA pairs directory path")
    parser.add_argument("--single-file", type=str,
                    help="Process single subgraph file")
    parser.add_argument("--output-file", type=str,
                    help="Single file output path")
    parser.add_argument("--num-qa", type=int, default=5,
                    help="Number of QA pairs to generate per subgraph")
    parser.add_argument("--analyze", type=str,
                    help="Analyze specified QA dataset file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)

    logger = get_logger(__name__, config.get("qa_generator.log_file"))
    config.print_config()
    
    if args.analyze:
        # Analysis mode
        analyze_qa_dataset(args.analyze)
    elif args.single_file:
        # Single file mode
        output_file = args.output_file or f"{Path(args.single_file).stem}_qa.json"
        generate_qa_from_single_file(config, args.single_file, output_file, args.num_qa)
    else:
        # Batch processing mode
        input_dir = args.input_dir
        output_dir = args.output_dir

        if input_dir.split("/")[-1] != "graph":
            all_dirs = get_all_directories_path(input_dir)

            for dir in all_dirs:
                if contains_dir(path=dir, dir_name="graph"):
                    subgraph_dir = os.path.join(dir, "graph")
                    subgraph_id = os.path.basename(dir)
                    save_dir = os.path.join(output_dir, subgraph_id)
                    logger.info(f"Currently processing {dir}")
                    generate_qa_from_subgraphs(config, subgraph_dir, save_dir, args.num_qa)
                else:
                    logger.error(f"{dir} doesn't have a graph directory")
        
        elif input_dir.split("/")[-1] == "graph":
            generate_qa_from_subgraphs(config, input_dir, output_dir, args.num_qa)


if __name__ == "__main__":
    main() 