#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subgraph Sampling Script

This script is used to sample subgraphs from expanded knowledge graphs, supporting multiple sampling strategies.
It can generate subgraphs with different topological structures for subsequent tasks such as QA generation.

Author: Evan Zuo
Date: January 2025
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from websailor.utils.config import Config
from websailor.utils.logger import get_logger
from websailor.data_synthesis.subgraph_sampler import SubgraphSampler, SubgraphData, SamplingStrategy
from websailor.data_synthesis.graph_builder import GraphBuilder, GraphNode, GraphEdge
from scripts.visualize_graphs import visualize_graph


def load_expanded_graph(file_path: str) -> Dict[str, Any]:
    """Load expanded knowledge graph

    Args:
        file_path: Graph file path

    Returns:
        Graph data dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def sample_from_graph(graph_data: Dict[str, Any], 
                     sampler: SubgraphSampler,
                     num_samples: int = 5,
                     config: Config = None,
                     storage_path: str = "") -> List[SubgraphData]:
    """Sample subgraphs from a single graph

    Args:
        graph_data: Graph data
        sampler: Sampler instance
        num_samples: Number of samples

    Returns:
        List of sampled subgraphs
    """
    # Create GraphBuilder instance and load data
    graph_builder = GraphBuilder(Config())
    
    # Rebuild GraphBuilder from expanded graph data
    for node_id, node_data in graph_data['nodes'].items():
        node = GraphNode.from_dict(node_data)
        graph_builder.nodes[node.id] = node
        graph_builder.graph.add_node(node.id, **node.to_dict())
    
    for edge_data in graph_data['edges']:
        edge = GraphEdge.from_dict(edge_data)
        graph_builder.edges.append(edge)
        graph_builder.graph.add_edge(edge.source, edge.target, **edge.to_dict())

    strategies = {
        SamplingStrategy.RANDOM: config.get("subgraph_sampler.strategy_weights.random"),
        SamplingStrategy.BFS: config.get("subgraph_sampler.strategy_weights.bfs"),
        SamplingStrategy.DFS: config.get("subgraph_sampler.strategy_weights.dfs"),
        SamplingStrategy.COMMUNITY: config.get("subgraph_sampler.strategy_weights.community"),
        SamplingStrategy.STAR: config.get("subgraph_sampler.strategy_weights.star"),
        SamplingStrategy.CHAIN: config.get("subgraph_sampler.strategy_weights.chain"),
        SamplingStrategy.TREE: config.get("subgraph_sampler.strategy_weights.tree"),
    }
    
    # Sample subgraphs
    subgraphs = sampler.sample_subgraphs(
        graph_builder=graph_builder,
        num_samples=num_samples,
        strategies=strategies,
        storage_path=storage_path
    )
    
    return subgraphs


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Sample subgraphs from knowledge graphs')
    parser.add_argument('--config', type=str, default='../configs/default_config.yaml',
                      help='Configuration file path')
    parser.add_argument('--input-dir', type=str, default='data/expanded_graphs',
                      help='Input graph directory')
    parser.add_argument('--output-dir', type=str, default='data/sampled_subgraphs',
                      help='Output subgraph directory')
    args = parser.parse_args()

    # Load configuration
    config = Config(args.config)
    # config.set("subgraph_sampler.storage_path", args.output_dir)
    
    # Create sampler
    sampler = SubgraphSampler(config)

    logger = get_logger(__name__, config.get("subgraph_sampler.log_file"))
    
    # Print configuration parameters and save to log
    config.print_config()
    
    # Get all expanded graph files
    graph_files = list(Path(args.input_dir).glob('*.json'))
    logger.info(f"Found {len(graph_files)} expanded graph files")
    
    total_samples = 0
    
    # Process each graph file
    for graph_file in graph_files:
        logger.info(f"Processing graph file: {graph_file.name}")
        
        try:
            # Load graph
            graph_data = load_expanded_graph(str(graph_file))

            # Change save path
            project_dir = Path(__file__).parent.parent
            graph_save_dir = os.path.join(project_dir, "data", "sampled_subgraphs", "20250718_1", graph_file.stem, "graph")
            vis_save_dir = os.path.join(project_dir, "data", "sampled_subgraphs", "20250718_1", graph_file.stem, "visual") 
            
            # Sample subgraphs
            subgraphs = sample_from_graph(
                graph_data=graph_data,
                sampler=sampler,
                num_samples=None,
                config=config,
                storage_path=graph_save_dir
            )

            # Generate visualization images
            visualize_graph(
                input_path=graph_save_dir,
                output_dir=vis_save_dir
            )
            
            total_samples += len(subgraphs)
            
            logger.info(f"Successfully sampled {len(subgraphs)} subgraphs from {graph_file.name}")
            
        except Exception as e:
            logger.error(f"Error processing {graph_file.name}: {e}")
            continue
    
    logger.info(f"Sampling completed! Total generated {total_samples} subgraphs")


if __name__ == '__main__':
    main() 