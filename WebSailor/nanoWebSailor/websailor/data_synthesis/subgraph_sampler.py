#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subgraph Sampler

This module is responsible for sampling subgraphs with different topological structures 
from knowledge graphs for generating diverse QA pairs.
Supports multiple sampling strategies to ensure subgraphs have different complexity 
and structural characteristics.

Main Classes:
- SubgraphSampler: Subgraph sampler
- SamplingStrategy: Sampling strategy enumeration
- SubgraphData: Subgraph data class

Features:
- Multiple sampling strategies (random, BFS, DFS, community detection, etc.)
- Topological diversity guarantee
- Subgraph complexity control
- Sampling result serialization
- Sampling statistics and analysis

Author: Evan Zuo
Date: January 2025
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import uuid
import random
import networkx as nx
from pathlib import Path
from loguru import logger

from websailor.utils.config import Config
from websailor.utils.logger import get_logger
from websailor.data_synthesis.graph_builder import GraphBuilder, GraphNode, GraphEdge


random.seed(42)
BFS_BRANCHING_FACTOR = 3
DFS_BRANCHING_FACTOR = 2
TREE_BRANCHING_FACTOR = 2


class SamplingStrategy(Enum):
    """Sampling strategy enumeration"""
    RANDOM = "random"
    BFS = "bfs"
    DFS = "dfs"
    COMMUNITY = "community"
    STAR = "star"
    CHAIN = "chain"
    TREE = "tree"
    CLIQUE = "clique"


@dataclass
class SubgraphData:
    """Subgraph data class"""
    id: str
    nodes: Dict[str, GraphNode]
    edges: List[GraphEdge]
    graph: nx.Graph
    strategy: SamplingStrategy
    complexity_score: float = 0.0
    topology_features: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'id': self.id,
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()},
            'edges': [edge.to_dict() for edge in self.edges],
            'strategy': self.strategy.value,
            'complexity_score': self.complexity_score,
            'topology_features': self.topology_features
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubgraphData':
        """Create subgraph data from dictionary"""
        # Rebuild nodes
        nodes = {}
        for nid, node_data in data['nodes'].items():
            nodes[nid] = GraphNode.from_dict(node_data)
        
        # Rebuild edges
        edges = [GraphEdge.from_dict(edge_data) for edge_data in data['edges']]
        
        # Rebuild graph
        graph = nx.Graph()
        for node in nodes.values():
            graph.add_node(node.id, **node.to_dict())
        for edge in edges:
            graph.add_edge(edge.source, edge.target, **edge.to_dict())
        
        return cls(
            id=data['id'],
            nodes=nodes,
            edges=edges,
            graph=graph,
            strategy=SamplingStrategy(data['strategy']),
            complexity_score=data.get('complexity_score', 0.0),
            topology_features=data.get('topology_features', {})
        )


class SubgraphSampler:
    """Subgraph Sampler
    
    Samples subgraphs with different topological structures from knowledge graphs, 
    supporting multiple sampling strategies.
    """
    
    def __init__(self, config: Config):
        """Initialize sampler
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Sampling configuration
        self.min_nodes = config.get('subgraph_sampler.min_nodes', 5)
        self.max_nodes = config.get('subgraph_sampler.max_nodes', 20)
        self.num_samples = config.get('subgraph_sampler.num_samples', 10)
        self.diversity_threshold = config.get('subgraph_sampler.diversity_threshold', 0.3)
        
        # Sampling strategy weights
        self.strategy_weights = {
            SamplingStrategy.RANDOM: 0.2,
            SamplingStrategy.BFS: 0.2,
            SamplingStrategy.DFS: 0.2,
            SamplingStrategy.COMMUNITY: 0.15,
            SamplingStrategy.STAR: 0.1,
            SamplingStrategy.CHAIN: 0.1,
            SamplingStrategy.TREE: 0.05
        }
        
        self.logger.info("Subgraph sampler initialization completed")

    def sample_subgraphs(self, graph_builder: GraphBuilder, 
                        num_samples: Optional[int] = None,
                        strategies: Optional[List[SamplingStrategy]] = None,
                        storage_path: str = "") -> List[SubgraphData]:
        """Sample subgraphs
        
        Args:
            graph_builder: Graph builder instance
            num_samples: Number of samples, if None use config value
            strategies: List of sampling strategies, if None use all strategies
            
        Returns:
            List of sampled subgraphs
        """
        if num_samples is None:
            num_samples = self.num_samples
        if strategies is None:
            strategies = self.strategy_weights

        self.logger.info(f"Starting to sample {num_samples} subgraphs")
        self.logger.info(f"Subgraph Sample Strategies: {strategies}")
        
        subgraphs = []
        attempts = 0
        max_attempts = num_samples * 10
        
        while len(subgraphs) < num_samples and attempts < max_attempts:
            # Randomly select strategy based on weights
            strategy = random.choices(
                population=list(strategies.keys()),
                weights=list(strategies.values()),
                k=1
            )[0]

            # Sample subgraph
            subgraph = self._sample_with_strategy(graph_builder, strategy)
            
            if subgraph and self._is_diverse(subgraph, subgraphs):
                subgraphs.append(subgraph)
                self._save_subgraph(subgraph, storage_path)
            
            attempts += 1
        
        self.logger.info(f"Successfully sampled {len(subgraphs)} subgraphs")
        return subgraphs
    
    def _sample_with_strategy(self, graph_builder: GraphBuilder, 
                            strategy: SamplingStrategy) -> Optional[SubgraphData]:
        """Sample subgraph using specified strategy
        
        Args:
            graph_builder: Graph builder instance
            strategy: Sampling strategy
            
        Returns:
            Sampled subgraph, returns None if failed
        """
        try:
            if strategy == SamplingStrategy.RANDOM:
                return self._random_sampling(graph_builder)
            elif strategy == SamplingStrategy.BFS:
                return self._bfs_sampling(graph_builder)
            elif strategy == SamplingStrategy.DFS:
                return self._dfs_sampling(graph_builder)
            elif strategy == SamplingStrategy.COMMUNITY:
                return self._community_sampling(graph_builder)
            elif strategy == SamplingStrategy.STAR:
                return self._star_sampling(graph_builder)
            elif strategy == SamplingStrategy.CHAIN:
                return self._chain_sampling(graph_builder)
            elif strategy == SamplingStrategy.TREE:
                return self._tree_sampling(graph_builder)
            else:
                self.logger.warning(f"Unknown sampling strategy: {strategy}")
                return None
                
        except Exception as e:
            self.logger.error(f"Sampling failed (strategy: {strategy}): {e}")
            return None
    
    def _random_sampling(self, graph_builder: GraphBuilder) -> Optional[SubgraphData]:
        """Random sampling"""
        node_ids = list(graph_builder.nodes.keys())
        if len(node_ids) < self.min_nodes:
            return None
        
        # Randomly select number of nodes
        num_nodes = random.randint(self.min_nodes, min(self.max_nodes, len(node_ids)))
        selected_nodes = random.sample(node_ids, num_nodes)
        
        return self._create_subgraph(graph_builder, selected_nodes, SamplingStrategy.RANDOM)
    
    def _bfs_sampling(self, graph_builder: GraphBuilder) -> Optional[SubgraphData]:
        """Breadth-first search sampling"""
        node_ids = list(graph_builder.nodes.keys())
        if len(node_ids) < self.min_nodes:
            return None
        
        # Randomly select starting node
        start_node = random.choice(node_ids)
        
        # BFS traversal
        visited = set()
        queue = [start_node]
        selected_nodes = []
        
        while queue and len(selected_nodes) < self.max_nodes:
            current = queue.pop(0)
            if current not in visited:
                visited.add(current)
                selected_nodes.append(current)
                
                # Add neighbor nodes
                neighbors = list(graph_builder.graph.neighbors(current))
                random.shuffle(neighbors)
                queue.extend(neighbors[:BFS_BRANCHING_FACTOR])  # Limit branching factor
        
        if len(selected_nodes) >= self.min_nodes:
            return self._create_subgraph(graph_builder, selected_nodes, SamplingStrategy.BFS)
        return None
    
    def _dfs_sampling(self, graph_builder: GraphBuilder) -> Optional[SubgraphData]:
        """Depth-first search sampling"""
        node_ids = list(graph_builder.nodes.keys())
        if len(node_ids) < self.min_nodes:
            return None
        
        # Randomly select starting node
        start_node = random.choice(node_ids)
        
        # DFS traversal
        visited = set()
        stack = [start_node]
        selected_nodes = []
        
        while stack and len(selected_nodes) < self.max_nodes:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                selected_nodes.append(current)
                
                # Add neighbor nodes
                neighbors = list(graph_builder.graph.neighbors(current))
                random.shuffle(neighbors)
                stack.extend(neighbors[:DFS_BRANCHING_FACTOR])  # Limit branching factor
        
        if len(selected_nodes) >= self.min_nodes:
            return self._create_subgraph(graph_builder, selected_nodes, SamplingStrategy.DFS)
        return None
    
    def _community_sampling(self, graph_builder: GraphBuilder) -> Optional[SubgraphData]:
        """Community detection sampling"""
        if len(graph_builder.nodes) < self.min_nodes:
            return None
        
        try:
            # Use Louvain algorithm to detect communities
            communities = nx.community.louvain_communities(graph_builder.graph)
            
            # Select the largest community
            largest_community = max(communities, key=len)
            
            if len(largest_community) >= self.min_nodes:
                # If community is too large, randomly sample
                if len(largest_community) > self.max_nodes:
                    selected_nodes = random.sample(list(largest_community), self.max_nodes)
                else:
                    selected_nodes = list(largest_community)
                
                return self._create_subgraph(graph_builder, selected_nodes, SamplingStrategy.COMMUNITY)
        except Exception as e:
            self.logger.warning(f"Community detection failed: {e}")
        
        return None
    
    def _star_sampling(self, graph_builder: GraphBuilder) -> Optional[SubgraphData]:
        """Star structure sampling"""
        node_ids = list(graph_builder.nodes.keys())
        if len(node_ids) < self.min_nodes:
            return None
        
        # Select center node
        center_node = random.choice(node_ids)
        
        # Get neighbors of center node
        neighbors = list(graph_builder.graph.neighbors(center_node))
        
        if len(neighbors) >= self.min_nodes - 1:
            # Select neighbor nodes
            num_neighbors = min(len(neighbors), self.max_nodes - 1)
            selected_neighbors = random.sample(neighbors, num_neighbors)
            
            selected_nodes = [center_node] + selected_neighbors
            return self._create_subgraph(graph_builder, selected_nodes, SamplingStrategy.STAR)
        
        return None
    
    def _chain_sampling(self, graph_builder: GraphBuilder) -> Optional[SubgraphData]:
        """Chain structure sampling"""
        node_ids = list(graph_builder.nodes.keys())
        if len(node_ids) < self.min_nodes:
            return None
        
        # Randomly select starting node
        start_node = random.choice(node_ids)
        
        # Build chain structure
        chain = [start_node]
        current = start_node
        
        while len(chain) < self.max_nodes:
            neighbors = list(graph_builder.graph.neighbors(current))
            unvisited = [n for n in neighbors if n not in chain]
            
            if not unvisited:
                break
            
            # Randomly select next node
            next_node = random.choice(unvisited)
            chain.append(next_node)
            current = next_node
        
        if len(chain) >= self.min_nodes:
            return self._create_subgraph(graph_builder, chain, SamplingStrategy.CHAIN)
        
        return None
    
    def _tree_sampling(self, graph_builder: GraphBuilder) -> Optional[SubgraphData]:
        """Tree structure sampling"""
        node_ids = list(graph_builder.nodes.keys())
        if len(node_ids) < self.min_nodes:
            return None
        
        # Randomly select root node
        root = random.choice(node_ids)
        
        # Build tree structure
        tree_nodes = [root]
        queue = [root]
        
        while queue and len(tree_nodes) < self.max_nodes:
            current = queue.pop(0)
            neighbors = list(graph_builder.graph.neighbors(current))
            unvisited = [n for n in neighbors if n not in tree_nodes]
            
            # Limit number of children for each node
            children = random.sample(unvisited, min(len(unvisited), TREE_BRANCHING_FACTOR))
            
            for child in children:
                if len(tree_nodes) < self.max_nodes:
                    tree_nodes.append(child)
                    queue.append(child)
        
        if len(tree_nodes) >= self.min_nodes:
            return self._create_subgraph(graph_builder, tree_nodes, SamplingStrategy.TREE)
        
        return None
    
    def _create_subgraph(self, graph_builder: GraphBuilder, 
                        node_ids: List[str], strategy: SamplingStrategy) -> SubgraphData | None:
        """Create subgraph data
        
        Args:
            graph_builder: Graph builder instance
            node_ids: List of node IDs
            strategy: Sampling strategy
            
        Returns:
            Subgraph data
        """
        # Get subgraph
        subgraph_data = graph_builder.get_subgraph(node_ids)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity(subgraph_data['graph'])
        
        # Calculate topology features
        topology_features = self._calculate_topology_features(subgraph_data['graph'])
        
        if topology_features["connected_components"] == 1:
            return SubgraphData(
                id=str(uuid.uuid4()),
                nodes=subgraph_data['nodes'],
                edges=subgraph_data['edges'],
                graph=subgraph_data['graph'],
                strategy=strategy,
                complexity_score=complexity_score,
                topology_features=topology_features
            )
        else:
            self.logger.warning(f"The sampled subgraph by {strategy} is not connected, it has {topology_features['connected_components']} components")
            return None
    
    def _calculate_complexity(self, graph: nx.Graph) -> float:
        """Calculate subgraph complexity score
        
        Args:
            graph: Subgraph
            
        Returns:
            Complexity score (0-1)
        """
        if len(graph.nodes()) == 0:
            return 0.0
        
        # Calculate various complexity metrics
        density = nx.density(graph)
        clustering = nx.average_clustering(graph)
        avg_degree = sum(dict(graph.degree()).values()) / len(graph.nodes())
        
        # Normalize and weight
        complexity = (density * 0.3 + clustering * 0.4 + min(avg_degree / 10, 1.0) * 0.3)
        return complexity
    
    def _calculate_topology_features(self, graph: nx.Graph) -> Dict[str, Any]:
        """Calculate topology features
        
        Args:
            graph: Subgraph
            
        Returns:
            Topology features dictionary
        """
        if len(graph.nodes()) == 0:
            return {}
        
        features = {
            'num_nodes': len(graph.nodes()),
            'num_edges': len(graph.edges()),
            'density': nx.density(graph),
            'clustering_coefficient': nx.average_clustering(graph),
            'avg_degree': sum(dict(graph.degree()).values()) / len(graph.nodes()),
            'diameter': nx.diameter(graph) if nx.is_connected(graph) else 0,
            'connected_components': nx.number_connected_components(graph)
        }
        
        return features
    
    def _is_diverse(self, new_subgraph: SubgraphData, 
                   existing_subgraphs: List[SubgraphData]) -> bool:
        """Check if the subgraph is sufficiently different from existing subgraphs
        
        Args:
            new_subgraph: New subgraph
            existing_subgraphs: List of existing subgraphs
            
        Returns:
            Whether it is sufficiently different
        """
        if not existing_subgraphs:
            return True
        
        for existing in existing_subgraphs:
            # Calculate similarity
            similarity = self._calculate_similarity(new_subgraph, existing)
            if similarity > (1.0 - self.diversity_threshold):
                logger.info(f"Generated subgraph is similar with existing subgraph, similarity: {similarity}")
                return False
        
        return True
    
    def _calculate_similarity(self, subgraph1: SubgraphData, 
                            subgraph2: SubgraphData) -> float:
        """Calculate similarity between two subgraphs
        
        Args:
            subgraph1: First subgraph
            subgraph2: Second subgraph
            
        Returns:
            Similarity score (0-1)
        """
        # Similarity based on topological features
        features1 = subgraph1.topology_features
        features2 = subgraph2.topology_features
        
        if not features1 or not features2:
            return 0.0
        
        # Calculate feature differences
        differences = []
        for key in ['density', 'clustering_coefficient', 'avg_degree']:
            if key in features1 and key in features2:
                diff = abs(features1[key] - features2[key])
                differences.append(diff)
        
        if not differences:
            return 0.0
        
        # Normalize differences
        avg_diff = sum(differences) / len(differences)
        similarity = 1.0 - min(avg_diff, 1.0)
        
        return similarity
    
    def _save_subgraph(self, subgraph: SubgraphData, storage_path: str) -> None:
        """Save subgraph to file
        
        Args:
            subgraph: Subgraph data
        """
        try:
            subgraph_file = Path(storage_path) / f"{subgraph.id}.json"

            logger.info(f"Saving to {subgraph_file}")

            os.makedirs(Path(storage_path), exist_ok=True)
            
            data = subgraph.to_dict()
            data['statistics'] = subgraph.topology_features
            
            with open(subgraph_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to save subgraph: {e}")
    
    def load_subgraph(self, subgraph_id: str, graph_builder: GraphBuilder, storage_path) -> Optional[SubgraphData]:
        """Load subgraph
        
        Args:
            subgraph_id: Subgraph ID
            graph_builder: Graph builder instance
            
        Returns:
            Subgraph data, returns None if failed
        """
        try:
            subgraph_file = Path(storage_path) / f"{subgraph_id}.json"
            if not subgraph_file.exists():
                self.logger.error(f"Subgraph file does not exist: {subgraph_file}")
                return None
            
            with open(subgraph_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return SubgraphData.from_dict(data, graph_builder)
            
        except Exception as e:
            self.logger.error(f"Failed to load subgraph: {e}")
            return None
    
    def list_subgraphs(self, storage_path: str) -> List[str]:
        """List all available subgraphs
        
        Returns:
            List of subgraph IDs
        """
        subgraphs = []
        for file in Path(storage_path).glob("*.json"):
            subgraphs.append(file.stem)
        return subgraphs 

    