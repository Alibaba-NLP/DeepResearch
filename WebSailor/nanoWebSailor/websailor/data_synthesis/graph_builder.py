#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Graph Builder

This module is responsible for building knowledge graphs from entities and relationships,
supporting dynamic graph expansion and maintenance.
Contains data structures for graph nodes, edges, and core graph building logic.

Main Classes:
- GraphNode: Graph node data class
- GraphEdge: Graph edge data class
- GraphBuilder: Graph builder

Features:
- Support entity to graph node conversion
- Support relationship to graph edge conversion
- Support dynamic graph expansion
- Support subgraph extraction
- Support graph serialization and persistence

Author: Evan Zuo
Date: January 2025
"""

import json
import uuid
import networkx as nx
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.config import Config
from ..utils.logger import get_logger
from .entity_generator import GeneratedEntity


@dataclass
class GraphNode:
    """Graph node data class"""
    id: str
    label: str
    entity_type: str
    domain: str
    properties: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'id': self.id,
            'label': self.label,
            'entity_type': self.entity_type,
            'domain': self.domain,
            'properties': self.properties,
            'features': self.features,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphNode':
        """Create node from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_entity(cls, entity: GeneratedEntity) -> 'GraphNode':
        """Create node from entity"""
        return cls(
            id=entity.id,
            label=entity.label,
            entity_type=entity.entity_type,
            domain=entity.domain,
            properties=entity.properties,
            features=entity.features,
            metadata=entity.metadata
        )


@dataclass
class GraphEdge:
    """Graph edge data class"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""
    target: str = ""
    relation_type: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'id': self.id,
            'source': self.source,
            'target': self.target,
            'relation_type': self.relation_type,
            'properties': self.properties,
            'weight': self.weight,
            'confidence': self.confidence,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphEdge':
        """Create edge from dictionary"""
        return cls(**data)


class GraphBuilder:
    """Knowledge Graph Builder
    
    Responsible for building knowledge graphs from entities and relationships,
    supporting dynamic expansion and maintenance.
    """
    
    def __init__(self, config: Config):
        """Initialize the builder
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Graph data structure
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.graph = nx.Graph()
        
        # Build configuration
        self.max_entities = config.get('data_synthesis.graph_builder.max_entities', 1000)
        self.max_relations = config.get('data_synthesis.graph_builder.max_relations', 5000)
        self.complexity_threshold = config.get('data_synthesis.graph_builder.complexity_threshold', 0.7)
        
        # Storage configuration
        self.storage_path = config.get('graph_builder.storage_path', 'data/knowledge_graphs')
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Graph builder initialization completed")
    
    def add_entity(self, entity: GeneratedEntity) -> GraphNode:
        """Add entity to the graph
        
        Args:
            entity: Entity object
            
        Returns:
            Created graph node
        """
        node = GraphNode.from_entity(entity)
        self.nodes[node.id] = node
        
        # Add to NetworkX graph
        self.graph.add_node(node.id, **node.to_dict())
        
        self.logger.debug(f"Added entity to graph: {node.label} (ID: {node.id})")
        return node
    
    def add_relation(self, source_id: str, target_id: str, 
                    relation_type: str, properties: Optional[Dict[str, Any]] = None,
                    weight: float = 1.0, confidence: float = 1.0) -> Optional[GraphEdge]:
        """Add relationship to the graph
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Relationship type
            properties: Relationship properties
            weight: Relationship weight
            confidence: Relationship confidence
            
        Returns:
            Created graph edge, returns None if failed
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            self.logger.warning(f"Cannot add relationship: nodes do not exist {source_id} -> {target_id}")
            return None
        
        edge = GraphEdge(
            source=source_id,
            target=target_id,
            relation_type=relation_type,
            properties=properties or {},
            weight=weight,
            confidence=confidence
        )
        
        self.edges.append(edge)
        
        # Add to NetworkX graph
        self.graph.add_edge(source_id, target_id, **edge.to_dict())
        
        self.logger.debug(f"Added relationship to graph: {relation_type} ({source_id} -> {target_id})")
        return edge
    
    def get_subgraph(self, node_ids: List[str]) -> Dict[str, Any]:
        """Get subgraph
        
        Args:
            node_ids: List of node IDs
            
        Returns:
            Dictionary containing nodes, edges, and NetworkX graph
        """
        # Filter nodes
        subgraph_nodes = {nid: node for nid, node in self.nodes.items() if nid in node_ids}
        
        # Filter edges (only include edges between nodes in the subgraph)
        subgraph_edges = [
            edge for edge in self.edges 
            if edge.source in node_ids and edge.target in node_ids
        ]
        
        # Create NetworkX subgraph
        subgraph = self.graph.subgraph(node_ids).copy()
        
        return {
            'nodes': subgraph_nodes,
            'edges': subgraph_edges,
            'graph': subgraph
        }
    
    def get_neighbors(self, node_id: str, depth: int = 1) -> Set[str]:
        """Get node neighbors
        
        Args:
            node_id: Node ID
            depth: Search depth
            
        Returns:
            Set of neighbor node IDs
        """
        if node_id not in self.nodes:
            return set()
        
        neighbors = set()
        current_level = {node_id}
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                if node in self.graph:
                    node_neighbors = set(self.graph.neighbors(node))
                    next_level.update(node_neighbors)
            
            neighbors.update(next_level)
            current_level = next_level
            
            if not current_level:
                break
        
        neighbors.discard(node_id)  # Remove self
        return neighbors
    
    def get_node_degree(self, node_id: str) -> int:
        """Get node degree
        
        Args:
            node_id: Node ID
            
        Returns:
            Node degree
        """
        if node_id in self.graph:
            return self.graph.degree(node_id)
        return 0
    
    def calculate_graph_stats(self) -> Dict[str, Any]:
        """Calculate graph statistics
        
        Returns:
            Dictionary of graph statistics
        """
        stats = {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'density': nx.density(self.graph) if len(self.nodes) > 1 else 0,
            'connected_components': nx.number_connected_components(self.graph),
            'average_clustering': nx.average_clustering(self.graph) if len(self.nodes) > 2 else 0,
            'average_degree': sum(dict(self.graph.degree()).values()) / len(self.nodes) if len(self.nodes) > 0 else 0
        }
        
        # Calculate diameter (only for connected graphs)
        if nx.is_connected(self.graph):
            stats['diameter'] = nx.diameter(self.graph)
        else:
            stats['diameter'] = 0
        
        return stats
    
    def save_graph(self, filename: Optional[str] = None) -> str:
        """Save graph to file
        
        Args:
            filename: Filename, auto-generated if None
            
        Returns:
            Saved file path
        """
        if filename is None:
            filename = f"graph_{uuid.uuid4().hex[:8]}.json"
        
        filepath = Path(self.storage_path) / filename
        
        # Prepare data for saving
        graph_data = {
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()},
            'edges': [edge.to_dict() for edge in self.edges],
            'stats': self.calculate_graph_stats(),
            'metadata': {
                'created_at': str(uuid.uuid4()),
                'num_nodes': len(self.nodes),
                'num_edges': len(self.edges)
            }
        }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Graph has been saved to: {filepath}")
        return str(filepath)
    
    def load_graph(self, filepath: str) -> bool:
        """Load graph from file
        
        Args:
            filepath: File path
            
        Returns:
            Whether loading was successful
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # Clear current graph
            self.nodes.clear()
            self.edges.clear()
            self.graph.clear()
            
            # Load nodes
            for node_data in graph_data['nodes'].values():
                node = GraphNode.from_dict(node_data)
                self.nodes[node.id] = node
                self.graph.add_node(node.id, **node.to_dict())
            
            # Load edges
            for edge_data in graph_data['edges']:
                edge = GraphEdge.from_dict(edge_data)
                self.edges.append(edge)
                self.graph.add_edge(edge.source, edge.target, **edge.to_dict())
            
            self.logger.info(f"Graph has been loaded from file: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load graph: {e}")
            return False
    
    def clear(self):
        """Clear graph data"""
        self.nodes.clear()
        self.edges.clear()
        self.graph.clear()
        self.logger.info("Graph data has been cleared") 