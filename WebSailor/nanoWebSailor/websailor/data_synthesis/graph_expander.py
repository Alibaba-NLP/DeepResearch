#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Graph Expander

This module is responsible for expanding knowledge graphs starting from rare entities,
building complex, non-linear graph structures through simulated web search and triplet extraction,
meeting the requirements of Level 3 tasks in the paper.

Main features:
- Iteratively expand graph from seed entities
- Use simulated web search to obtain related information
- Discover entity relationships through triplet extraction
- Build densely interconnected graph structures
- Support probabilistic node selection and expansion

Author: Evan Zuo
Date: January 2025
"""

import json
import os
import random
import time
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import importlib
from tenacity import RetryError

from websailor.utils.config import Config
from websailor.utils.logger import get_logger
from websailor.data_synthesis.graph_builder import GraphBuilder, GraphNode, GraphEdge
from websailor.data_synthesis.mock_web_search import MockWebSearch
from websailor.data_synthesis.triplet_extractor import TripleExtractor
from websailor.data_synthesis.entity_generator import GeneratedEntity


class GraphExpander:
    """Knowledge Graph Expander
    
    Iteratively expands knowledge graphs from rare entities to build complex non-linear graph structures.
    """
    
    def __init__(self, config: Config, save_dir: str):
        """Initialize graph expander
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.graph_builder = GraphBuilder(config)
        self.web_search = MockWebSearch(config)
        self.triplet_extractor = TripleExtractor(config)
        
        # Expansion configuration
        self.max_depth = config.get('graph_expander.max_depth', 3)
        self.max_branches_per_node = config.get('graph_expander.max_branches_per_node', 3)
        self.min_confidence = config.get('graph_expander.min_confidence', 0.7)
        self.expansion_probability = config.get('graph_expander.expansion_probability', 0.8)
        self.max_search_results = config.get('graph_expander.max_search_results', 3)

        # Storage configuration
        self.save_dir = save_dir
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Knowledge graph expander initialization completed")
    
    def expand_from_entity_dict(self, entity_dict: Dict) -> str:
        """Expand knowledge graph from entity dictionary
        
        Args:
            entity_dict: Entity dictionary data
            
        Returns:
            Path to the saved graph file
        """
        self.logger.info(f"Starting graph expansion from entity: {entity_dict['label']}")
        
        # Clear current graph
        self.graph_builder.clear()
        
        # Create seed node
        seed_entity = GeneratedEntity.from_dict(entity_dict)
        seed_node = self.graph_builder.add_entity(seed_entity)
        
        # Add initial entity relationships
        if "relationships" in entity_dict:
            for rel in entity_dict["relationships"]:
                # Create relationship target entity
                target_entity = GeneratedEntity(
                    id=f"rel_{len(self.graph_builder.nodes)}",
                    label=rel["target"],
                    description=rel.get("description", f"Related to {seed_entity.label}"),
                    entity_type="related_entity",
                    domain=seed_entity.domain
                )
                target_node = self.graph_builder.add_entity(target_entity)
                
                # Add relationship
                self.graph_builder.add_relation(
                    source_id=seed_node.id,
                    target_id=target_node.id,
                    relation_type=rel["type"],
                    confidence=1.0,
                    properties={"description": rel.get("description", "")}
                )
        
        # Iteratively expand graph
        start_time = time.time()
        self._iterative_expand()
        end_time = time.time()

        # Save graph
        save_path = self._save_expanded_graph(entity_dict["id"], entity_dict["label"])
        
        stats = self.graph_builder.calculate_graph_stats()
        self.logger.info(f"Graph expansion completed: {stats['num_nodes']} nodes, {stats['num_edges']} edges, time taken: {end_time - start_time}")
        
        return save_path
    
    def expand_all_entities(self, entities_file: str) -> List[str]:
        """Expand all entities and save graphs
        
        Args:
            entities_file: Path to entities file
            
        Returns:
            List of saved graph file paths
        """
        self.logger.info(f"Starting batch entity expansion: {entities_file}")
        
        with open(entities_file, 'r', encoding='utf-8') as f:
            entities = json.load(f)
        
        saved_graphs = []
        for i, entity in enumerate(entities):
            try:
                self.logger.info(f"Processing entity {i+1}/{len(entities)}: {entity['label']}")
                save_path = self.expand_from_entity_dict(entity)
                saved_graphs.append(save_path)
                
            except Exception as e:
                self.logger.error(f"Failed to expand entity {entity.get('label', 'unknown')}: {e}")
                continue
        
        self.logger.info(f"Batch expansion completed, successfully generated {len(saved_graphs)} graphs")
        return saved_graphs
    
    def _iterative_expand(self):
        """Iteratively expand graph"""
        current_depth = 1
        
        while current_depth <= self.max_depth:
            self.logger.debug(f"Starting expansion at depth {current_depth}")
            
            # Get candidate nodes at current depth
            candidate_nodes = self._get_expansion_candidates(current_depth)
            
            if not candidate_nodes:
                self.logger.debug(f"No candidate nodes at depth {current_depth}, stopping expansion")
                break
            
            # Randomly select nodes for expansion (probabilistic selection)
            nodes_to_expand = self._probabilistic_node_selection(candidate_nodes)
            
            for node_id in nodes_to_expand:
                self._expand_from_node(node_id, current_depth)
            
            current_depth += 1
    
    def _get_expansion_candidates(self, depth: int) -> List[str]:
        """Get expansion candidate nodes
        
        Args:
            depth: Current depth
            
        Returns:
            List of candidate node IDs
        """
        candidates = []
        
        for node_id, node in self.graph_builder.nodes.items():
            # Only expand nodes with low degree to avoid super nodes
            node_degree = self.graph_builder.get_node_degree(node_id)
            if node_degree < self.max_branches_per_node:
                candidates.append(node_id)
        
        return candidates
    
    def _probabilistic_node_selection(self, candidates: List[str]) -> List[str]:
        """Probabilistic node selection
        
        Args:
            candidates: List of candidate nodes
            
        Returns:
            List of selected nodes
        """
        selected = []
        for node_id in candidates:
            if random.random() < self.expansion_probability:
                selected.append(node_id)
        
        # Ensure at least one node is selected
        if not selected and candidates:
            selected.append(random.choice(candidates))
        
        return selected
    
    def _expand_from_node(self, node_id: str, depth: int):
        """Expand from specified node
        
        Args:
            node_id: Node ID
            depth: Current depth
        """
        node = self.graph_builder.nodes[node_id]
        self.logger.debug(f"Expanding node: {node.label}")
        
        # Simulated web search
        search_results = self.web_search.search_entity_info(
            entity_name=node.label,
            entity_type=node.entity_type,
            max_results=self.max_search_results
        )

        if not search_results:
            self.logger.warning(f"Failed to search information for entity {node.label}")
            return
        
        # Combine search result texts
        combined_text = "\n\n".join([result.content for result in search_results])

        self.logger.info(f"Combined search result text: {combined_text}")
        
        # Extract triplets
        triples = self.triplet_extractor.extract_triples(
            text=combined_text,
            context_entity=node.label
        )
        
        # Process extracted triplets
        branch_count = 0
        for triple in triples:
            if branch_count >= self.max_branches_per_node:
                break
                
            if triple.confidence < self.min_confidence:
                continue
            
            # Check if subject matches current node
            if self._is_entity_match(triple.subject, node.label):
                # Create or find target node
                target_node_id = self._create_or_find_target_node(triple.object, triple.metadata['explanation'], depth + 1)
                
                if target_node_id and target_node_id != node_id:
                    # Add relationship
                    success = self.graph_builder.add_relation(
                        source_id=node_id,
                        target_id=target_node_id,
                        relation_type=triple.predicate,
                        confidence=triple.confidence,
                        properties={
                            "source_text": triple.source_text,
                            "extraction_method": "llm_triplet"
                        }
                    )
                    
                    if success:
                        branch_count += 1
    
    def _is_entity_match(self, extracted_entity: str, node_label: str) -> bool:
        """Check if extracted entity matches node label
        
        Args:
            extracted_entity: Extracted entity name
            node_label: Node label
            
        Returns:
            Whether they match
        """
        # Simple string matching, can be optimized to semantic matching later
        extracted_lower = extracted_entity.lower().strip()
        label_lower = node_label.lower().strip()
        
        return (extracted_lower == label_lower or 
                extracted_lower in label_lower or 
                label_lower in extracted_lower)
    
    def _create_or_find_target_node(self, entity_name: str, explanation: str, depth: int) -> Optional[str]:
        """Create or find target node
        
        Args:
            entity_name: Entity name
            depth: Node depth
            
        Returns:
            Node ID, returns None if failed
        """
        # First check if similar node already exists
        for node_id, node in self.graph_builder.nodes.items():
            if self._is_entity_match(entity_name, node.label):
                return node_id
        
        # Create new node
        try:
            new_entity = GeneratedEntity(
                id=f"expanded_{len(self.graph_builder.nodes)}",
                label=entity_name.strip(),
                description=explanation,
                entity_type="expanded_entity",
                domain="general",
                metadata={"expansion_depth": depth}
            )
            
            new_node = self.graph_builder.add_entity(new_entity)
            return new_node.id
            
        except Exception as e:
            self.logger.warning(f"Failed to create target node: {e}")
            return None
    
    def _save_expanded_graph(self, entity_id: str, entity_label: str) -> str:
        """Save expanded graph
        
        Args:
            entity_id: Seed entity ID
            entity_label: Seed entity label
            
        Returns:
            Saved file path
        """
        # Generate filename
        safe_label = "".join(c for c in entity_label if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_label = safe_label.replace(' ', '_')[:50]  # Limit length
        filename = f"{entity_id}_{safe_label}_expanded.json"
        filepath = Path(self.save_dir) / filename
        
        # Prepare save data
        graph_data = {
            "seed_entity": {
                "id": entity_id,
                "label": entity_label
            },
            "nodes": {nid: node.to_dict() for nid, node in self.graph_builder.nodes.items()},
            "edges": [edge.to_dict() for edge in self.graph_builder.edges],
            "statistics": self.graph_builder.calculate_graph_stats(),
            "metadata": {
                "expansion_config": {
                    "max_depth": self.max_depth,
                    "max_branches_per_node": self.max_branches_per_node,
                    "min_confidence": self.min_confidence,
                    "expansion_probability": self.expansion_probability
                },
                "created_at": str(self._get_current_timestamp())
            }
        }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        self.logger.debug(f"Graph has been saved to: {filepath}")
        return str(filepath)
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()


# Convenience function: batch expand entities
def expand_entities_from_file(entities_file: str, config: Config, save_dir: str) -> List[str]:
    """Convenience function: batch expand entities from file
    
    Args:
        entities_file: Path to entities file
        config: Configuration object
        save_dir: Save directory
    Returns:
        List of saved graph file paths
    """
    expander = GraphExpander(config, save_dir)
    return expander.expand_all_entities(entities_file) 