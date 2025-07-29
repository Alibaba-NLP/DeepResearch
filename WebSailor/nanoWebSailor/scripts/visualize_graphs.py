#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Graph Visualization Script

This script is used to visualize knowledge graphs in the expanded_graphs directory,
using Plotly to generate interactive visualization results.

Author: Evan Zuo
"""

import json
import os
from pathlib import Path
import argparse
import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import random


def load_graph_data(file_path: str) -> Dict:
    """Load graph data
    
    Args:
        file_path: Graph data file path
        
    Returns:
        Graph data dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_networkx_graph(graph_data: Dict) -> nx.Graph:
    """Create NetworkX graph object
    
    Args:
        graph_data: Graph data dictionary
        
    Returns:
        NetworkX graph object
    """
    G = nx.Graph()
    
    # 添加节点
    for node_id, node_data in graph_data['nodes'].items():
        G.add_node(
            node_id,
            label=node_data['label'],
            entity_type=node_data['entity_type'],
            description=node_data.get('description', '')
        )
    
    # 添加边
    for edge in graph_data['edges']:
        G.add_edge(
            edge['source'],
            edge['target'],
            relation_type=edge['relation_type'],
            confidence=edge['confidence']
        )
    
    return G

def get_node_positions(G: nx.Graph) -> Dict:
    """Get node layout positions
    
    Args:
        G: NetworkX graph object
        
    Returns:
        Node position dictionary
    """
    # 使用force-directed布局
    return nx.spring_layout(G, k=1/pow(len(G.nodes), 0.3), iterations=50)

def create_plotly_figure(G: nx.Graph, pos: Dict, seed_entity_id: str) -> go.Figure:
    """Create Plotly figure
    
    Args:
        G: NetworkX graph object
        pos: Node position dictionary
        seed_entity_id: Seed entity ID
        
    Returns:
        Plotly figure object
    """
    # Prepare node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    # Assign different colors for different entity types
    color_map = {
        'seed_entity': '#FF6B6B',  # 红色
        'expanded_entity': '#4ECDC4',  # 青色
        'related_entity': '#45B7D1',  # 蓝色
        'general': '#96CEB4'  # 绿色
    }
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Prepare node hover text
        node_data = G.nodes[node]
        hover_text = f"Label: {node_data['label']}<br>"
        hover_text += f"Type: {node_data['entity_type']}<br>"
        if node_data['description']:
            hover_text += f"Description: {node_data['description']}"
        node_text.append(hover_text)
        
        # Set node color and size
        if node == seed_entity_id:
            node_color.append('#FFD700')  # Gold
            node_size.append(20)  # Larger size
        else:
            node_color.append(color_map.get(node_data['entity_type'], '#96CEB4'))
            node_size.append(10)  # Normal size
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Add edge path
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Prepare edge hover text
        edge_data = G.edges[edge]
        edge_text.append(
            f"Relation: {edge_data['relation_type']}<br>"
            f"Confidence: {edge_data['confidence']:.2f}"
        )
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        text=edge_text,
        mode='lines'
    )
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line_width=2,
            line=dict(color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Knowledge Graph Visualization',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="Seed entity shown in gold",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    font=dict(size=10)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    return fig

def visualize_graph(input_path: str, output_dir: str = 'data/visualization_results'):
    """Visualize knowledge graph
    
    Args:
        input_path: Input file or directory path
        output_dir: Output directory path
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    def process_single_file(file_path: str):
        print(f"Processing: {file_path}")
        
        # Load data and create graph
        graph_data = load_graph_data(file_path)
        try:
            seed_entity_id = graph_data['seed_entity']['id']
        except:
            seed_entity_id = graph_data['nodes'][
                str(list(graph_data['nodes'].keys())[0])
            ]["id"]
        G = create_networkx_graph(graph_data)
        pos = get_node_positions(G)
        fig = create_plotly_figure(G, pos, seed_entity_id)
        
        # Generate output file path
        output_name = Path(file_path).stem + '_viz.html'
        output_path = os.path.join(output_dir, output_name)
        
        # Save figure
        fig.write_html(output_path)
        print(f"Visualization saved to: {output_path}")
        
        # Print graph statistics
        print("\nGraph Statistics:")
        print(f"Number of nodes: {len(G.nodes)}")
        print(f"Number of edges: {len(G.edges)}")
        print(f"Average degree: {sum(dict(G.degree()).values()) / len(G.nodes):.2f}")
    
    # Process input path
    if os.path.isfile(input_path):
        process_single_file(input_path)
    else:
        # 处理目录中的所有json文件
        for file_name in os.listdir(input_path):
            if file_name.endswith('.json'):
                file_path = os.path.join(input_path, file_name)
                process_single_file(file_path)

def main():
    parser = argparse.ArgumentParser(description='Knowledge Graph Visualization Tool')
    parser.add_argument(
        '--input_path',
        required=True,
        help='Path to the input graph file or directory containing graph files'
    )
    parser.add_argument(
        '--output_dir',
        default='data/visualization_results',
        help='Directory to save visualization results'
    )
    
    args = parser.parse_args()
    visualize_graph(args.input_path, args.output_dir)

if __name__ == '__main__':
    main() 