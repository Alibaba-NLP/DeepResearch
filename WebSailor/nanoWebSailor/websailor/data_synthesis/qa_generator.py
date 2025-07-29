#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA Generator - Generate Level 3 high-difficulty QA pairs based on WebSailor paper
Using information obfuscation techniques and complex entity coupling to generate high-uncertainty tasks
"""
import importlib
import json
import random
import uuid
import re
from typing import List, Dict, Any, Optional

from websailor.utils.config import Config
from websailor.utils.logger import get_logger
from websailor.data_synthesis.subgraph_sampler import SubgraphData


# InformationObfuscator类已移至query_obfuscation.py模块


class QAGenerator:
    """Generate Level 3 high-uncertainty QA pairs - Based on WebSailor paper"""
    
    def __init__(self, config: Config):
        """
        Initialize QA generator
        
        Args:
            config: Generator configuration parameters
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Model configuration
        self.generate_model = config.get('qa_generator.generate_model', 'claude3d7_thinking')
        self.api_key = config.get('entity_generator.api_key', '')
        
        # QA generation configuration
        self.obfuscation_rate = config.get('data_synthesis.qa_generator.obfuscation_rate', 0.5)  # Entity obfuscation ratio
                
        self.logger.info("Level 3 QA Generator initialized - Smart obfuscation enabled")
    
    def generate_qa_pairs(self, subgraph: SubgraphData, num_qa_pairs: int = 3) -> List[Dict[str, Any]]:
        """
        Generate Level 3 high-difficulty QA pairs based on subgraph
        
        Args:
            subgraph: Subgraph data
            num_qa_pairs: Number of QA pairs to generate
            
        Returns:
            List of Level 3 QA pairs
        """
        self.logger.info(f"Starting to generate {num_qa_pairs} Level 3 QA pairs for subgraph {subgraph.id}")
        
        qa_pairs = []
        unique_answer = []
        
        try:
            # 构建完整的子图信息（包含三元组和原始文本）
            subgraph_info = self._build_complete_subgraph_info(subgraph)
            
            # 生成高不确定性QA对
            for i in range(num_qa_pairs):
                qa_pair = self._generate_level3_qa(subgraph_info, subgraph)
                
                if qa_pair:
                    # 对问句进行模糊化处理
                    obfuscated_question = self._generate_obfuscated_question(
                        qa_pair.get('question', ''), subgraph_info
                    )
                    
                    # 防止出现相同问句
                    answers = [answer.lower() for answer in qa_pair.get('answer', [])]
                    if not any(ans in unique_answer for ans in answers):
                        unique_answer.extend(answers)
                    
                        qa_pair.update({
                            'id': str(uuid.uuid4()),
                            'subgraph_id': subgraph.id,
                            'difficulty_level': 3,
                            'original_question': qa_pair.get('question', ''),
                            'obfuscated_question': obfuscated_question,
                            'uncertainty_score': self._calculate_uncertainty_score(qa_pair),
                            'metadata': {
                                'topology_features': subgraph.topology_features
                            }
                        })

                        qa_pairs.append(qa_pair)
                    
        except Exception as e:
            self.logger.error(f"Failed to generate Level 3 QA pairs: {e}")
            
        self.logger.info(f"Successfully generated {len(qa_pairs)} Level 3 QA pairs")
        return qa_pairs
    
    def _build_complete_subgraph_info(self, subgraph: SubgraphData) -> Dict[str, Any]:
        """Build complete subgraph information, including triples and original text for each node"""
        
        # Collect all node information and original text
        nodes_info = []
        all_source_texts = set()
        
        for node_id, node in subgraph.nodes.items():
            node_info = {
                'id': node_id,
                'label': node.label,
                'type': node.entity_type,
                'domain': node.domain,
                'properties': node.properties,
                'features': node.features,
                'source_texts': []
            }
            
            # Collect node's original text
            if hasattr(node, 'features') and node.features:
                for feature_key, feature_value in node.features.items():
                    if 'source_text' in feature_key.lower() or 'text' in feature_key.lower():
                        if isinstance(feature_value, str) and feature_value.strip():
                            node_info['source_texts'].append(feature_value.strip())
                            all_source_texts.add(feature_value.strip())
            
            # Also collect text from properties
            if hasattr(node, 'properties') and node.properties:
                for prop_key, prop_value in node.properties.items():
                    if isinstance(prop_value, str) and len(prop_value) > 20:  # Might be text description
                        node_info['source_texts'].append(prop_value.strip())
                        all_source_texts.add(prop_value.strip())
            
            nodes_info.append(node_info)
        
        # Collect all edge/relationship information
        edges_info = []
        for edge in subgraph.edges:
            edge_info = {
                'source': edge.source,
                'target': edge.target,
                'relation': edge.relation_type,
                'properties': edge.properties if hasattr(edge, 'properties') else {},
                'source_text': ''
            }
            
            # Try to get edge's original text
            if hasattr(edge, 'properties') and edge.properties:
                source_text = edge.properties.get('source_text', '')
                if source_text:
                    edge_info['source_text'] = source_text
                    all_source_texts.add(source_text)
            
            edges_info.append(edge_info)
        
        # Generate triple representation
        triples = set()
        for edge in subgraph.edges:
            source_label = subgraph.nodes[edge.source].label if edge.source in subgraph.nodes else edge.source
            target_label = subgraph.nodes[edge.target].label if edge.target in subgraph.nodes else edge.target
            
            triple = f"({source_label}, {edge.relation_type}, {target_label})"
            triples.add(triple)
        
        # Deduplicate original text
        unique_source_texts = list(all_source_texts)
        
        return {
            'nodes': nodes_info,
            'edges': edges_info,
            'triples': triples,
            'source_texts': unique_source_texts,
            'statistics': {
                'num_nodes': len(nodes_info),
                'num_edges': len(edges_info),
                'num_source_texts': len(unique_source_texts)
            }
        }
    
    def _generate_level3_qa(self, subgraph_info: Dict[str, Any], subgraph: SubgraphData) -> Optional[Dict[str, Any]]:
        """Generate Level 3 complex reasoning questions - directly using complete subgraph information"""
        
        # Generate question
        prompt = self._build_qa_generation_prompt(subgraph_info)
        
        response = self._call_llm(prompt)
        qa_data = self._parse_qa_response(response)
        
        return qa_data
    
    def _build_qa_generation_prompt(self, subgraph_info: Dict[str, Any]) -> str:
        """Build QA generation prompt, directly using complete subgraph information"""
        
        # Build node information description, marking original text corresponding to each node
        nodes_desc = "Nodes infomation:\n"
        for i, node in enumerate(subgraph_info['nodes'], 1):
            nodes_desc += f"{i}. {node['label']} (ID: {node['id']}, type: {node['type']}, domain: {node['domain']})"
            
            # Display node properties
            if node['properties']:
                key_props = {k: v for k, v in node['properties'].items() 
                           if isinstance(v, (str, int, float)) and str(v).strip()}
                if key_props:
                    nodes_desc += f"   key properties: {key_props}\n"
            
            # Display original text corresponding to this node
            if node['source_texts']:
                nodes_desc += f"   original source text:\n"
                for j, text in enumerate(node['source_texts'], 1):
                    nodes_desc += f"     {j}. {text}\n"
            # nodes_desc += "\n"
        
        # Build relationship/triple information
        triples_desc = "knowledge triples:\n"
        for i, triple in enumerate(subgraph_info['triples'], 1):
            triples_desc += f"{i}. {triple}\n"
        
        # Build detailed edge information (including original text)
        edges_desc = "relationship detail information:\n"
        for i, edge in enumerate(subgraph_info['edges'], 1):
            source_node = next((n for n in subgraph_info['nodes'] if n['id'] == edge['source']), None)
            target_node = next((n for n in subgraph_info['nodes'] if n['id'] == edge['target']), None)
            
            source_label = source_node['label'] if source_node else edge['source']
            target_label = target_node['label'] if target_node else edge['target']
            
            edges_desc += f"{i}. {source_label} --[{edge['relation']}]--> {target_label}\n"
        
        # Build deduplicated original text list
        texts_desc = "All related source text:\n"
        for i, text in enumerate(set(subgraph_info['source_texts']), 1):
            texts_desc += f"{i}. {text}...\n\n"
        
        
        prompt = f"""
You are a professional Level 3 complex reasoning question generation expert. Please generate a high-quality question-answer pair based on the following complete knowledge graph subgraph information.

{nodes_desc}

{triples_desc}

{edges_desc}

{texts_desc}

Please generate a Level 3 complex reasoning question with the following requirements:
1. The question requires reasoning across multiple entities and relationships, but the question doesn't show the background of the knowledge graph
2. The answer cannot be directly obtained from a single text fragment and answer should be an entity which from the source text
3. You should give possible answers list. e.g: ["2018 01 01", "January 1, 2018"]
3. It requires synthesizing multiple sources of information
4. It should have a certain complexity of reasoning(**linear or non-linear reasoning path**)
5. Also generate a detailed search path explaining how to find the answer step by step
6. Please confitm the relation refer to the given source texts, the source texts is guarantee to be true since it is the extracted from wiki
7. If you think, it is impossible to generate the question, just return 'None'

Output format as JSON:
{{
    "question": "Complex reasoning question",
    "answer": ["concise entity answer", "entity answer's alias"], 
    "reasoning": "Explanation of the reasoning process",
    "reference": {{
        "search_path": [
            "Step 1: Search for what information, expect to find what",
            "Step 2: Based on the results of step 1, further search for what",
            ...
            "Last step: Synthesize information to draw conclusions"
        ],
        "nodes_used": ["List of used node IDs"],
        "triples_used": ["Used triples"],
        "source_texts_used": ["Portions of the original text fragments used"]
    }},
}}
"""
        
        return prompt
    
    def _generate_obfuscated_question(self, original_question: str, subgraph_info: Dict[str, Any]) -> str:
        """Generate obfuscated questions by randomly selecting entities for obfuscation"""
        if not original_question:
            return ""
        
        # Extract entities appearing in the question
        entities_in_question = []
        for node in subgraph_info['nodes']:
            if node['label'].lower() in original_question.lower():
                entities_in_question.append({
                    'label': node['label'],
                    'type': node['type'],
                    'id': node['id'],
                    'properties': node.get('properties', {})
                })
        
        if not entities_in_question:
            return original_question
        
        # Randomly select entities for obfuscation (according to configured ratio)
        num_to_obfuscate = max(1, int(len(entities_in_question) * self.obfuscation_rate))
        entities_to_obfuscate = random.sample(entities_in_question, 
                                            min(num_to_obfuscate, len(entities_in_question)))
        
        self.logger.debug(f"Selected entities for obfuscation: {[e['label'] for e in entities_to_obfuscate]}")
        
        # Generate obfuscation prompt
        entities_desc = []
        for entity in entities_to_obfuscate:
            desc = f"- {entity['label']}"
            if entity['properties']:
                key_props = [f"{k}: {v}" for k, v in list(entity['properties'].items())[:2]]
                if key_props:
                    desc += f" properties: {', '.join(key_props)}"
            entities_desc.append(desc)
        
        obfuscation_prompt = f"""
You are an expert at intelligently obfuscating questions to make them more challenging while keeping them answerable. Please obfuscate the specified entities in the following question using indirect descriptions.

# Original Question
{original_question}

# Entities to obfuscate
{chr(10).join(entities_desc)}

# Here are some examples of effective obfuscation techniques

**Example 1 - Temporal Obfuscation:**
Original: "What did Steve Jobs announce in 2007?"
Obfuscated: "What did the co-founder of Apple announce in the early 21st century that revolutionized mobile technology?"

**Example 2 - Geographic + Institutional Obfuscation:**
Original: "Which university in Boston was founded by John Harvard?"
Obfuscated: "Which prestigious institution in a major New England city was established by a clergyman whose name now graces the university?"

**Example 3 - Entity Name + Temporal Obfuscation:**
Original: "Who wrote 'Romeo and Juliet' in the 16th century?"
Obfuscated: "Who authored a famous tragedy about star-crossed lovers during the Elizabethan era?"

**Example 4 - Quantitative + Geographic Obfuscation:**
Original: "Which company has a 95% market share in Mountain View, California?"
Obfuscated: "Which tech giant, with an overwhelming market dominance, is headquartered in a city in the heart of Silicon Valley?"

**Example 5 - Complex Multi-entity Obfuscation:**
Original: "What algorithm did Larry Page and Sergey Brin develop at Stanford University?"
Obfuscated: "What revolutionary search algorithm was developed by two computer science researchers at a prestigious West Coast university in the late 1990s?"

# Obfuscation Guidelines:
1. Replace entity names with indirect descriptions, but never mention the actual names
2. Use temporal, geographic, relational, or characteristic clues to indirectly reference entities
3. Maintain the question's answerability - ensure it can still be solved through reasoning
4. Increase reasoning difficulty without making it incomprehensible
5. Keep semantic coherence and natural language flow
6. Prefer descriptive phrases over specific identifiers
7. Use vague time periods (early/mid/late + century/decade) instead of exact dates
8. Use descriptive locations instead of specific place names
9. Use role-based or achievement-based descriptions for people
10. Use approximate quantities or ranges instead of exact numbers

Please return only the obfuscated question, no other content.
"""
        
        try:
            response = self._call_llm(obfuscation_prompt)
            
            # Clean response
            obfuscated = response.strip()
            # Remove possible quotes or other formatting symbols
            obfuscated = obfuscated.strip('"\'`')
            
            if obfuscated and len(obfuscated) > 10:
                self.logger.debug(f"Successfully obfuscated question: {original_question[:50]}... -> {obfuscated[:50]}...")
                return obfuscated
            else:
                self.logger.warning(f"Poor quality obfuscation result, using original question")
                return original_question
                
        except Exception as e:
            self.logger.warning(f"Question obfuscation failed: {e}")
            return original_question
    
    def _calculate_uncertainty_score(self, qa_pair: Dict[str, Any]) -> float:
        """Calculate uncertainty score for the question"""
        score = 0.0
        
        question = qa_pair.get('question', '')
        
        # Vague term detection
        vague_terms = ['around', 'approximately', 'early', 'late', 'mid-', 'prominent', 'notable', 'distinguished', 'respected']
        vague_count = sum(1 for term in vague_terms if term in question.lower())
        score += vague_count * 0.1
        
        # Indirect reference detection
        indirect_phrases = ['who', 'that', 'which', 'where', 'an individual', 'someone', 'a person']
        indirect_count = sum(1 for phrase in indirect_phrases if phrase in question.lower())
        score += indirect_count * 0.15
        
        # Complexity indicators
        if len(qa_pair.get('reasoning_steps', [])) > 3:
            score += 0.3
        
        if qa_pair.get('reference', {}).get('search_path') and len(qa_pair['reference']['search_path']) > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _call_llm(self, prompt: str) -> str:
        """Call large language model for content filtering
        
        Args:
            prompt: Prompt text
            
        Returns:
            Model response content
        """
        try:
            llm_module = importlib.import_module(f"websailor.utils.models.{self.generate_model}")

            self.logger.info(f"relation determine prompt: {prompt}")

            llm_response = llm_module.generate(
                model=self.generate_model,
                system="You are a helpful assistant",
                user=prompt,
                api_key=self.api_key
            )
            self.logger.info(f"entity context filter llm_response: {llm_response}\n")
            return llm_response
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return None
    
    def _parse_qa_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse QA response from the large language model"""
        try:
            # Clean response text
            if "```json" in response:
                json_pattern = re.compile(r'```json\n(.*?)\n```', re.DOTALL)
                json_match = json_pattern.search(response)
                if json_match:
                    response = json_match.group(1).strip()
            
            # Parse JSON
            qa_data = json.loads(response)
            
            # Validate required fields
            required_fields = ['question', 'answer']
            for field in required_fields:
                if field not in qa_data or not qa_data[field]:
                    self.logger.warning(f"QA response missing required field: {field}")
                    return None
            
            # Ensure reference field exists and contains search_path
            if 'reference' not in qa_data:
                qa_data['reference'] = {}
            if 'search_path' not in qa_data['reference']:
                qa_data['reference']['search_path'] = ["Search steps not explicitly specified"]
            
            return qa_data
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse QA response JSON: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to process QA response: {e}")
            return None
        
    def obfuscate_information(self, subgraph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simplified information obfuscation - now mainly handled at question level
        
        Args:
            subgraph: Input subgraph
            
        Returns:
            Return subgraph as is (obfuscation handled during question generation)
        """
        return subgraph
    
    def generate_batch_qa_pairs(self, subgraphs: List[SubgraphData], 
                               qa_pairs_per_subgraph: int = 3) -> List[Dict[str, Any]]:
        """
        Batch generate Level 3 QA pairs
        
        Args:
            subgraphs: List of subgraphs
            qa_pairs_per_subgraph: Number of QA pairs to generate per subgraph
            
        Returns:
            List of all Level 3 QA pairs
        """
        all_qa_pairs = []
        
        for i, subgraph in enumerate(subgraphs):
            self.logger.info(f"Processing subgraph {i+1}/{len(subgraphs)}: {subgraph.id}")
            
            qa_pairs = self.generate_qa_pairs(subgraph, qa_pairs_per_subgraph)
            all_qa_pairs.extend(qa_pairs)
        
        self.logger.info(f"Batch generation completed, generated {len(all_qa_pairs)} Level 3 QA pairs in total")
        return all_qa_pairs