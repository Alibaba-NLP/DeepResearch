#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triple Extractor Module

This module uses large language models to extract knowledge triples (subject-predicate-object) from unstructured text,
for building and expanding knowledge graphs. Supports extraction and validation of multiple relationship types.

Main Classes:
- Triple: Triple data class
- TripleExtractor: Triple extractor

Features:
- Knowledge extraction using large language models
- Support for multiple relationship type recognition
- Triple validation and filtering
- Batch text processing
- Confidence assessment

Author: Evan Zuo
Date: January 2025
"""

import importlib
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from openai import OpenAI

from ..utils.config import Config
from ..utils.logger import get_logger


@dataclass
class Triple:
    """Triple data class"""
    subject: str
    predicate: str
    object: str
    confidence: float = 0.8
    source_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'subject': self.subject,
            'predicate': self.predicate,
            'object': self.object,
            'confidence': self.confidence,
            'source_text': self.source_text,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Triple':
        """Create triple from dictionary"""
        return cls(**data)
    
    def __str__(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.object})"


class TripleExtractor:
    """Triple Extractor
    
    Extract knowledge triples from text using large language models.
    """
    
    def __init__(self, config: Config):
        """Initialize extractor
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Model configuration
        self.model_name = config.get('triplet_extractor.model_name', 'gpt-4o')
        self.max_tokens = config.get('triplet_extractor.max_tokens', 8192)
        self.temperature = config.get('triplet_extractor.temperature', 0.3)
        self.api_key = config.get('entity_generator.api_key', '')
                
        # Extraction configuration
        self.min_confidence = config.get('triplet_extractor.min_confidence', 0.6)
        self.max_triples_per_text = config.get('triplet_extractor.max_triples_per_text', 20)
        
        # Relationship type mapping
        self.relation_types = {
            'is_a': ['is a', 'is an', 'is', 'are', 'was a', 'were'],
            'located_in': ['located in', 'situated in', 'based in', 'found in'],
            'part_of': ['part of', 'belongs to', 'member of', 'component of'],
            'created_by': ['created by', 'founded by', 'established by', 'made by'],
            'worked_at': ['worked at', 'employed by', 'affiliated with'],
            'collaborated_with': ['collaborated with', 'worked with', 'partnered with'],
            'succeeded_by': ['succeeded by', 'followed by', 'replaced by'],
            'influenced_by': ['influenced by', 'inspired by', 'based on'],
            'related_to': ['related to', 'connected to', 'associated with'],
            'has_property': ['has', 'contains', 'includes', 'features']
        }
        
        self.logger.info("Triple extractor initialization completed")
    
    def extract_triples(self, text: str, context_entity: Optional[str] = None) -> List[Triple]:
        """Extract triples from text
        
        Args:
            text: Input text
            context_entity: Context entity (optional, used to guide extraction)
            
        Returns:
            List of extracted triples
        """
        self.logger.debug(f"Starting to extract triples, text length: {len(text)}")
        
        try:
            # Build extraction prompt
            prompt = self._build_extraction_prompt(text, context_entity)
            
            # Call large language model
            response = self._call_llm(prompt)
            
            # Parse response
            triples = self._parse_llm_response(response, text)
            
            # Validate and filter
            valid_triples = self._validate_triples(triples)
            
            self.logger.debug(f"Successfully extracted {len(valid_triples)} valid triples: {valid_triples}")
            return valid_triples
            
        except Exception as e:
            self.logger.error(f"Triple extraction failed: {e}")
            return []
        
    def extract_triples_batch(self, texts: List[str], 
                            context_entities: Optional[List[str]] = None) -> List[List[Triple]]:
        """Batch extract triples
        
        Args:
            texts: List of texts
            context_entities: List of context entities (optional)
            
        Returns:
            List of triples corresponding to each text
        """
        self.logger.info(f"Starting batch triple extraction, number of texts: {len(texts)}")
        
        results = []
        for i, text in enumerate(texts):
            context_entity = context_entities[i] if context_entities and i < len(context_entities) else None
            triples = self.extract_triples(text, context_entity)
            results.append(triples)
        
        total_triples = sum(len(triples) for triples in results)
        self.logger.info(f"Batch extraction completed, extracted {total_triples} triples in total")
        return results
    
    def _build_extraction_prompt(self, text: str, context_entity: Optional[str] = None) -> str:
        """Build triple extraction prompt
        
        Args:
            text: Input text
            context_entity: Context entity
            
        Returns:
            Built prompt
        """
        context_instruction = ""
        if context_entity:
            context_instruction = f"\nPay special attention to relationships involving '{context_entity}'."
        
        prompt = f"""
As an expert knowledge extraction system, please extract knowledge triples (subject-predicate-object) from the following text.

INSTRUCTIONS:
1. Extract triples that represent factual relationships between entities
2. Focus on meaningful, verifiable relationships
3. Use clear, normalized predicate names (e.g., "is_a", "located_in", "created_by")
4. Ensure subjects and objects are specific entities or concepts
5. Avoid extracting overly general or obvious relationships
6. Each triple should be independent and meaningful{context_instruction}

RELATIONSHIP TYPES EXAMPLES:
- is_a: Entity type relationships
- located_in: Geographic or containment relationships  
- part_of: Component or membership relationships
- created_by: Creation or authorship relationships
- worked_at: Employment or affiliation relationships
- collaborated_with: Partnership or cooperation relationships
- succeeded_by: Succession or replacement relationships
- influenced_by: Influence or inspiration relationships
- related_to: General association relationships
- has_property: Attribute or characteristic relationships
... and so on, you can use other relationship types as long as it is reasonable

TEXT TO ANALYZE:
{text}

Please return the results in the following JSON format:
{{
    "triples": [
        {{
            "subject": "Entity or concept name",
            "predicate": "relationship_type",
            "object": "Related entity or concept",
            "confidence": 0.0-1.0,
            "explanation": "Brief explanation of why this relationship is valid"
        }},
        ...
    ]
}}

IMPORTANT:
- Return valid JSON only
- Include only high-confidence relationships (>0.6)
- Limit to maximum {self.max_triples_per_text} triples, extract only important triples, if there are no important {self.max_triples_per_text} triples, this is ok
- Use consistent entity naming (proper nouns, no articles)
- Ensure predicates match the relationship types listed above
"""
        return prompt.strip()
    
    def _call_llm(self, prompt: str) -> str:
        """Call large language model for content filtering
        
        Args:
            prompt: Prompt text
            
        Returns:
            Model response content
        """
        try:
            llm_module = importlib.import_module(f"websailor.utils.models.{self.model_name}")
            llm_response = llm_module.generate(
                api_key=self.api_key,
                model=self.model_name,
                system="You are an expert knowledge extraction system.",
                user=prompt
            )
            self.logger.info(f"entity context filter llm_response: {llm_response}")
            return llm_response
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return None
        
    def _parse_llm_response(self, response: str, source_text: str) -> List[Triple]:
        """Parse LLM response
        
        Args:
            response: Model response
            source_text: Source text
            
        Returns:
            List of parsed triples
        """
        triples = []
        
        try:
            # Try to parse JSON
            if response.startswith('```json'):
                response = response.replace('```json', '').replace('```', '').strip()
            elif response.startswith('```'):
                response = response.replace('```', '').strip()
            
            data = json.loads(response)
            
            if 'triples' in data:
                for triple_data in data['triples']:
                    if self._is_valid_triple_data(triple_data):
                        triple = Triple(
                            subject=triple_data['subject'].strip(),
                            predicate=self._normalize_predicate(triple_data['predicate'].strip()),
                            object=triple_data['object'].strip(),
                            confidence=float(triple_data.get('confidence', 0.8)),
                            source_text=source_text,
                            metadata={
                                'explanation': triple_data.get('explanation', ''),
                                'raw_predicate': triple_data['predicate'].strip()
                            }
                        )
                        triples.append(triple)
                        
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parsing failed, trying regex parsing: {e}")
            triples = self._parse_with_regex(response, source_text)
        except Exception as e:
            self.logger.error(f"Response parsing failed: {e}")
        
        return triples
    
    def _parse_with_regex(self, response: str, source_text: str) -> List[Triple]:
        """Parse response using regex (fallback method)
        
        Args:
            response: Model response
            source_text: Source text
            
        Returns:
            List of parsed triples
        """
        triples = []
        
        # Try to match triple patterns
        patterns = [
            r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)',  # (subject, predicate, object)
            r'([^-]+)\s*-\s*([^-]+)\s*-\s*(.+)',   # subject - predicate - object
            r'(\w+(?:\s+\w+)*)\s+(\w+(?:\s+\w+)*)\s+(\w+(?:\s+\w+)*)'  # subject predicate object
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.MULTILINE)
            for match in matches:
                if len(match) == 3:
                    subject, predicate, obj = [item.strip() for item in match]
                    if subject and predicate and obj:
                        triple = Triple(
                            subject=subject,
                            predicate=self._normalize_predicate(predicate),
                            object=obj,
                            confidence=0.7,  # Default confidence for regex parsing
                            source_text=source_text[:200] + "..." if len(source_text) > 200 else source_text,
                            metadata={'parsed_with': 'regex'}
                        )
                        triples.append(triple)
            
            if triples:  # If triples are found, don't try other patterns
                break
        
        return triples
    
    def _normalize_predicate(self, predicate: str) -> str:
        """Normalize predicate
        
        Args:
            predicate: Original predicate
            
        Returns:
            Normalized predicate
        """
        predicate_lower = predicate.lower().strip()
        
        # Direct match with standard predicates
        for standard_predicate, variations in self.relation_types.items():
            if predicate_lower in [v.lower() for v in variations]:
                return standard_predicate
            if predicate_lower == standard_predicate:
                return standard_predicate
        
        # Partial match
        for standard_predicate, variations in self.relation_types.items():
            for variation in variations:
                if variation.lower() in predicate_lower or predicate_lower in variation.lower():
                    return standard_predicate
        
        # If no match, keep original form but clean it
        normalized = re.sub(r'[^a-zA-Z0-9_\s]', '', predicate_lower)
        normalized = re.sub(r'\s+', '_', normalized.strip())
        return normalized if normalized else 'related_to'
    
    def _is_valid_triple_data(self, triple_data: Dict[str, Any]) -> bool:
        """Validate triple data
        
        Args:
            triple_data: Triple data dictionary
            
        Returns:
            Whether valid
        """
        required_fields = ['subject', 'predicate', 'object']
        
        # Check required fields
        for field in required_fields:
            if field not in triple_data or not triple_data[field]:
                return False
        
        # Check field lengths
        if (len(triple_data['subject'].strip()) < 2 or 
            len(triple_data['predicate'].strip()) < 2 or 
            len(triple_data['object'].strip()) < 2):
            return False
        
        # Check confidence
        if 'confidence' in triple_data:
            try:
                confidence = float(triple_data['confidence'])
                if confidence < 0 or confidence > 1:
                    return False
            except (ValueError, TypeError):
                return False
        
        return True
    
    def _validate_triples(self, triples: List[Triple]) -> List[Triple]:
        """Validate and filter triples
        
        Args:
            triples: Original list of triples
            
        Returns:
            List of validated triples
        """
        valid_triples = []
        seen_triples = set()
        
        for triple in triples:
            # Confidence filtering
            if triple.confidence < self.min_confidence:
                continue
            
            # Deduplication (based on subject-predicate-object)
            triple_key = (triple.subject.lower(), triple.predicate.lower(), triple.object.lower())
            if triple_key in seen_triples:
                continue
            
            # Length validation
            if (len(triple.subject) < 2 or len(triple.predicate) < 2 or len(triple.object) < 2):
                continue
            
            # Avoid circular relationships (same subject and object)
            if triple.subject.lower() == triple.object.lower():
                continue
            
            # Avoid too generic relationships
            if self._is_too_generic(triple):
                continue
            
            seen_triples.add(triple_key)
            valid_triples.append(triple)
        
        return valid_triples
    
    def _is_too_generic(self, triple: Triple) -> bool:
        """Check if relationship is too generic
        
        Args:
            triple: Triple
            
        Returns:
            Whether too generic
        """
        generic_subjects = {
            'it', 'this', 'that', 'they', 'he', 'she', 'we', 'you', 'i', 
            'one', 'someone', 'anyone', 'everyone', 'nobody', 'somebody', 
            'anybody', 'everybody', 'people', 'person', 'ones', 'who', 
            'whoever', 'these', 'those', 'there'
        }
        
        generic_objects = {
            'something', 'anything', 'everything', 'nothing', 'things', 'stuff',
            'item', 'items', 'object', 'objects', 'entity', 'entities', 
            'element', 'elements', 'matter', 'substance', 'material', 
            'piece', 'pieces', 'part', 'parts', 'thing', 'ones', 'what', 
            'whatever', 'way', 'ways'
        }
        
        if (triple.subject.lower() in generic_subjects or 
            triple.object.lower() in generic_objects):
            return True
        
        # Check if contains meaningless words
        meaningless_words = {
            'various', 'different', 'many', 'some', 'several', 'other',
            'certain', 'particular', 'specific', 'general', 'common',
            'usual', 'typical', 'standard', 'normal', 'regular',
            'ordinary', 'numerous', 'countless', 'multiple', 'diverse',
            'assorted', 'miscellaneous', 'few', 'couple', 'handful',
            'plenty', 'lot', 'lots', 'variety', 'range', 'selection',
            'any', 'all', 'most', 'much', 'more', 'less', 'extra',
            'certain', 'such', 'kind', 'kinds', 'type', 'types', 'sort', 'sorts'
        }
        
        if (any(word in triple.subject.lower() for word in meaningless_words) or
            any(word in triple.object.lower() for word in meaningless_words)):
            return True
        
        return False

    
    def get_relation_stats(self, triples: List[Triple]) -> Dict[str, int]:
        """Get relationship type statistics
        
        Args:
            triples: List of triples
            
        Returns:
            Dictionary of relationship type statistics
        """
        stats = {}
        for triple in triples:
            predicate = triple.predicate
            stats[predicate] = stats.get(predicate, 0) + 1
        
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True)) 