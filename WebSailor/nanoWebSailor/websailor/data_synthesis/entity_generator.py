#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entity Generator Module

This module uses large language models to generate rare entities through prompts, replacing the original Wikidata SPARQL query method.
Generated entities are saved to local files for subsequent graph construction.

Main Classes:
- EntityGenerator: Entity generator using large language models to generate rare entities
- GeneratedEntity: Generated entity data class

Features:
- Generate rare entities using large language models
- Support multiple entity types and domains
- Local storage of entity information
- Batch generation and caching mechanism
- Entity validation and filtering
- Entity deduplication and similarity checking

Author: Evan Zuo
Date: January 2025
"""

import json
import os
import uuid
import random
import hashlib
from pathlib import Path
from openai import OpenAI
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from websailor.utils.config import Config
from websailor.utils.logger import get_logger


@dataclass
class GeneratedEntity:
    """Generated entity data class
    
    Encapsulates entity information generated through large language models, including basic attributes and feature data.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: str = ""
    description: str = ""
    entity_type: str = "entity"
    domain: str = "general"
    properties: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    rarity_score: float = 0.5
    generation_prompt: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    relationships: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if not self.label:
            self.label = f"Entity_{self.id[:8]}"
        
        # Ensure property value types are correct
        self.rarity_score = float(self.rarity_score)
        if not 0 <= self.rarity_score <= 1:
            self.rarity_score = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'id': self.id,
            'label': self.label,
            'description': self.description,
            'entity_type': self.entity_type,
            'domain': self.domain,
            'properties': self.properties,
            'features': self.features,
            'rarity_score': self.rarity_score,
            'generation_prompt': self.generation_prompt,
            'created_at': self.created_at,
            'relationships': self.relationships,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeneratedEntity':
        """Create entity from dictionary"""
        return cls(**data)
    
    def get_signature(self) -> str:
        """Get entity signature for deduplication
        
        Returns:
            Entity signature (hash based on label)
        """
        content = f"{self.label}".encode('utf-8')
        return hashlib.md5(content).hexdigest()


class EntityGenerator:
    """Entity Generator
    
    Generate rare entities using large language models through prompts
    """
    
    def __init__(self, config: Config):
        """Initialize generator
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Model configuration
        self.model_name = config.get('entity_generator.model_name', 'gpt-4o')
        self.max_tokens = config.get('entity_generator.max_tokens', 4096)
        self.temperature = config.get('entity_generator.temperature', 0.7)
        
        # Storage configuration
        self.storage_path = Path(config.get('entity_generator.storage_path', 'data/generated_entities'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Entity type and domain configuration
        if config.get('entity_generator.entity_types', None) is not None:
            self.entity_types = config.get('entity_generator.entity_types', None)
        else:
            self.entity_types = [
                # People and organizations
                "person", "organization", "institution", "company", "team",
                # Locations and facilities
                "location", "city", "country", "landmark", "building", "infrastructure",
                # Culture and arts
                "artwork", "book", "movie", "music", "performance", "cultural_heritage",
                # Science and technology
                "invention", "discovery", "technology", "scientific_theory", "algorithm",
                # Nature and environment
                "species", "ecosystem", "natural_phenomenon", "geological_formation",
                # Concepts and ideas
                "concept", "theory", "philosophy", "methodology", "movement",
                # Events and history
                "event", "historical_period", "revolution", "disaster", "achievement",
                # Products and items
                "product", "tool", "device", "artifact", "material",
                # Society and systems
                "law", "policy", "system", "tradition", "social_movement",
                # Disciplines and fields
                "field", "discipline", "research_area", "industry", "profession"
            ]
        
        if config.get('entity_generator.domains', None) is not None:
            self.domains = config.get('entity_generator.domains', None)
        else:
            self.domains = [
                # Science and technology
                "computer_science", "physics", "chemistry", "biology", "mathematics",
                "engineering", "aerospace", "robotics", "artificial_intelligence",
                # Humanities and social sciences
                "history", "archaeology", "anthropology", "sociology", "psychology",
                "philosophy", "linguistics", "literature", "religious_studies",
                # Arts and culture
                "fine_arts", "music", "theater", "film", "architecture",
                "design", "fashion", "photography", "digital_art",
                # Business and economics
                "business", "economics", "finance", "entrepreneurship", "marketing",
                "management", "industry", "trade", "market_research",
                # Medicine and health
                "medicine", "healthcare", "public_health", "nutrition", "biotechnology",
                "pharmaceuticals", "medical_research", "mental_health",
                # Environment and earth sciences
                "environmental_science", "ecology", "geology", "oceanography",
                "climate_science", "astronomy", "space_exploration",
                # Politics and law
                "politics", "law", "international_relations", "public_policy",
                "human_rights", "governance", "diplomacy",
                # Education and research
                "education", "academic_research", "pedagogy", "educational_technology",
                "skill_development", "professional_training",
                # Sports and athletics
                "sports", "athletics", "recreation", "fitness", "competitive_gaming",
                # Media and communication
                "media", "journalism", "communication", "social_media", "broadcasting",
                "digital_communication", "content_creation"
            ]
        
        # Entity validation configuration
        self.min_description_length = config.get('entity_generator.min_description_length', 50)
        self.min_properties = config.get('entity_generator.min_properties', 3)
        
        # Cache
        self._entity_cache: Dict[str, GeneratedEntity] = {}
        self._entity_signatures: Set[str] = set()
        self._load_existing_entities()
        
        self.logger.info("Entity generator initialization completed")

    def generate_rare_entities(self, 
                             num_entities: int = 10, 
                             entity_types: Optional[List[str]] = None,
                             domains: Optional[List[str]] = None,
                             batch_size: int = 5) -> List[GeneratedEntity]:
        """Generate rare entities
        
        Args:
            num_entities: Number of entities to generate
            entity_types: List of entity types, if None use all types
            domains: List of domains, if None use all domains
            batch_size: Batch generation size
            
        Returns:
            List of generated entities
        """
        self.logger.info(f"Starting entity generation - Target count: {num_entities}, Batch size: {batch_size}")
        self.logger.info(f"Entity types: {entity_types or 'all'}")
        self.logger.info(f"Domains: {domains or 'all'}")

        if entity_types is None:
            entity_types = self.entity_types
        if domains is None:
            domains = self.domains
        
        generated_entities = []
        attempts = 0
        max_attempts = num_entities * 3  # Maximum number of attempts
        
        while len(generated_entities) < num_entities and attempts < max_attempts:
            # Calculate current batch size
            current_batch_size = min(batch_size, num_entities - len(generated_entities))
            self.logger.debug(f"Starting generation of batch {attempts//batch_size + 1}, batch size: {current_batch_size}")
            
            # Generate entities in batch
            batch_entities = self._generate_entity_batch(
                current_batch_size, entity_types, domains
            )
            
            # Validate and filter entities
            valid_entities = self._validate_and_filter_entities(batch_entities)
            self.logger.debug(f"Batch generation results - Total: {len(batch_entities)}, Valid: {len(valid_entities)}")
            
            # Add valid entities
            for entity in valid_entities:
                if len(generated_entities) < num_entities:
                    generated_entities.append(entity)
                    self._entity_cache[entity.id] = entity
                    self._entity_signatures.add(entity.get_signature())
                    self.logger.debug(f"Added new entity: {entity.label} (ID: {entity.id})")
            
            attempts += current_batch_size
            self.logger.info(f"Current progress: {len(generated_entities)}/{num_entities} (Attempts: {attempts})")
        
        # Save to file
        self._save_entities(generated_entities)
        
        success_rate = len(generated_entities) / attempts * 100 if attempts > 0 else 0
        self.logger.info(f"Entity generation completed - Success: {len(generated_entities)}, Attempts: {attempts}, Success rate: {success_rate:.1f}%")
        return generated_entities
    
    def _generate_entity_batch(self, 
                             batch_size: int,
                             entity_types: List[str],
                             domains: List[str]) -> List[GeneratedEntity]:
        """Generate entities in batch using thread pool
        
        Args:
            batch_size: Batch size
            entity_types: List of entity types
            domains: List of domains
            
        Returns:
            List of generated entities
        """
        self.logger.debug(f"Starting batch entity generation, batch size: {batch_size}")
        entities = []
        
        # Create task list
        tasks = [
            (random.choice(entity_types), random.choice(domains))
            for _ in range(batch_size)
        ]
        
        # Use thread pool for concurrent generation
        with ThreadPoolExecutor(max_workers=min(batch_size, 10)) as executor:
            self.logger.debug(f"Creating thread pool, max threads: {min(batch_size, 10)}")
            # Submit all tasks
            future_to_task = {
                executor.submit(self._generate_single_entity, entity_type, domain): (entity_type, domain)
                for entity_type, domain in tasks
            }
            
            # Collect results
            for future in as_completed(future_to_task):
                entity_type, domain = future_to_task[future]
                try:
                    entity = future.result()
                    if entity:
                        entities.append(entity)
                        self.logger.debug(f"Successfully generated entity: {entity.label} ({entity_type}/{domain})")
                    else:
                        self.logger.warning(f"Failed to generate entity: {entity_type}/{domain}")
                except Exception as e:
                    self.logger.error(f"Entity generation exception: {entity_type}/{domain} - {e}")
        
        self.logger.debug(f"Batch generation completed, success count: {len(entities)}/{batch_size}")
        return entities
    
    def _generate_single_entity(self, entity_type: str, domain: str) -> Optional[GeneratedEntity]:
        """Generate a single entity
        
        Args:
            entity_type: Entity type
            domain: Domain
            
        Returns:
            Generated entity, or None if failed
        """
        try:
            # Build generation prompt
            prompt = self._build_generation_prompt(entity_type, domain)
            
            # Call large language model to generate entity information
            response = self._call_llm(prompt)
            
            # Parse response
            entity_data = self._parse_llm_response(response, entity_type, domain)
            
            if entity_data:
                entity_data['generation_prompt'] = prompt
                entity = GeneratedEntity(**entity_data)
                return entity
            else:
                self.logger.warning(f"Failed to parse LLM response: {response}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to generate entity: {e}")
            return None
    
    def _build_generation_prompt(self, entity_type: str, domain: str) -> str:
        """Build generation prompt
        
        Args:
            entity_type: Entity type
            domain: Domain
            
        Returns:
            Generated prompt
        """
        prompt = f"""
As a professional rare entity generator, please generate a rare {entity_type} entity in the field of {domain}.

Requirements:
1. The entity must be real but relatively rare or little-known.
2. The entity should have unique and interesting characteristics.
3. Provide detailed attribute and relationship information.
4. Ensure the information is accurate and verifiable.
5. The entity should be found in wiki data.
6. The entity should be in english.

The generated content should include:
1. Basic information (name, description)
2. Key properties (at least 3)
3. Unique features (at least 2)
4. Potential relationships with other entities
5. A rarity score (between 0 and 1; the higher the score, the rarer the entity)

Please return the output in valid JSON format with the following structure:
{{
    "label": "Entity name",
    "description": "Detailed description (at least 50 words)",
    "properties": {{
        "Property1": "Value1",
        "Property2": "Value2",
        ...
    }},
    "features": {{
        "Feature1": "Description1",
        "Feature2": "Description2",
        ...
    }},
    "relationships": [
        {{
            "type": "Relationship type",
            "target": "Related entity",
            "description": "Description of the relationship"
        }},
        ...
    ],
    "rarity_score": 0.0-1.0
}}

Make sure the response is valid JSON.
"""
        return prompt.strip()
    
    def _call_llm(self, prompt: str) -> str:
        """Call large language model
        
        Args:
            prompt: Input prompt
            
        Returns:
            Model response
        """
        try:
            self.logger.debug(f"Preparing to call LLM API - Model: {self.model_name}")
            # Initialize OpenAI client
            client = OpenAI(
                api_key=self.config.get('entity_generator.api_key', ""),
                base_url=self.config.get('entity_generator.base_url', 'https://api.openai.com/v1')
            )
            
            # Call API
            self.logger.debug("Sending API request...")
            response = client.chat.completions.create(
                model=self.model_name,  # Use model specified in config
                messages=[
                    {"role": "system", "content": "You are a professional rare entity generator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Return generated content
            content = response.choices[0].message.content
            self.logger.debug("Successfully received API response")
            return content
            
        except Exception as e:
            self.logger.error(f"Failed to call LLM API: {str(e)}")
            return ""
        
    def _parse_llm_response(self, response: str, 
                        entity_type: str, domain: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response
        
        Args:
            response: LLM response
            entity_type: Entity type
            domain: Domain
            
        Returns:
            Parsed entity data, or None if failed
        """
        try:
            # Parse JSON
            data = json.loads(response)
            
            # Add type and domain information
            data['entity_type'] = entity_type
            data['domain'] = domain
            
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Response parsing failed: {e}")
            return None
    
    def _validate_and_filter_entities(self, entities: List[GeneratedEntity]) -> List[GeneratedEntity]:
        """Validate and filter entities
        
        Args:
            entities: List of entities
            
        Returns:
            List of validated entities
        """
        self.logger.debug(f"Starting entity validation and filtering, input count: {len(entities)}")
        valid_entities = []
        
        for entity in entities:
            # Check for duplicates
            if entity.get_signature() in self._entity_signatures:
                self.logger.debug(f"Entity duplicate, skipped: {entity.label}")
                continue
            
            # Validate description length
            if len(entity.description) < self.min_description_length:
                self.logger.debug(f"Description too short, skipped: {entity.label} ({len(entity.description)} < {self.min_description_length})")
                continue
            
            # Validate property count
            if len(entity.properties) < self.min_properties:
                self.logger.debug(f"Insufficient properties, skipped: {entity.label} ({len(entity.properties)} < {self.min_properties})")
                continue
            
            # Validate required fields
            if not all([entity.label, entity.description, entity.properties, entity.features]):
                self.logger.debug(f"Missing required fields, skipped: {entity.label}")
                continue
            
            valid_entities.append(entity)
            self.logger.debug(f"Entity validation passed: {entity.label}")
        
        self.logger.debug(f"Validation completed - Passed: {len(valid_entities)}, Failed: {len(entities) - len(valid_entities)}")
        return valid_entities
    
    def _load_existing_entities(self) -> None:
        """Load existing entities"""
        try:
            entities_file = self.storage_path / "entities.json"
            if not entities_file.exists():
                self.logger.info("Entity file does not exist, will create new file")
                entities_file.parent.mkdir(parents=True, exist_ok=True)
                return
            
            try:
                with open(entities_file, 'r', encoding='utf-8') as f:
                    entities_data = json.load(f)
                    for entity_data in entities_data:
                        entity = GeneratedEntity.from_dict(entity_data)
                        self._entity_cache[entity.id] = entity
                        self._entity_signatures.add(entity.get_signature())
                
                self.logger.info(f"Loaded {len(self._entity_cache)} entities")
                
            except Exception as e:
                self.logger.error(f"Failed to load entity file: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to load entities: {e}")
    
    def _save_entities(self, entities: List[GeneratedEntity]) -> None:
        """Save entities to file
        
        Args:
            entities: List of entities
        """
        try:
            self.logger.info(f"Starting to save entities to file, count: {len(entities)}")
            entities_file = self.storage_path / "entities.json"
            
            # Read existing data
            existing_entities = []
            if entities_file.exists():
                try:
                    with open(entities_file, 'r', encoding='utf-8') as f:
                        existing_entities = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to read existing entity file, will create new file: {e}")
            
            # Add new entities to existing data
            existing_entity_ids = {e.get('id') for e in existing_entities}
            for entity in entities:
                if entity.id not in existing_entity_ids:
                    existing_entities.append(entity.to_dict())
                    self.logger.debug(f"Added new entity: {entity.label} (ID: {entity.id})")
            
            # Save all data
            with open(entities_file, 'w', encoding='utf-8') as f:
                json.dump(existing_entities, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Entity saving completed, total count: {len(existing_entities)}")
            
        except Exception as e:
            self.logger.error(f"Failed to save entities: {str(e)}")

    def get_entity(self, entity_id: str) -> Optional[GeneratedEntity]:
        """Get entity by ID
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Entity object, or None if not found
        """
        return self._entity_cache.get(entity_id)
    
    def list_entities(self, 
                     entity_type: Optional[str] = None,
                     domain: Optional[str] = None) -> List[GeneratedEntity]:
        """List entities matching criteria
        
        Args:
            entity_type: Entity type filter
            domain: Domain filter
            
        Returns:
            List of entities
        """
        entities = list(self._entity_cache.values())
        
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]
        if domain:
            entities = [e for e in entities if e.domain == domain]
        
        return entities
    
    def clear_cache(self) -> None:
        """Clear entity cache"""
        self._entity_cache.clear()
        self._entity_signatures.clear() 


if __name__ == "__main__":
    config = Config()
    # Load default configuration file
    config_path = Path(__file__).parent.parent.parent / "configs" / "default_config.yaml"
    config.load_from_file(config_path)
    # Print configuration
    print(config.to_dict())
    entity_generator = EntityGenerator(config)
    entity_generator.generate_rare_entities(num_entities=200, batch_size=5)