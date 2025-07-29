#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mock Web Search Module

This module uses large language models to simulate web search functionality, generating realistic search results for graph expansion.
Through LLM, it generates rich content related to entities, including potential relationship information.

Main features:
- Generate search results using large language models
- Support customized content for different entity types
- Provide rich text for triple extraction

Author: Evan Zuo
Date: January 2025
"""

import requests
import random
import importlib
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from tenacity import RetryError, retry, stop_after_attempt, wait_fixed

from websailor.utils.logger import get_logger
from websailor.utils.config import Config


MAX_SEARCH_ATTEMPTS = 3
WAIT_TIME = 10


@dataclass
class MockSearchResult:
    """Mock search result data class"""
    title: str
    content: str
    source_type: str = "wiki"
    relevance_score: float = 0.8
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MockWebSearch:
    """
    Use wiki-18 Retrieval for information retrieval
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize mock searcher
        
        Args:
            config: Configuration object, if None use default configuration
        """
        self.logger = get_logger(__name__)
        
        if config is None:
            config = Config()

        self.max_search_results = config.get("graph_expander.max_search_results", 3)
        self.wiki_retrieval_url = config.get("mock_web_search.wiki_retrieval_url", "http://10.200.64.10/10-flash-e2e-agent/retrieve")
        self.filter_model = config.get("mock_web_search.filter_model", "gpt-4o")
        self.retrieval_method = config.get("mock_web_search.retrieval_method", "wiki-18")
        self.api_key = config.get("entity_generator.api_key", "")
        self.max_tokens = config.get("entity_generator.max_tokens", 8192)
        self.temperature = config.get("entity_generator.temperature", 0.3)

    def build_filter_prompt(self, retrieval: Dict, context_entity: Optional[str] = None) -> str:
        """Build prompt for LLM to filter whether wiki Retrieval documents are related to the entity

        Args:
            retrievals: Single document retrieved from wiki Retrieval interface
            context_entity: Context entity

        Returns:
            Built prompt        
        """
        entity_name = context_entity if context_entity else "No provided entity"
        content = retrieval.get("content", "")
        
        prompt = f"""Please determine if the following text content is **explicitly** related to the entity "{entity_name}".
If it is related, please return "RELATED" and briefly explain how the text is connected to this entity.
If it is not related, please return "NOT RELATED".

Text content:
{content}

Response format:
Reason: [brief explanation]
Relevance: [RELATED/NOT RELATED]
"""
        return prompt

    def search_entity_info(self, entity_name: str, entity_type: str = "entity", 
                          max_results: int = 5) -> List[MockSearchResult]:
        """Search for entity-related information
        
        Args:
            entity_name: Entity name
            entity_type: Entity type
            max_results: Maximum number of results
            
        Returns:
            List of search results
        """
        self.logger.debug(f"Mock searching entity information: {entity_name} (type: {entity_type})")
        
        results = []
        try:
            if self.retrieval_method == "wiki-18":
                results = self._wiki_retrieval(entity_name, entity_type, max_results)
            elif self.retrieval_method == "llm":
                results = self._llm_retrieval(entity_name, entity_type, max_results)
            else:
                raise ValueError(f"Unsupported retrieval method: {self.retrieval_method}")
        except RetryError as e:
            self.logger.warning(f"Search data failed after {MAX_SEARCH_ATTEMPTS} retries: {e}")
        
        self.logger.debug(f"Generated {len(results)} search results")
        return results
    
    @retry(stop=stop_after_attempt(MAX_SEARCH_ATTEMPTS), wait=wait_fixed(WAIT_TIME))
    def _llm_retrieval(self, entity_name: str, entity_type: str, max_results: int = 5) -> List[MockSearchResult]:
        """Use LLM's internal knowledge base for retrieval
        
        Args:
            entity_name: Entity name
            entity_type: Entity type
            max_results: Maximum number of results

        Returns:
            List of mock search results
        """
        self.logger.debug(f"Using LLM to retrieve entity information: {entity_name} (type: {entity_type})")
        
        # Build retrieval prompt
        prompt = self._build_retrieval_prompt(entity_name, entity_type, max_results)
        
        self.logger.info(f"llm_retrieval prompt: {prompt}")

        try:
            # Call LLM to generate content
            llm_response = self._call_llm(prompt)
            
            if not llm_response:
                self.logger.warning(f"LLM did not return valid content for entity: {entity_name}")
                return []
            
            # Parse LLM response and format results
            results = self._parse_llm_response(llm_response, entity_name, entity_type)
            
            # Limit number of results
            results = results[:max_results]
            
            self.logger.debug(f"LLM retrieval generated {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.warning(f"LLM retrieval failed, retrying: {e}")
            raise
    
    def _build_retrieval_prompt(self, entity_name: str, entity_type: str, max_results: int) -> str:
        """Build prompt for LLM retrieval
        
        Args:
            entity_name: Entity name
            entity_type: Entity type
            max_results: Maximum number of results
            
        Returns:
            Built prompt string
        """
        prompt = f"""Please generate {max_results} pieces of relevant information for the entity "{entity_name}" (type: {entity_type}).

Requirements:
1. Each piece of content should contain rich factual information that helps build a knowledge graph
2. The content should cover different aspects of the entity, such as definition, properties, relationships, historical background, etc.
3. Ensure the information is accurate, specific, and valuable
4. Each piece of content should be between 200-500 words

Please return in the following JSON format:
{{
    "results": [
        {{
            "title": "Content Title 1",
            "content": "Detailed Content Description 1",
            "relevance_score": 0.9
        }},
        {{
            "title": "Content Title 2", 
            "content": "Detailed Content Description 2",
            "relevance_score": 0.8
        }}
    ]
}}

Entity: {entity_name}
Type: {entity_type}"""
        
        return prompt
    
    def _parse_llm_response(self, llm_response: str, entity_name: str, entity_type: str) -> List[MockSearchResult]:
        """Parse LLM response and convert to list of MockSearchResult
        
        Args:
            llm_response: LLM response text
            entity_name: Entity name
            entity_type: Entity type
            
        Returns:
            List of MockSearchResult
        """
        results = []
        
        try:
            # Try to parse JSON format response
            if llm_response.strip().startswith('{'):
                import json
                response_data = json.loads(llm_response)
                
                if "results" in response_data:
                    for i, result_data in enumerate(response_data["results"]):
                        title = result_data.get("title", f"{entity_name} related information {i+1}")
                        content = result_data.get("content", "")
                        relevance_score = result_data.get("relevance_score", 0.8)
                        
                        if content.strip():  # Ensure content is not empty
                            results.append(
                                MockSearchResult(
                                    title=title,
                                    content=content,
                                    source_type="llm_knowledge",
                                    relevance_score=float(relevance_score),
                                    metadata={
                                        'entity_type': entity_type,
                                        'generated_by': 'llm_retrieval',
                                        'entity_name': entity_name
                                    }
                                )
                            )
            else:
                # If not JSON format, try to extract information from text
                self.logger.debug("LLM response is not JSON format, trying text parsing")
                results = self._parse_text_response(llm_response, entity_name, entity_type)
                
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parsing failed, trying text parsing: {e}")
            results = self._parse_text_response(llm_response, entity_name, entity_type)
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            results = self._create_fallback_result(llm_response, entity_name, entity_type)
        
        return results
    
    def _parse_text_response(self, text_response: str, entity_name: str, entity_type: str) -> List[MockSearchResult]:
        """Parse text format LLM response
        
        Args:
            text_response: Text response
            entity_name: Entity name
            entity_type: Entity type
            
        Returns:
            List of MockSearchResult
        """
        results = []
        
        # Split text by paragraphs
        paragraphs = [p.strip() for p in text_response.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs[:self.max_search_results]):
            if len(paragraph) > 50:  # Ensure content has sufficient length
                title = f"{entity_name} - Information Segment {i+1}"
                
                # If paragraph is long, try to extract title from the beginning
                if len(paragraph) > 200:
                    lines = paragraph.split('\n')
                    if lines[0] and len(lines[0]) < 100:
                        title = lines[0].strip()
                        content = '\n'.join(lines[1:]).strip()
                    else:
                        content = paragraph
                else:
                    content = paragraph
                
                results.append(
                    MockSearchResult(
                        title=title,
                        content=content,
                        source_type="llm_knowledge", 
                        relevance_score=0.8 - (i * 0.1),  # Decreasing relevance score
                        metadata={
                            'entity_type': entity_type,
                            'generated_by': 'llm_retrieval_text_parse',
                            'entity_name': entity_name
                        }
                    )
                )
        
        return results
    
    def _create_fallback_result(self, llm_response: str, entity_name: str, entity_type: str) -> List[MockSearchResult]:
        """Create fallback search result
        
        Args:
            llm_response: LLM response
            entity_name: Entity name  
            entity_type: Entity type
            
        Returns:
            List containing a single fallback result
        """
        return [
            MockSearchResult(
                title=f"{entity_name} Basic Information",
                content=llm_response[:1000] if llm_response else f"Basic information about {entity_name}",
                source_type="llm_knowledge",
                relevance_score=0.6,
                metadata={
                    'entity_type': entity_type,
                    'generated_by': 'llm_retrieval_fallback',
                    'entity_name': entity_name
                }
            )
        ]
    
    
    @retry(stop=stop_after_attempt(MAX_SEARCH_ATTEMPTS), wait=wait_fixed(WAIT_TIME))
    def _wiki_retrieval(self, entity_name: str, entity_type: str, top_k: int) -> List[MockSearchResult]:
        """Use large language model to create search results
        
        Args:
            entity_name: Entity name
            entity_type: Entity type
            top_k: Top k search results
            
        Returns:
            List of mock search results
        """
        url = self.wiki_retrieval_url
        headers = {"Content-Type": "application/json;charset=utf-8"}

        try:
            # Configure request body
            payload = {
                "queries": [entity_name],
                "topk": top_k,
                "return_scores": True
            }

            self.logger.debug(f"Tool call: {json.dumps(payload, ensure_ascii=False)}")

            # Send POST request
            response = requests.post(url, json=payload, headers=headers, timeout=120)

            # Handle exception requests
            if response.status_code != 200:
                self.logger.error(f"Request failed!\nRequest status code: {response.status_code}\nResponse data:\n{response.text}")
                raise Exception(f"Request failed, status code: {response.status_code}")

            response_data = response.json()
            
            # Parse response
            result_data_list = self.format_results(response_data)
            
            if not result_data_list:
                return []
            
            results = []
            for result_data in result_data_list:
                # Filter wiki documents through large language model RAG
                prompt = self.build_filter_prompt(result_data, entity_name)
                filtered_response = self._call_llm(prompt)
                
                # Check if the large model's response contains "RELATED"
                if "RELATED" in filtered_response and "NOT RELATED" not in filtered_response:
                    results.append(
                        MockSearchResult(
                            title=result_data.get('title', f"{entity_name} related information"),
                            content=result_data.get('content', ''),
                            source_type="wiki_retrieval",
                            relevance_score=result_data.get('score', 0.8),
                            metadata={
                                'top_k': top_k, 
                                'entity_type': entity_type,
                                'generated_by': 'wiki_retrieval_api',
                                'doc_id': result_data.get('id', '')
                            }
                        )
                    )
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Large model generation failed, retrying: {e}")
            raise

    def _call_llm(self, prompt: str) -> str:
        """Call large language model for content filtering
        
        Args:
            prompt: Prompt text
            
        Returns:
            Large language model response content
        """
        try:
            llm_module = importlib.import_module(f"websailor.utils.models.{self.filter_model}")

            llm_response = llm_module.generate(
                api_key=self.api_key,
                model=self.filter_model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are a helper assistant ",
                user=prompt
            )
            self.logger.info(f"entity context filter llm_response: {llm_response}\n")
            return llm_response
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return None

    def format_results(self, results) -> List[Dict]:
        """
        Format the retrieved documents into a readable string
        
        Args:
            results: Dictionary containing retrieval results
            
        Returns:
            List[Dict]: Formatted retrieval result list
        """
        output = []
        
        # Check result format
        if not isinstance(results, dict) or "result" not in results:
            self.logger.error(f"Retrieval result format error: {results}")
            return []
            
        result_list = results["result"]
        
        for i, documents in enumerate(result_list):
            for j, doc in enumerate(documents):
                if isinstance(doc, dict):
                    doc_res = doc.get("document", {})
                    score = doc.get("score", 0.0)

                    id = doc_res.get("id", f"unknown-{i}-{j}")
                    content = doc_res.get("contents", "")
                    title = doc_res.get("title", None)

                    if title is None or not title:
                        title = content.split("\n")[0][:50] + "..." if len(content) > 50 else content

                    data_info_item = {
                        "id": id,
                        "content": content,
                        "title": title,
                        "score": score
                    }
                    
                    output.append(data_info_item)
        
        return output
