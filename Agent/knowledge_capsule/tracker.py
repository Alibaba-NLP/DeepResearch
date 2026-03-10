"""Research Knowledge Tracker"""
from enum import Enum
from typing import Dict, List
import time
import uuid

class Phase(Enum):
    SPROUT = "sprout"
    GREEN_LEAF = "green_leaf"
    YELLOW_LEAF = "yellow_leaf"  
    RED_LEAF = "red_leaf"
    SOIL = "soil"

class ResearchKnowledge:
    def __init__(self, content: str, source: str, priority="P2"):
        self.id = str(uuid.uuid4())[:8]
        self.content = content
        self.source = source
        self.priority = priority
        self.confidence = 0.7
        self.phase = Phase.SPROUT
        self.created = time.time()
        self.accessed = time.time()
    
    def boost(self):
        self.confidence = min(1.0, self.confidence + 0.03)
        self.accessed = time.time()
        if self.confidence >= 0.8:
            self.phase = Phase.GREEN_LEAF

class ResearchTracker:
    def __init__(self):
        self.knowledge: Dict[str, ResearchKnowledge] = {}
    
    def add(self, content: str, source: str, priority="P2") -> ResearchKnowledge:
        k = ResearchKnowledge(content, source, priority)
        self.knowledge[k.id] = k
        return k
    
    def access(self, kid: str) -> bool:
        if kid not in self.knowledge: return False
        self.knowledge[kid].boost()
        return True
    
    def get_high_conf(self, thresh=0.8) -> List[ResearchKnowledge]:
        return [k for k in self.knowledge.values() if k.confidence >= thresh]
    
    def decay(self):
        for k in self.knowledge.values():
            d = {'P0': 0, 'P1': 0.004, 'P2': 0.008}.get(k.priority, 0.008)
            k.confidence = max(0, k.confidence - d)
