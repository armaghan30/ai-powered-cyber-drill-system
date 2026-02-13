from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAgent(ABC):
   

    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Given an observation
        return an action dict."""
        raise NotImplementedError
