from abc import ABC, abstractmethod
from typing import List, Dict, Optional

# =====================================================================
# 1. THE ABSTRACT INTERFACE (The Contract)
# =====================================================================
# This acts as a strict blueprint. Any database class we build in the 
# future MUST implement these exact methods, or Python will refuse to boot.
class BaseMemoryStore(ABC):
    
    @abstractmethod
    def get(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieve the entire conversational thread history for a unique session."""
        pass

    @abstractmethod
    def set(self, session_id: str, message: Dict[str, str]) -> None:
        """Append a single message frame to the session thread history."""
        pass

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Purge a context thread from operational memory."""
        pass

# =====================================================================
# 2. IN-MEMORY IMPLEMENTATION (The MVP Variant)
# =====================================================================
# Our current runtime storage array. It implements the base contract strictly.
class InMemoryStore(BaseMemoryStore):
    def __init__(self) -> None:
        # Protected internal dictionary tracking live session arrays
        self._store: Dict[str, List[Dict[str, str]]] = {}

    def get(self, session_id: str) -> List[Dict[str, str]]:
        # Return an existing history thread, or initialize a new sequence seamlessly
        if session_id not in self._store:
            self._store[session_id] = [
                {"role": "system", "content": "You are the Early Warning Platform Assistant advising on student risk indicators."}
            ]
        return self._store[session_id]

    def set(self, session_id: str, message: Dict[str, str]) -> None:
        # Trigger explicit history population if session is initialized out of order
        if session_id not in self._store:
            self.get(session_id)
        self._store[session_id].append(message)

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)