import os
import logging
from backend.app.core.memory import InMemoryStore

# Setup core logging configurations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("student-performance-api")

# Establish unified system paths
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "behavioral_ridge_v1.joblib")

# Global instances initialized empty, to be populated during application startup lifecycle hooks
model_pipeline = None

# =====================================================================
# PLUGGABLE STORAGE LAYER ROUTER
# =====================================================================
# Right now, we instantiate the InMemoryStore machine. During the Redis 
# integration phase, you will swap this to `RedisMemoryStore()` right here. 
# Because your routes only call `.get()` and `.set()`, the rest of your app stays 100% untouched!
memory_store = InMemoryStore()