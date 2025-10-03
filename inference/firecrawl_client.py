import os
from functools import lru_cache
from firecrawl import Firecrawl

class FirecrawlConfigurationError(RuntimeError):
    """Raised when Firecrawl is not installed or misconfigured."""

@lru_cache(maxsize=1)
def get_firecrawl_client():

    api_key = os.getenv("FIRECRAWL_API_KEY")

    if not api_key:
        raise FirecrawlConfigurationError("FIRECRAWL_API_KEY environment variable is not set")

    return Firecrawl(api_key=api_key)

