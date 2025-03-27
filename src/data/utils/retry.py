import time
from functools import wraps
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)

def retry_with_backoff(
    retries: int = 3,
    backoff_in_seconds: int = 1,
    max_backoff: int = 32,
    exceptions: tuple = (Exception,)
):
    """Retry decorator with exponential backoff."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_backoff = backoff_in_seconds
            last_exception = None
            
            for retry in range(retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(
                        f"Attempt {retry + 1}/{retries} failed: {str(e)}. "
                        f"Retrying in {retry_backoff} seconds..."
                    )
                    time.sleep(retry_backoff)
                    retry_backoff = min(retry_backoff * 2, max_backoff)
            
            logger.error(f"All {retries} attempts failed. Last error: {str(last_exception)}")
            raise last_exception
            
        return wrapper
    return decorator