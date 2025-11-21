"""
Caching utilities with Redis support for API rate limiting
Streamlit-compatible caching with fallback to in-memory cache
"""

import os
import json
import hashlib
import time
from typing import Any, Optional, Callable
from functools import wraps

try:
    import streamlit as st
except ImportError:
    st = None

# Try to import Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# In-memory cache fallback
_memory_cache = {}
_cache_timestamps = {}


def get_cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments"""
    key_str = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def get_redis_client():
    """Get Redis client if available"""
    if not REDIS_AVAILABLE:
        return None
    
    try:
        # Try to get Redis URL from environment or Streamlit secrets
        redis_url = os.getenv("REDIS_URL")
        if not redis_url and st:
            try:
                redis_url = st.secrets.get("REDIS_URL", None)
            except:
                pass
        if redis_url:
            return redis.from_url(redis_url)
        
        # Try local Redis
        return redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    except Exception:
        return None


def cached_with_retry(ttl: int = 3600, max_retries: int = 3, retry_delay: float = 1.0):
    """
    Decorator for caching function results with retry logic
    
    Args:
        ttl: Time to live in seconds
        max_retries: Maximum number of retries on failure
        retry_delay: Delay between retries in seconds
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{get_cache_key(*args, **kwargs)}"
            redis_client = get_redis_client()
            
            # Try to get from cache
            if redis_client:
                try:
                    cached = redis_client.get(cache_key)
                    if cached:
                        return json.loads(cached)
                except Exception:
                    pass
            
            # Fallback to memory cache
            if cache_key in _memory_cache:
                cached_time = _cache_timestamps.get(cache_key, 0)
                if time.time() - cached_time < ttl:
                    return _memory_cache[cache_key]
            
            # Execute with retry logic
            last_exception = None
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    
                    # Store in cache
                    if redis_client:
                        try:
                            redis_client.setex(cache_key, ttl, json.dumps(result, default=str))
                        except Exception:
                            pass
                    
                    _memory_cache[cache_key] = result
                    _cache_timestamps[cache_key] = time.time()
                    
                    return result
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        raise last_exception
            
            raise last_exception
        
        return wrapper
    return decorator


def clear_cache(pattern: Optional[str] = None):
    """Clear cache entries"""
    redis_client = get_redis_client()
    
    if redis_client:
        try:
            if pattern:
                keys = redis_client.keys(pattern)
                if keys:
                    redis_client.delete(*keys)
            else:
                redis_client.flushdb()
        except Exception:
            pass
    
    # Clear memory cache
    if pattern:
        keys_to_delete = [k for k in _memory_cache.keys() if pattern in k]
        for k in keys_to_delete:
            _memory_cache.pop(k, None)
            _cache_timestamps.pop(k, None)
    else:
        _memory_cache.clear()
        _cache_timestamps.clear()

