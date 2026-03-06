#!/usr/bin/env python3
"""
Cache Management System for APGI Framework
========================================

Efficient caching system for storing and retrieving computed results,
preprocessed data, and intermediate calculations.
"""

import functools
import hashlib
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from rich.console import Console

console = Console()


class CacheManager:
    """Advanced caching system for APGI framework."""

    def __init__(self, cache_dir: Union[str, Path] = "cache", max_size_mb: int = 1024):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024

        # Cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()

        # Thread lock for thread safety
        self._lock = threading.Lock()

        # Statistics
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "total_requests": 0}

    def _load_metadata(self) -> Dict:
        """Load cache metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
                print(f"Error loading cache metadata: {type(e).__name__}: {e}")
                return {}
        return {}

    def _save_metadata(self):
        """Save cache metadata to file."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except (json.JSONDecodeError, FileNotFoundError, PermissionError, OSError) as e:
            print(f"Error saving cache metadata: {type(e).__name__}: {e}")

    def _is_json_serializable(self, obj: Any) -> bool:
        """Check if object is JSON serializable."""
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False

    def _save_json(self, path: Path, data: Any):
        """Save data as JSON if possible."""
        with open(path, "w") as f:
            json.dump(data, f, separators=(",", ":"), sort_keys=True)

    def _load_json(self, path: Path) -> Any:
        """Load data from JSON file."""
        with open(path, "r") as f:
            return json.load(f)

    def _generate_key(self, key_data: Any) -> str:
        """Generate unique cache key from data."""
        if isinstance(key_data, str):
            key_string = key_data
        elif isinstance(key_data, dict):
            # Sort dictionary for consistent key generation
            key_string = json.dumps(key_data, sort_keys=True, default=str)
        elif isinstance(key_data, (list, tuple)):
            key_string = str(key_data)
        else:
            key_string = str(key_data)

        # Generate hash
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{key}.cache"

    def _get_cache_info(self, key: str) -> Dict:
        """Get cache information for key."""
        cache_path = self._get_cache_path(key)

        if cache_path.exists():
            stat = cache_path.stat()
            return {
                "key": key,
                "path": str(cache_path),
                "size": stat.st_size,
                "created": stat.st_ctime,
                "accessed": stat.st_atime,
                "modified": stat.st_mtime,
            }
        return {}

    def _update_access_time(self, key: str):
        """Update access time for cache entry."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.touch()

            # Update metadata
            if key in self.metadata:
                self.metadata[key]["accessed"] = time.time()
                self._save_metadata()

    def _evict_if_needed(self):
        """Evict cache entries if size limit exceeded."""
        total_size = sum(info.get("size", 0) for info in self.metadata.values())

        if total_size > self.max_size_bytes:
            # Sort by least recently used
            entries = sorted(
                self.metadata.items(), key=lambda x: x[1].get("accessed", 0)
            )

            # Evict entries until under limit
            for key, info in entries:
                if total_size <= self.max_size_bytes * 0.8:  # Leave 20% headroom
                    break

                cache_path = Path(info["path"])
                if cache_path.exists():
                    cache_path.unlink()
                    total_size -= info.get("size", 0)
                    del self.metadata[key]
                    self.stats["evictions"] += 1

            self._save_metadata()

    def get(self, key: Any, default: Any = None) -> Any:
        """Get value from cache."""
        with self._lock:
            self.stats["total_requests"] += 1

            cache_key = self._generate_key(key)
            cache_path = self._get_cache_path(cache_key)
            json_path = cache_path.with_suffix(".json")

            # Check if entry exists and is not expired
            if cache_key in self.metadata:
                entry = self.metadata[cache_key]
                if "expires" in entry and entry["expires"] < time.time():
                    # Entry expired, remove it
                    if cache_path.exists():
                        cache_path.unlink()
                    if json_path.exists():
                        json_path.unlink()
                    del self.metadata[cache_key]
                    self._save_metadata()
                    self.stats["misses"] += 1
                    return default

            if json_path.exists() or cache_path.exists():
                try:
                    # Try JSON first for better performance
                    json_path = cache_path.with_suffix(".json")
                    if json_path.exists():
                        data = self._load_json(json_path)
                    else:
                        # Fall back to joblib for complex objects
                        with open(cache_path, "rb") as f:
                            data = joblib.load(f)

                    self._update_access_time(cache_key)
                    self.stats["hits"] += 1
                    return data

                except (
                    joblib.JoblibException,
                    FileNotFoundError,
                    PermissionError,
                    OSError,
                    EOFError,
                ) as e:
                    print(
                        f"Error loading cache entry {cache_key}: {type(e).__name__}: {e}"
                    )
                    # Remove corrupted entry
                    cache_path.unlink(missing_ok=True)
                    json_path.unlink(missing_ok=True)
                    if cache_key in self.metadata:
                        del self.metadata[cache_key]
                        self._save_metadata()

            self.stats["misses"] += 1
            return default

    def set(self, key: Any, value: Any, ttl: Optional[int] = None):
        """Set value in cache with optional TTL (seconds)."""
        with self._lock:
            cache_key = self._generate_key(key)
            cache_path = self._get_cache_path(cache_key)

            try:
                # Use JSON for serializable data (faster than pickle)
                if self._is_json_serializable(value):
                    json_path = cache_path.with_suffix(".json")
                    self._save_json(json_path, value)
                    file_size = json_path.stat().st_size
                else:
                    # Fall back to joblib for complex objects
                    with open(cache_path, "wb") as f:
                        joblib.dump(value, f, compress=3)
                    file_size = cache_path.stat().st_size

                # Update metadata
                current_time = time.time()
                self.metadata[cache_key] = {
                    "path": str(cache_path),
                    "size": file_size,
                    "created": current_time,
                    "accessed": current_time,
                    "modified": current_time,
                }

                if ttl:
                    self.metadata[cache_key]["expires"] = current_time + ttl

                self._save_metadata()
                self._evict_if_needed()

            except (
                joblib.JoblibException,
                FileNotFoundError,
                PermissionError,
                OSError,
            ) as e:
                print(f"Error saving cache entry {cache_key}: {type(e).__name__}: {e}")

    def warm_cache(self, data_sources: List[Union[str, Path]], max_workers: int = 4):
        """Warm up cache with common data sources."""
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        console.print(
            f"[blue]Warming up cache with {len(data_sources)} data sources...[/blue]"
        )
        start_time = time.time()

        def warm_single_source(source_path):
            try:
                source_path = Path(source_path)
                if not source_path.exists():
                    return f"Source not found: {source_path}"

                # Generate cache key for this source
                file_mtime = source_path.stat().st_mtime
                cache_key = {
                    "type": "preprocessed_data",
                    "file_path": str(source_path),
                    "config": {"normalize": True, "filter": True},
                    "file_mtime": file_mtime,
                }

                # Check if already cached
                existing_data = self.get(cache_key)
                if existing_data is not None:
                    return f"Already cached: {source_path.name}"

                # Load and preprocess data to warm cache
                from data_validation import DataPreprocessor

                preprocessor = DataPreprocessor()
                df = preprocessor.load_data(source_path)

                # Apply preprocessing to warm up common operations
                df_processed = preprocessor.clean_missing_data(df)
                df_processed = preprocessor.remove_outliers(df_processed)
                df_processed = preprocessor.normalize_data(df_processed)

                # Cache the processed data
                self.set(cache_key, df_processed, ttl=3600)  # 1 hour TTL
                return f"Warmed: {source_path.name} ({df_processed.shape})"

            except (
                FileNotFoundError,
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
                ValueError,
                TypeError,
                PermissionError,
                OSError,
            ) as e:
                return f"Error warming {source_path}: {type(e).__name__}: {e}"

        # Warm up sources in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(warm_single_source, source) for source in data_sources
            ]

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                if "Error" in result:
                    console.print(f"[red]{result}[/red]")
                else:
                    console.print(f"[green]{result}[/green]")
                console.print(f"[blue]Progress: {completed}/{len(data_sources)}[/blue]")

        elapsed_time = time.time() - start_time
        console.print(
            f"[green]✓[/green] Cache warming completed in {elapsed_time:.1f}s"
        )

        # Update stats
        self.stats["cache_warms"] = self.stats.get("cache_warms", 0) + 1
        self._save_metadata()

    def get_cache_warm_suggestions(self) -> List[str]:
        """Get suggestions for cache warming based on common data files."""
        suggestions = []
        data_dir = Path("data")

        if data_dir.exists():
            # Look for common data files
            for pattern in ["*.csv", "*.json", "*.h5", "*.hdf5"]:
                files = list(data_dir.glob(pattern))
                suggestions.extend([str(f) for f in files[:5]])  # Limit to 5 per type

        return suggestions

    def delete(self, key: Any) -> bool:
        """Delete cache entry."""
        with self._lock:
            cache_key = self._generate_key(key)
            cache_path = self._get_cache_path(cache_key)
            json_path = cache_path.with_suffix(".json")

            deleted = False
            if cache_path.exists():
                cache_path.unlink()
                deleted = True
            if json_path.exists():
                json_path.unlink()
                deleted = True

            if deleted and cache_key in self.metadata:
                del self.metadata[cache_key]
                self._save_metadata()
            return deleted

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            # Clear both JSON and joblib cache files
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            for json_file in self.cache_dir.glob("*.json"):
                json_file.unlink()

            self.metadata = {}
            self._save_metadata()

            # Reset stats
            self.stats = {"hits": 0, "misses": 0, "evictions": 0, "total_requests": 0}

    def cleanup_expired(self):
        """Remove expired cache entries."""
        with self._lock:
            current_time = time.time()
            expired_keys = []

            for key, info in self.metadata.items():
                if "expires" in info and info["expires"] < current_time:
                    expired_keys.append(key)

            for key in expired_keys:
                cache_path = Path(self.metadata[key]["path"])
                if cache_path.exists():
                    cache_path.unlink()
                del self.metadata[key]

            if expired_keys:
                self._save_metadata()
                print(f"Cleaned up {len(expired_keys)} expired cache entries")

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            hit_rate = (
                self.stats["hits"] / self.stats["total_requests"] * 100
                if self.stats["total_requests"] > 0
                else 0
            )

            total_size = sum(info.get("size", 0) for info in self.metadata.values())

            return {
                "hit_rate": hit_rate,
                "total_requests": self.stats["total_requests"],
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "evictions": self.stats["evictions"],
                "total_entries": len(self.metadata),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "max_size_mb": self.max_size_mb,
            }

    def list_entries(self) -> List[Dict]:
        """List all cache entries."""
        with self._lock:
            entries = []
            for key, info in self.metadata.items():
                entries.append(
                    {
                        "key": key,
                        "size": info.get("size", 0),
                        "created": datetime.fromtimestamp(info.get("created", 0)),
                        "accessed": datetime.fromtimestamp(info.get("accessed", 0)),
                        "expires": (
                            datetime.fromtimestamp(info["expires"])
                            if "expires" in info
                            else None
                        ),
                    }
                )

            return sorted(entries, key=lambda x: x["accessed"], reverse=True)


# Module-level default cache manager to prevent memory leaks
_default_cache_manager = None


def get_default_cache_manager() -> CacheManager:
    """Get or create the default cache manager instance"""
    global _default_cache_manager
    if _default_cache_manager is None:
        _default_cache_manager = CacheManager()
    return _default_cache_manager


def cached(cache_manager: Optional[CacheManager] = None, ttl: Optional[float] = None):
    """Decorator to cache function results"""

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use provided cache manager or default instance
            cm = cache_manager or get_default_cache_manager()

            # Generate cache key from function name and arguments
            cache_key = {"function": func.__name__, "args": args, "kwargs": kwargs}

            # Try to get from cache
            result = cm.get(cache_key)

            if result is not None:
                return result

            # Compute result and cache it
            result = func(*args, **kwargs)
            cm.set(cache_key, result, ttl=ttl)

            return result

        return wrapper

    return decorator


class DataCache:
    """Specialized cache for data processing results."""

    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache = cache_manager or CacheManager()

    def cache_preprocessed_data(
        self,
        file_path: Union[str, Path],
        preprocessing_config: Dict,
        processed_data: pd.DataFrame,
        ttl: int = 3600,
    ):  # 1 hour default
        """Cache preprocessed data."""
        file_path = Path(file_path)
        file_mtime = file_path.stat().st_mtime if file_path.exists() else 0

        cache_key = {
            "type": "preprocessed_data",
            "file_path": str(file_path),
            "config": preprocessing_config,
            "file_mtime": file_mtime,
        }

        self.cache.set(cache_key, processed_data, ttl)

    def get_preprocessed_data(
        self, file_path: Union[str, Path], preprocessing_config: Dict
    ) -> Optional[pd.DataFrame]:
        """Get cached preprocessed data."""
        file_path = Path(file_path)
        file_mtime = file_path.stat().st_mtime if file_path.exists() else 0

        cache_key = {
            "type": "preprocessed_data",
            "file_path": str(file_path),
            "config": preprocessing_config,
            "file_mtime": file_mtime,
        }

        return self.cache.get(cache_key)

    def cache_simulation_results(
        self,
        model_params: Dict,
        simulation_config: Dict,
        results: Dict,
        ttl: int = 7200,
    ):  # 2 hours default
        """Cache simulation results."""
        cache_key = {
            "type": "simulation_results",
            "model_params": model_params,
            "simulation_config": simulation_config,
        }

        self.cache.set(cache_key, results, ttl)

    def get_simulation_results(
        self, model_params: Dict, simulation_config: Dict
    ) -> Optional[Dict]:
        """Get cached simulation results."""
        cache_key = {
            "type": "simulation_results",
            "model_params": model_params,
            "simulation_config": simulation_config,
        }

        return self.cache.get(cache_key)

    def cache_validation_results(
        self,
        protocol_name: str,
        validation_config: Dict,
        results: Dict,
        ttl: int = 86400,
    ):  # 24 hours default
        """Cache validation results."""
        cache_key = {
            "type": "validation_results",
            "protocol": protocol_name,
            "config": validation_config,
        }

        self.cache.set(cache_key, results, ttl)

    def get_validation_results(
        self, protocol_name: str, validation_config: Dict
    ) -> Optional[Dict]:
        """Get cached validation results."""
        cache_key = {
            "type": "validation_results",
            "protocol": protocol_name,
            "config": validation_config,
        }

        return self.cache.get(cache_key)


def main():
    """Demonstrate cache management system."""
    print("APGI Framework - Cache Management System")
    print("=" * 50)

    # Initialize cache manager
    cache = CacheManager(max_size_mb=100)  # 100MB limit

    print(f"Cache directory: {cache.cache_dir}")
    print(f"Max size: {cache.max_size_mb} MB")

    # Demonstrate basic caching
    print("\nBasic caching demonstration:")

    # Cache some data
    test_data = {
        "simulation_results": [1, 2, 3, 4, 5],
        "parameters": {"tau_S": 0.1, "tau_theta": 0.2},
        "timestamp": datetime.now().isoformat(),
    }

    cache.set("test_key", test_data)
    print("  Cached data with key: test_key")

    # Retrieve data
    retrieved_data = cache.get("test_key")
    print(f"  Retrieved data: {retrieved_data is not None}")

    # Test cache miss
    miss_data = cache.get("non_existent_key", "default")
    print(f"  Cache miss result: {miss_data}")

    # Demonstrate data cache
    print("\nData cache demonstration:")
    data_cache = DataCache(cache)

    # Cache some preprocessed data
    import pandas as pd

    sample_df = pd.DataFrame(
        {
            "eeg": np.random.normal(0, 1, 100),
            "pupil": np.random.normal(5, 1, 100),
            "eda": np.random.normal(1, 0.5, 100),
        }
    )

    preprocessing_config = {"normalize": True, "filter": True}
    data_cache.cache_preprocessed_data("test_file.csv", preprocessing_config, sample_df)
    print("  Cached preprocessed data")

    # Retrieve cached data
    cached_df = data_cache.get_preprocessed_data("test_file.csv", preprocessing_config)
    print(f"  Retrieved cached data: {cached_df is not None}")
    print(f"  Data shape: {cached_df.shape if cached_df is not None else 'None'}")

    # Show cache statistics
    print("\nCache statistics:")
    stats = cache.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # List cache entries
    print("\nCache entries:")
    entries = cache.list_entries()
    for entry in entries[:3]:  # Show first 3 entries
        print(
            f"  Key: {entry['key'][:16]}... Size: {entry['size']} bytes Accessed: {entry['accessed']}"
        )

    if len(entries) > 3:
        print(f"  ... and {len(entries) - 3} more entries")

    print("\nCache management system ready!")


if __name__ == "__main__":
    main()
