#!/usr/bin/env python3
"""
Module Cache System
==================
Optimizes dynamic imports in hot paths with intelligent caching and preloading.
Reduces import overhead for frequently accessed modules.
"""

import hashlib
import importlib.util
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set

logger = logging.getLogger("module_cache")


@dataclass
class CacheEntry:
    """Represents a cached module entry."""

    module: Any
    load_time: float
    file_hash: Optional[str]
    access_count: int = 0
    last_access: float = 0.0


class ModuleCache:
    """Thread-safe module cache with intelligent eviction."""

    def __init__(self, max_size: int = 100, ttl_seconds: float = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._preloaded_modules: Set[str] = set()

    def _compute_file_hash(self, file_path: Path) -> Optional[str]:
        """Compute hash of file contents for cache invalidation."""
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return None

    def get(
        self, module_path: Path, module_name: Optional[str] = None
    ) -> Optional[Any]:
        """Get module from cache or load and cache it."""
        module_name = module_name or module_path.stem
        cache_key = str(module_path.resolve())

        with self._lock:
            # Check cache
            if cache_key in self._cache:
                entry = self._cache[cache_key]

                # Check TTL
                if time.time() - entry.load_time > self.ttl_seconds:
                    logger.debug(f"Cache entry expired for {module_name}")
                    del self._cache[cache_key]
                    self._misses += 1
                else:
                    # Check if file has changed
                    current_hash = self._compute_file_hash(module_path)
                    if (
                        current_hash
                        and entry.file_hash
                        and current_hash != entry.file_hash
                    ):
                        logger.debug(
                            f"File changed for {module_name}, invalidating cache"
                        )
                        del self._cache[cache_key]
                        self._misses += 1
                    else:
                        # Update access statistics
                        entry.access_count += 1
                        entry.last_access = time.time()
                        self._hits += 1
                        logger.debug(f"Cache hit for {module_name}")
                        return entry.module

            self._misses += 1

        # Load module
        try:
            module = self._load_module_securely(module_path, module_name)

            # Cache the loaded module
            with self._lock:
                # Evict if cache is full
                if len(self._cache) >= self.max_size:
                    self._evict_lru()

                file_hash = self._compute_file_hash(module_path)
                entry = CacheEntry(
                    module=module,
                    load_time=time.time(),
                    file_hash=file_hash,
                    access_count=1,
                    last_access=time.time(),
                )
                self._cache[cache_key] = entry

            logger.debug(f"Cached module {module_name}")
            return module

        except Exception as e:
            logger.error(f"Failed to load module {module_name}: {e}")
            return None

    def _load_module_securely(self, module_path: Path, module_name: str) -> Any:
        """Safely load a Python module with path validation."""
        # Resolve the absolute path and validate it's within project root
        resolved_path = module_path.resolve()
        project_root = Path(__file__).parent.parent

        try:
            resolved_path.relative_to(project_root)
        except ValueError:
            raise ValueError(f"Module path outside project root: {module_path}")

        # Ensure it's a .py file
        if not resolved_path.suffix == ".py":
            raise ValueError(f"Module must be a .py file: {module_path}")

        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, resolved_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec for {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module

    def _evict_lru(self) -> None:
        """Evict least recently used module from cache."""
        if not self._cache:
            return

        # Find LRU entry
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_access)

        logger.debug(f"Evicting LRU module: {lru_key}")
        del self._cache[lru_key]

    def preload_modules(self, module_paths: Dict[str, Path]) -> None:
        """Preload frequently used modules into cache."""
        logger.info(f"Preloading {len(module_paths)} modules")

        for module_name, module_path in module_paths.items():
            if module_path.exists():
                module = self.get(module_path, module_name)
                if module:
                    self._preloaded_modules.add(module_name)
                    logger.debug(f"Preloaded module: {module_name}")
                else:
                    logger.warning(f"Failed to preload module: {module_name}")
            else:
                logger.warning(f"Module file not found for preloading: {module_path}")

    def invalidate(self, module_path: Path) -> None:
        """Invalidate cache entry for a specific module."""
        cache_key = str(module_path.resolve())

        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                logger.debug(f"Invalidated cache for {module_path}")

    def clear(self) -> None:
        """Clear all cached modules."""
        with self._lock:
            self._cache.clear()
            self._preloaded_modules.clear()
            logger.info("Module cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0

            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "cached_modules": len(self._cache),
                "preloaded_modules": len(self._preloaded_modules),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
            }

    def preload_common_modules(self) -> None:
        """Preload commonly used APGI modules."""
        project_root = Path(__file__).parent.parent

        common_modules = {
            "SurpriseIgnitionSystem": project_root
            / "Falsification"
            / "FP_ALL_Aggregator.py",
            "CrossSpeciesScaling": project_root
            / "Theory"
            / "APGI_Cross_Species_Scaling.py",
            "APGIMultimodalIntegration": project_root
            / "Theory"
            / "APGI_Multimodal_Integration.py",
            "APGIParameterEstimation": project_root
            / "Theory"
            / "APGI_Parameter_Estimation.py",
            "APGIPsychologicalStates": project_root
            / "Theory"
            / "APGI_Psychological_States.py",
            "APGIMasterValidator": project_root / "Validation" / "Master_Validation.py",
            "APGIValidationPipeline": project_root
            / "Validation"
            / "VP_01_SyntheticEEG_MLClassification.py",
        }

        self.preload_modules(common_modules)


# Global module cache instance
_module_cache: Optional[ModuleCache] = None


def get_module_cache() -> ModuleCache:
    """Get the global module cache instance."""
    global _module_cache
    if _module_cache is None:
        _module_cache = ModuleCache()
        # Preload common modules
        _module_cache.preload_common_modules()
    return _module_cache


def cached_import(
    module_path: Path, module_name: Optional[str] = None
) -> Optional[Any]:
    """Import module using cache."""
    cache = get_module_cache()
    return cache.get(module_path, module_name)


def secure_cached_import(
    module_path: Path, module_name: Optional[str] = None
) -> Optional[Any]:
    """Import module using cache with security validation."""
    return cached_import(module_path, module_name)


def preload_apgi_modules() -> None:
    """Preload all commonly used APGI modules."""
    cache = get_module_cache()
    cache.preload_common_modules()


def get_cache_stats() -> Dict[str, Any]:
    """Get module cache statistics."""
    cache = get_module_cache()
    return cache.get_stats()


def clear_module_cache() -> None:
    """Clear the module cache."""
    cache = get_module_cache()
    cache.clear()


# Decorator for caching function imports
def cache_imports(func):
    """Decorator to cache imports within a function."""

    def wrapper(*args, **kwargs):
        # Ensure module cache is available
        get_module_cache()
        return func(*args, **kwargs)

    return wrapper


# Context manager for batch preloading
class ModuleBatchLoader:
    """Context manager for batch loading modules."""

    def __init__(self):
        self.cache = get_module_cache()
        self.modules_to_load: Dict[str, Path] = {}

    def add_module(self, name: str, path: Path) -> "ModuleBatchLoader":
        """Add a module to be preloaded."""
        self.modules_to_load[name] = path
        return self

    def __enter__(self) -> "ModuleBatchLoader":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.modules_to_load:
            self.cache.preload_modules(self.modules_to_load)
            self.modules_to_load.clear()
