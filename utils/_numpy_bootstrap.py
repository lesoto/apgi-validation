"""
LEVEL DESIGNATION: Level 2 (information-theoretic)

Bridge to Level 1
"""

import sys

# Apply NumPy patches for Python 3.14+ compatibility BEFORE any other imports
if sys.version_info >= (3, 14):
    try:
        import numpy as np

        # Global flag to prevent multiple patching
        if not hasattr(np, "_apgi_patching_applied"):
            np._apgi_patching_applied = True

            # Store original functions
            _original_prod = np.prod
            _original_sum = np.sum

            # Safe prod that handles _NoValueType initial parameter and empty arrays
            def _safe_prod(
                a,
                axis=None,
                dtype=None,
                out=None,
                keepdims=np._NoValue,
                initial=np._NoValue,
                where=np._NoValue,
            ):
                """Safe prod that handles _NoValueType initial parameter and empty arrays."""
                try:
                    # Try the original function first
                    return _original_prod(
                        a,
                        axis=axis,
                        dtype=dtype,
                        out=out,
                        keepdims=keepdims,
                        initial=initial,
                        where=where,
                    )
                except (ValueError, TypeError) as e:
                    if "zero-size array" in str(e) or "no identity" in str(e):
                        # Handle zero-size arrays by returning identity element (1 for product)
                        if hasattr(a, "size") and a.size == 0:
                            # Return scalar 1 for empty product
                            if dtype is not None:
                                return np.array(1, dtype=dtype)
                            else:
                                return np.array(1, dtype=getattr(a, "dtype", float))
                        elif hasattr(a, "shape") and a.shape == (0,):
                            if dtype is not None:
                                return np.array(1, dtype=dtype)
                            else:
                                return np.array(1, dtype=getattr(a, "dtype", float))
                    raise

            # Safe sum that handles zero-size arrays
            def _safe_sum(
                a,
                axis=None,
                dtype=None,
                out=None,
                keepdims=np._NoValue,
                initial=np._NoValue,
                where=np._NoValue,
            ):
                """Safe sum that handles zero-size arrays and broadcasting issues."""
                try:
                    # Try the original function first
                    return _original_sum(
                        a,
                        axis=axis,
                        dtype=dtype,
                        out=out,
                        keepdims=keepdims,
                        initial=initial,
                        where=where,
                    )
                except (ValueError, TypeError) as e:
                    if "zero-size array" in str(e) or "broadcast together" in str(e):
                        # Handle zero-size arrays by returning identity element (0 for sum)
                        if hasattr(a, "size") and a.size == 0:
                            if dtype is not None:
                                return np.array(0, dtype=dtype)
                            else:
                                return np.array(0, dtype=getattr(a, "dtype", float))
                        elif hasattr(a, "shape") and a.shape == (0,):
                            if dtype is not None:
                                return np.array(0, dtype=dtype)
                            else:
                                return np.array(0, dtype=getattr(a, "dtype", float))
                    raise

            # Apply patches
            np.prod = _safe_prod  # type: ignore[assignment]
            np.sum = _safe_sum  # type: ignore[assignment]

            # Also patch in _core._methods where the actual implementations are
            if hasattr(np, "_core") and hasattr(np._core, "_methods"):
                np._core._methods._prod = _safe_prod
                np._core._methods._sum = _safe_sum
    except ImportError:
        pass  # NumPy not available yet
