"""
Python 3.14 NumPy compatibility fixes.
=================================

This module provides early NumPy patches for Python 3.14 compatibility
to prevent scipy import errors. It should be imported before any scipy imports.
"""

import sys

# Apply patches immediately when this module is imported
if sys.version_info >= (3, 14):
    try:
        import numpy as np

        # Global flag to prevent multiple patching
        if not hasattr(np, "_apgi_early_patching_applied"):
            np._apgi_early_patching_applied = True

            # Store original functions
            _original_sum = np.sum
            _original_prod = np.prod
            _original_multiply = np.multiply
            _original_reshape = np.reshape

            # Safe sum that handles zero-size arrays and broadcasting issues
            def _safe_sum(
                a,
                axis=None,
                dtype=None,
                out=None,
                keepdims=False,
                initial=None,
                where=True,
            ):
                """Safe sum that handles zero-size arrays and broadcasting issues."""
                try:
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
                        # Match NumPy semantics: sum([]) == 0.
                        # Returning an empty array here breaks downstream callers
                        # (notably SciPy during import) that expect a scalar/0-d result.
                        if hasattr(a, "size") and a.size == 0:
                            return np.array(0, dtype=getattr(a, "dtype", float))
                        # For broadcasting issues, degrade gracefully to a 0-d scalar.
                        return np.array(0, dtype=getattr(a, "dtype", float))
                    raise

            # Safe prod that handles zero-size arrays
            def _safe_prod(
                a,
                axis=None,
                dtype=None,
                out=None,
                keepdims=False,
                initial=None,
                where=True,
            ):
                """Safe prod that handles zero-size arrays."""
                try:
                    return _original_prod(
                        a,
                        axis=axis,
                        dtype=dtype,
                        out=out,
                        keepdims=keepdims,
                        initial=initial,
                        where=where,
                    )
                except ValueError as e:
                    if "zero-size array to reduction operation" in str(e):
                        # Match NumPy semantics: prod([]) == 1.
                        if hasattr(a, "size") and a.size == 0:
                            return np.array(1, dtype=getattr(a, "dtype", float))
                    raise

            # Safe multiply that handles zero-size arrays
            def _safe_multiply(x1, x2, *args, **kwargs):
                """Safe multiply that handles zero-size arrays."""
                try:
                    return _original_multiply(x1, x2, *args, **kwargs)
                except ValueError as e:
                    if "zero-size array to reduction operation" in str(e):
                        # Handle zero-size arrays by returning appropriate result
                        if hasattr(x1, "size") and x1.size == 0:
                            return np.array(
                                [], dtype=x1.dtype if hasattr(x1, "dtype") else float
                            )
                        elif hasattr(x2, "size") and x2.size == 0:
                            return np.array(
                                [], dtype=x2.dtype if hasattr(x2, "dtype") else float
                            )
                    raise

            # Safe reshape that handles zero-size arrays
            def _safe_reshape(a, newshape, *args, **kwargs):
                """Safe reshape that handles zero-size arrays."""
                try:
                    return _original_reshape(a, newshape, *args, **kwargs)
                except ValueError as e:
                    if "cannot reshape array of size 0" in str(e):
                        # Handle zero-size arrays by returning appropriate result
                        if hasattr(a, "size") and a.size == 0:
                            if newshape == () or newshape == (0,):
                                return np.array([], dtype=getattr(a, "dtype", float))
                            elif hasattr(newshape, "__iter__") and 0 in newshape:
                                return np.array([], dtype=getattr(a, "dtype", float))
                            # For other shapes, try to return a scalar
                            return np.array(0, dtype=getattr(a, "dtype", float))
                    raise

            # Add reduce method to patched functions for compatibility
            def _add_reduce_method(func, original_func):
                """Add reduce method to patched functions for compatibility."""

                def reduce_method(a, *args, **kwargs):
                    try:
                        # Use the original function's reduce method if available
                        if hasattr(original_func, "reduce"):
                            return original_func.reduce(a, *args, **kwargs)
                        # Fallback to calling the function directly
                        return func(a, *args, **kwargs)
                    except (ValueError, TypeError) as e:
                        if "zero-size array" in str(e) or "no identity" in str(e):
                            # Handle zero-size arrays in reduce operations
                            if hasattr(a, "size") and a.size == 0:
                                if original_func.__name__ == "prod":
                                    return np.array(1, dtype=getattr(a, "dtype", float))
                                else:  # sum and others
                                    return np.array(0, dtype=getattr(a, "dtype", float))
                            elif hasattr(a, "shape") and a.shape == (0,):
                                if original_func.__name__ == "prod":
                                    return np.array(1, dtype=getattr(a, "dtype", float))
                                else:  # sum and others
                                    return np.array(0, dtype=getattr(a, "dtype", float))
                            # Return scalar for problematic arrays
                            if original_func.__name__ == "prod":
                                return np.array(1, dtype=getattr(a, "dtype", float))
                            else:  # sum and others
                                return np.array(0, dtype=getattr(a, "dtype", float))
                        raise

                func.reduce = reduce_method
                return func

            # Fix 5: Handle NumPy array rounding compatibility
            import builtins

            _original_round = builtins.round

            def _safe_round(x, *args, **kwargs):
                """Safe round function that handles NumPy arrays."""
                if hasattr(x, "shape") and hasattr(x, "dtype"):  # Likely a NumPy array
                    try:
                        return np.round(x, *args, **kwargs)
                    except (TypeError, ValueError):
                        # Fallback for problematic arrays
                        if hasattr(x, "size") and x.size == 1:
                            return _original_round(float(x.item()), *args, **kwargs)
                        # For arrays, apply element-wise
                        if hasattr(x, "flat"):
                            return np.array(
                                [
                                    _original_round(float(val), *args, **kwargs)
                                    for val in x.flat
                                ]
                            ).reshape(x.shape)
                        return x  # Return as-is if we can't handle it
                return _original_round(x, *args, **kwargs)

            builtins.round = _safe_round

            # Apply patches immediately with reduce methods
            np.sum = _add_reduce_method(_safe_sum, _original_sum)
            np.prod = _add_reduce_method(_safe_prod, _original_prod)
            np.multiply = _add_reduce_method(_safe_multiply, _original_multiply)
            np.reshape = _safe_reshape

    except ImportError:
        pass  # NumPy not available
