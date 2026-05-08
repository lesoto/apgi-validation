"""
Threshold module consolidation tests.

Project convention:
- Single source of truth lives at `utils/falsification_thresholds.py`
- The former root-level `falsification_thresholds.py` shim is intentionally removed
"""

from pathlib import Path

import pytest


def test_root_threshold_shim_removed():
    """Root shim was removed as part of consolidation into utils/."""
    project_root = Path(__file__).parent.parent
    assert not (project_root / "falsification_thresholds.py").exists()


def test_all_threshold_constants_exported_from_utils():
    """Test that all important threshold constants are exported from utils/__init__.py."""
    from utils import (
        F1_1_ALPHA,
        F1_1_MIN_ADVANTAGE_PCT,
        F1_1_MIN_COHENS_D,
        F5_3_FALSIFICATION_RATIO,
        F6_2_MIN_R2,
    )

    # Verify they're all accessible and have correct types
    assert isinstance(F1_1_MIN_ADVANTAGE_PCT, float)
    assert isinstance(F1_1_MIN_COHENS_D, float)
    assert isinstance(F1_1_ALPHA, float)
    assert isinstance(F6_2_MIN_R2, float)
    assert isinstance(F5_3_FALSIFICATION_RATIO, float)

    # Verify key values match canonical file
    from utils.falsification_thresholds import (
        F5_3_FALSIFICATION_RATIO as canonical_F5_3_FALSIFICATION_RATIO,
    )
    from utils.falsification_thresholds import F6_2_MIN_R2 as canonical_F6_2_MIN_R2

    assert F6_2_MIN_R2 == canonical_F6_2_MIN_R2
    assert F5_3_FALSIFICATION_RATIO == canonical_F5_3_FALSIFICATION_RATIO


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
