"""
APGI Validation Protocol 0: HEP Proxy Validation (EP-0)
========================================================

Canonical protocol for EP-0: Heart-Evoked Potential Proxy Validation Sub-Protocol.

STATUS: STUB — not yet implemented.

This protocol validates the HEP (Heart-Evoked Potential) as a proxy measure for
interoceptive signalling within the APGI framework. EP-0 establishes the foundational
empirical grounding for interoceptive precision weighting, which feeds into
several downstream protocols (EP-1, EP-8, EP-11).

PARTIAL COVERAGE
----------------
FP_09_NeuralSignaturesP3bHEP.py provides partial coverage from the *falsification* side:
it tests whether P3b amplitude and HEP modulation are consistent with APGI predictions.
However, that script does not constitute a full validation pass for EP-0 —
it does not confirm the HEP-proxy relationship in the positive sense required by
the APGI empirical credibility roadmap.

REQUIRED IMPLEMENTATION
-----------------------
A full EP-0 implementation would need to:
1. Extract HEP epochs from pre-collected ECG+EEG recordings (500 ms post-R-peak).
2. Confirm HEP amplitude correlates with interoceptive accuracy (heartbeat detection task).
3. Demonstrate HEP modulation tracks Π (precision weight) changes under APGI predictions.
4. Show that source-localised HEP generators overlap with anterior insula / vmPFC ROIs.

FALSIFICATION CRITERIA
----------------------
If HEP amplitude does not differ significantly between high- and low-interoceptive-accuracy
conditions (p > 0.05, corrected), or if source localisation fails to implicate
anterior insula, then the HEP-proxy assumption underlying EP-0 is falsified.
"""

from __future__ import annotations

from typing import Any, Dict


def validate(**kwargs: Any) -> Dict[str, Any]:
    """Entry point for VP framework — returns not-implemented sentinel."""
    return {
        "status": "not_implemented",
        "canonical": "EP-0",
        "protocol": "VP_00_HEPProxyValidation",
        "message": (
            "EP-0 (HEP Proxy Validation) stub — no implementation yet. "
            "Partial falsification coverage exists in FP_09_NeuralSignaturesP3bHEP.py."
        ),
    }


def run_validation(**kwargs: Any) -> Dict[str, Any]:
    return validate(**kwargs)
