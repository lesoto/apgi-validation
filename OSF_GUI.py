#!/usr/bin/env python3
"""
APGI Open Science Framework (OSF) Protocol Management GUI

Manages 15 prediction suites (EP-0 through EP-14) defined in the APGI
Framework Prediction Registry: pre-registration tracking, dependency
visualization, and report export.
"""

import json
import logging
import os
import queue
import sys
import tempfile
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import TclError, filedialog, messagebox, scrolledtext, ttk
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))

# ── APGI Design System ────────────────────────────────────────────────────────

_BG = "#f8f9fa"
_FG = "#212529"
_BLUE = "#2874a6"
_GREEN = "#155724"
_RED = "#721c24"
_BORDER = "#dee2e6"
_SURFACE = "#ffffff"
_MUTED = "#6c757d"
_SOFT = "#e9ecef"


def apply_apgi_theme(root: tk.Tk) -> ttk.Style:
    """Apply unified APGI theme to tkinter application."""
    style = ttk.Style()
    style.theme_use("clam")

    style.configure("TFrame", background=_BG)
    style.configure("TLabel", background=_BG, foreground=_FG, font=("Noto Sans", 10))
    style.configure("Header.TLabel", font=("Noto Sans", 12, "bold"), background=_BG, foreground=_FG)
    style.configure("Title.TLabel", font=("Noto Sans", 14, "bold"), background=_SURFACE, foreground=_FG)
    style.configure("Subtitle.TLabel", font=("Noto Sans", 10), background=_SURFACE, foreground=_MUTED)
    style.configure("Monospace.TLabel", font=("Noto Sans Mono", 11), background=_SURFACE, foreground=_BLUE)
    style.configure("TLabelframe", background=_BG, bordercolor=_BORDER)
    style.configure("TLabelframe.Label", background=_BG, foreground=_MUTED, font=("Noto Sans", 9, "bold"))
    style.configure("Card.TFrame", background=_SURFACE, borderwidth=1, relief="solid")
    style.configure("Card.TCheckbutton", background=_SURFACE)
    style.configure("TButton", padding=6, background=_SOFT, font=("Noto Sans", 10))
    style.map(
        "TButton", background=[("active", _BORDER), ("disabled", "#f1f3f5")], foreground=[("disabled", "#adb5bd")]
    )
    style.configure("Primary.TButton", background=_GREEN, foreground="white", font=("Noto Sans", 10, "bold"), padding=8)
    style.map(
        "Primary.TButton",
        background=[("active", "#0f3d1a"), ("disabled", _MUTED)],
        foreground=[("active", "white"), ("disabled", _BORDER)],
    )
    style.configure("Secondary.TButton", background=_BLUE, foreground="white", font=("Noto Sans", 10), padding=6)
    style.map(
        "Secondary.TButton",
        background=[("active", "#1f5a82"), ("disabled", _MUTED)],
        foreground=[("active", "white"), ("disabled", _BORDER)],
    )
    style.configure("Danger.TButton", background=_RED, foreground="white", font=("Noto Sans", 10, "bold"), padding=8)
    style.map(
        "Danger.TButton",
        background=[("active", "#5a161d"), ("disabled", _MUTED)],
        foreground=[("active", "white"), ("disabled", _BORDER)],
    )
    style.configure("Horizontal.TProgressbar", background=_BLUE, troughcolor=_BORDER, borderwidth=0)
    style.configure("TCombobox", font=("Noto Sans", 10))
    style.configure("TNotebook", background=_BG, tabmargins=[2, 5, 2, 0])
    style.configure("TNotebook.Tab", font=("Noto Sans", 10), padding=[10, 5])
    style.map("TNotebook.Tab", background=[("selected", _SURFACE)], expand=[("selected", [1, 1, 1, 0])])

    root.configure(background=_BG)
    return style


class APGICard(ttk.Frame):
    """Standardized information card for APGI applications."""

    def __init__(self, parent: tk.Widget, title: str, value: str, subtitle: str = "", **kwargs: Any) -> None:
        super().__init__(parent, style="Card.TFrame", **kwargs)
        container = ttk.Frame(self, padding=15, style="Card.TFrame")
        container.pack(fill="both", expand=True)

        ttk.Label(
            container, text=title.upper(), font=("Noto Sans", 11, "bold"), background=_SURFACE, foreground=_FG
        ).pack(anchor="w")

        ttk.Label(
            container, text=value, font=("Noto Sans Mono", 11), background=_SURFACE, foreground=_BLUE, wraplength=580
        ).pack(anchor="w", pady=(4, 8))

        if subtitle:
            ttk.Separator(container, orient="horizontal").pack(fill="x", pady=(0, 6))
            ttk.Label(
                container,
                text=subtitle,
                font=("Noto Sans", 9, "italic"),
                background=_SURFACE,
                foreground=_MUTED,
                wraplength=580,
            ).pack(anchor="w")


# ── Protocol Definitions ──────────────────────────────────────────────────────

EP_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    # ── Empirical Protocols ───────────────────────────────────────────────────
    "EP-0: HEP Proxy Validation": {
        "id": "EP-0",
        "title": "HEP Proxy Validation",
        "priority": 1,
        "type": "Empirical",
        "prereg_required": True,
        "depends_on": [],
        "status": "Not started",
        "description": (
            "Heartbeat-evoked potential (HEP) proxy validation for interoceptive precision gating. "
            "Pred 0.A: r(HEP, interoceptive d′) > 0.35, p < 0.01, two-tailed; replicated r ≥ 0.25 "
            "(N ≥ 30). Pred 0.B: physostigmine increases HEP amplitude ≥ 15% vs. placebo "
            "(Cohen's d ≥ 0.50, BF₁₀ ≥ 10). Pred 0.C: r(HEP amplitude, aINS BOLD) > 0.30 "
            "within participants after arousal covariate control."
        ),
        "platform": "OSF",
        "sample_size": "N ≥ 30 primary; N ≥ 30 independent validation subsample (Pred 0.A replication)",
        "measures": [
            "HEP amplitude",
            "Interoceptive d′ (heartbeat discrimination)",
            "aINS BOLD (fMRI concurrent)",
            "Cardiac cycle phase",
            "Pupil diameter (ACh target-engagement control)",
            "RMSSD (HRV arousal covariate)",
        ],
        "analysis": (
            "Pearson r (HEP vs. interoceptive d′); linear mixed-effects (HEP ~ ACh × time); "
            "within-participant partial correlation (HEP amplitude vs. aINS BOLD, arousal covariate controlled); "
            "Bayes factor (BF₁₀) for physostigmine effect"
        ),
        "falsification_criterion": (
            "Pred 0.A: r < 0.20 across two independent samples. "
            "Pred 0.B: No HEP amplitude change despite verified ACh elevation (pupil constriction confirmed). "
            "Pred 0.C: aINS BOLD and HEP statistically independent after arousal covariate."
        ),
        "prereg_status": "Pending",
        "notes": "",
    },
    "EP-1: EEG Interoceptive Precision Gating": {
        "id": "EP-1",
        "title": "EEG Interoceptive Precision Gating",
        "priority": 1,
        "type": "Empirical",
        "prereg_required": True,
        "depends_on": ["EP-0"],
        "status": "Not started",
        "description": (
            "Tests APGI prediction that interoceptive precision gates sensory processing. "
            "Pred 1.A (state): interoceptive focus produces greater P3b for near-threshold stimuli than "
            "exteroceptive focus or dual-task control (ηₚ² ≥ 0.06, FWE); HEP predicts P3b (partial r > 0.4). "
            "Pred 1.A-trait: high-IA d′ > low-IA d′ (Cohen's d ≥ 0.5) with group × modality dissociation. "
            "Pred 1.B: cardiac-phase-locked detection advantage at diastole (300–500 ms) vs. systole (0–150 ms). "
            "Pred 1.C: top-tertile IA shows strongest P3b condition effect (accuracy × condition ηₚ² ≥ 0.08)."
        ),
        "platform": "OSF",
        "sample_size": "N ≥ 30 in preregistered contrast; N ≥ 30 per group for Pred 1.A-trait",
        "measures": [
            "P3b amplitude (near-threshold stimuli)",
            "HEP amplitude",
            "Interoceptive d′ (heartbeat detection)",
            "Cardiac phase at stimulus onset (systole vs. diastole)",
            "RMSSD and pupil diameter (arousal covariates)",
        ],
        "analysis": (
            "Repeated-measures ANOVA / linear mixed-effects (P3b ~ condition, FWE-corrected); "
            "logistic mixed-effects (detection ~ cardiac_phase × contrast + cardiac_phase | participant); "
            "partial correlation HEP vs. P3b (RMSSD and pupil diameter controlled); "
            "Mann–Whitney or independent-samples t (high-IA vs. low-IA d′)"
        ),
        "falsification_criterion": (
            "Pred 1.A: No significant P3b condition main effect (all p > 0.10) across N ≥ 30 in preregistered "
            "contrast; or HEP–P3b partial correlation < r = 0.20 after arousal covariate control. "
            "Pred 1.A-trait: High-IA and Low-IA do not differ on interoceptive d′ (d < 0.3 across N ≥ 30 per "
            "group); or d′ group difference replicates equally in exteroceptive condition (no interoceptive specificity). "
            "Pred 1.B: Diastole vs. systole hit rate advantage < 5% across two independent samples (interaction p > 0.15). "
            "Pred 1.C: Accuracy × condition interaction p > 0.10."
        ),
        "prereg_status": "Pending",
        "notes": "Near-threshold contrast titration mandatory; suprathreshold matched control block required.",
    },
    "EP-2: Causal TMS Insula/dlPFC": {
        "id": "EP-2",
        "title": "Causal TMS Insula/dlPFC",
        "priority": 2,
        "type": "Empirical",
        "prereg_required": True,
        "depends_on": ["EP-0"],
        "status": "Not started",
        "description": (
            "Causal TMS intervention over pIC, dlPFC/PPC, and vertex sham. "
            "Pred 4.A: pIC TMS reduces PCI ~20%; dlPFC/PPC TMS reduces PCI 15–25%; "
            "HEP–PCI coupling abolished under pIC TMS only (site × coupling interaction p < 0.05). "
            "Pred 4.B: pIC TMS reduces HEP–P3b coupling to < 0.15 (baseline > 0.35) while sparing "
            "exteroceptive P3b within 10% of vertex; dlPFC TMS leaves HEP unaffected (BF₀₁ ≥ 6). "
            "Pred 4.C: high-baseline IA × TMS-site interaction on PCI (η² > 0.10); effect absent for dlPFC."
        ),
        "platform": "OSF",
        "sample_size": "N ≥ 24 (three-site within-participant crossover: pIC, dlPFC/PPC, vertex)",
        "measures": [
            "PCI (Perturbational Complexity Index)",
            "HEP amplitude",
            "HEP–PCI coupling coefficient",
            "HEP–P3b coupling coefficient",
            "Exteroceptive P3b amplitude",
            "Interoceptive accuracy (IA)",
        ],
        "analysis": (
            "Linear mixed-effects (PCI ~ TMS_site, p < 0.05); site × coupling interaction; "
            "Bayesian equivalence testing (ROPE d = [−0.15, +0.15], BF₀₁ for dlPFC HEP effect); "
            "accuracy × TMS-site ANOVA on PCI"
        ),
        "falsification_criterion": (
            "Pred 4.A: No PCI change vs. vertex sham across all active sites; or equivalent PCI reduction "
            "at vertex as active sites; or HEP–PCI coupling equally abolished by dlPFC TMS (no dissociation). "
            "Pred 4.B: Uniform suppression of both interoceptive and exteroceptive P3b across pIC and dlPFC TMS; "
            "or HEP equally reduced by dlPFC and pIC TMS. "
            "Pred 4.C: No accuracy × TMS-site interaction; high- and low-accuracy groups show equivalent "
            "PCI response across all TMS sites."
        ),
        "prereg_status": "Pending",
        "notes": "",
    },
    "EP-3: Active Inference Simulations": {
        "id": "EP-3",
        "title": "Active Inference Simulations",
        "priority": 1,
        "type": "Computational",
        "prereg_required": True,
        "depends_on": [],
        "status": "Not started",
        "description": (
            "Computational validation of the APGI active inference architecture (Protocol 2). "
            "Pred 2.A: Full APGI agents converge within 50–80 trials on IGT; cumulative reward ratio "
            "APGI/β_SM-lesion > 1.15; APGI > GNWT-only > Standard PP (all pairwise permutation p < 0.05). "
            "Pred 2.B: 70–85% of ignition events satisfy Πⁱ·|zⁱ| > Πᵉ·|zᵉ|. "
            "Pred 2.C: M̂ leads θₜ crossing by ≥ 1 trial in ≥ 75% of ignition events (Kendall τ > 0.3 at negative lag). "
            "Pred 2.D: β_SM lesion degrades performance most at high-volatility σ_env = 0.6. "
            "Pred 2.E: APGI BIC lower than alternatives by ΔBIC ≥ 10."
        ),
        "platform": "OSF",
        "sample_size": "N/A (computational) — convergence criterion: 50–80 trials per Pred 2.A",
        "measures": [
            "Cumulative reward (IGT, all volatility conditions)",
            "Trial-to-criterion (convergence within 50–80 trials)",
            "Post-ignition action selection entropy",
            "Interoceptive dominance fraction (Πⁱ·|zⁱ| > Πᵉ·|zᵉ|)",
            "Cross-correlation M̂ vs. θₜ crossing (lag −5 to +5 trials)",
            "BIC/WAIC model comparison (APGI vs. Standard PP vs. GNWT-only)",
        ],
        "analysis": (
            "Permutation tests (pairwise reward comparisons); Wilcoxon (entropy pre vs. post ignition); "
            "cross-correlation analysis with Kendall τ; BIC/log-likelihood ratio test (LRT p < 0.01); "
            "β_SM vs. Πⁱ-lesion contrast at σ_env = 0.6"
        ),
        "falsification_criterion": (
            "Pred 2.A: No significant pairwise performance advantage across all volatility conditions; "
            "or APGI fails convergence within 80 trials even when cumulative reward advantage present. "
            "Pred 2.B: < 60% of ignition events satisfy Πⁱ·|zⁱ| > Πᵉ·|zᵉ| (interoceptive dominance independently falsified). "
            "Pred 2.C: M̂ activation simultaneous with or lagging threshold crossing; cross-correlation peak at lag 0 or positive. "
            "Pred 2.D: β_SM produces equivalent or lesser deficit vs. Πⁱ-lesion under all volatility conditions. "
            "Pred 2.E: APGI BIC ≥ any alternative model's BIC; or ΔBIC < 3 against GNWT-only."
        ),
        "prereg_status": "Pending",
        "notes": "Advantage must emerge within 80 trials, not just cumulatively (Pred 2.A).",
    },
    "EP-4: Disorders of Consciousness": {
        "id": "EP-4",
        "title": "Disorders of Consciousness Biomarker",
        "priority": 3,
        "type": "Clinical",
        "prereg_required": True,
        "depends_on": ["EP-0"],
        "status": "Not started",
        "description": (
            "Clinical validation of the APGI interoceptive precision model in VS/UWS and MCS patients (Protocol 6). "
            "Pred 6.A: Joint HEP + PCI model ΔR² ≥ 0.05 over best univariate model (PRIMARY); AUC > 0.80 "
            "(aspirational secondary only). "
            "Pred 6.B: HEP amplitude MCS > VS/UWS (Mann–Whitney p < 0.05, d > 0.5); four-group ordinal "
            "gradient confirmed for both HEP and PCI; DMN–PCI r > 0.5. "
            "Pred 6.C: Interoceptive perturbation increases PCI ≥ 10% in MCS; VS/UWS shows no significant change. "
            "Pred 6.D: Within-participant Spearman r(HEP, GCS-S) > 0.4 at 3-month follow-up."
        ),
        "platform": "OSF",
        "sample_size": "N ≥ 20 DOC patients (VS/UWS and MCS) + N ≥ 20 healthy controls",
        "measures": [
            "HEP amplitude",
            "PCI (Perturbational Complexity Index)",
            "GCS-S score (3-month and 6-month follow-up)",
            "DMN–PCI correlation",
            "Perturbation-evoked ΔPCI",
            "HEP–PCI change correlation",
        ],
        "analysis": (
            "Linear regression (joint HEP + PCI model, ΔR², LRT p < 0.05); "
            "Mann–Whitney (HEP MCS vs. VS/UWS); ordinal gradient test (four-group: VS/UWS < MCS < EMCS < controls); "
            "paired t (PCI pre vs. post perturbation, within MCS and VS/UWS); "
            "within-participant Spearman r(HEP, GCS-S) at 3-month follow-up"
        ),
        "falsification_criterion": (
            "Pred 6.A (PRIMARY): Joint model R² ≤ max(univariate HEP R², univariate PCI R²); or ΔR² < 0.05. "
            "Pred 6.B: No significant HEP difference between MCS and VS/UWS; four-group gradient absent for HEP or PCI. "
            "Pred 6.C: No significant PCI change post-perturbation in MCS (ΔPCI < 5%, p > 0.10); or equivalent PCI "
            "change in VS/UWS and MCS; or perturbation-evoked ΔPCI statistically indistinguishable from arousal-matched control. "
            "Pred 6.D: HEP shows no significant longitudinal correlation with GCS-S (r < 0.2) at 3-month follow-up."
        ),
        "prereg_status": "Pending",
        "notes": (
            "Ethics approval and patient consent protocols required before data collection. "
            "6-month follow-up (Pred 6.D) reported as exploratory pending OSF amendment."
        ),
    },
    "EP-5: fMRI Anticipation vs. Experience": {
        "id": "EP-5",
        "title": "fMRI Anticipation vs. Experience",
        "priority": 1,
        "type": "Empirical",
        "prereg_required": True,
        "depends_on": ["EP-0"],
        "status": "Not started",
        "description": (
            "fMRI study testing M̂ (somatic marker anticipation) vs. εⁱ (interoceptive error) dissociation (Protocol 3). "
            "Pred 3.A: vmPFC BOLD parametrically modulated by EV during anticipation foreperiod (p < 0.05 FWE) "
            "but does NOT correlate with outcome-locked SCR (r < 0.20); posterior insula–SCR r > 0.40. "
            "Pred 3.B: vmPFC→aINS PPI coefficient increases during anticipation foreperiod (Δr ≥ 0.30); "
            "vmPFC→pIC PPI remains flat (BF₀₁ ≥ 6, ROPE d=[−0.15,+0.15]). "
            "Pred 3.C: vmPFC modulated by option EV (somatic valence), not sensory contrast (d > 0.4). "
            "Pred 3.D: Removing foreperiod (0 ms ISI) abolishes vmPFC→aINS coupling (BF₀₁ ≥ 6) and "
            "vmPFC EV parametric modulation, while leaving posterior insula outcome-locked activity intact."
        ),
        "platform": "OSF",
        "sample_size": "N ≥ 32",
        "measures": [
            "BOLD vmPFC (anticipation foreperiod, parametric EV modulation)",
            "BOLD posterior insula (outcome-locked, SCR correlation)",
            "SCR (skin conductance response)",
            "PPI coefficient vmPFC→aINS",
            "PPI coefficient vmPFC→pIC",
            "Option expected value (EV) and sensory contrast regressors",
        ],
        "analysis": (
            "GLM with parametric EV modulator (anticipation foreperiod vs. neutral baseline, p < 0.05 FWE); "
            "PPI analysis (vmPFC seed → aINS and pIC); "
            "Bayesian equivalence testing vmPFC→pIC coupling (ROPE d=[−0.15,+0.15], BF₀₁ ≥ 6); "
            "0 ms ISI condition as foreperiod-abolition control"
        ),
        "falsification_criterion": (
            "Pred 3.A: vmPFC BOLD correlates significantly with outcome-locked SCR (r > 0.35), indicating vmPFC "
            "encodes raw εⁱ rather than anticipatory M̂; or vmPFC shows no anticipation-period EV parametric "
            "modulation (p > 0.20). "
            "Pred 3.B: vmPFC→pIC coupling increases during anticipation (Δr > 0.30, p < 0.05); or vmPFC→aINS "
            "coupling is absent; or aINS/pIC dissociation not significant (both estimates statistically equivalent). "
            "Pred 3.C: vmPFC activation correlates with sensory contrast regardless of option valence; or valence "
            "and contrast effects statistically equivalent. "
            "Pred 3.D: vmPFC→aINS coupling in 0 ms foreperiod equivalent to standard foreperiod condition "
            "(foreperiod removal does not reduce coupling)."
        ),
        "prereg_status": "Pending",
        "notes": "",
    },
    "EP-6: iEEG All-or-None Dynamics": {
        "id": "EP-6",
        "title": "iEEG All-or-None Dynamics",
        "priority": 2,
        "type": "Clinical-Empirical",
        "prereg_required": True,
        "depends_on": ["EP-3"],
        "status": "Not started",
        "description": (
            "Intracranial EEG study in epilepsy patients testing APGI ignition predictions (Protocol 5). "
            "Pred 5.A: frontoparietal cortex shows bimodal high-gamma power (Hartigan's dip p < 0.05; "
            "2-component Gaussian ΔBIC > 10). "
            "Pred 5.B: Bimodality specific to frontoparietal ignition network (not occipital); "
            "intermediate-state bouts < 100 ms mean duration (prevalence < 15% of trial duration). "
            "Pred 5.C: AC1 of pre-ignition high-gamma (70–150 Hz) increases monotonically 500 ms before "
            "detected stimuli (Kendall τ > 0.3, permutation p < 0.05); absent in non-detected trials. "
            "Pred 5.D: Long-range gamma coherence (15–80 Hz) frontoparietal sites predicts seen vs. unseen "
            "(point-biserial r > 0.4, 200–400 ms); HEP–coherence r > 0.25 (APGI-specific tier)."
        ),
        "platform": "OSF",
        "sample_size": "N ≥ 10 iEEG patients",
        "measures": [
            "High-gamma power (70–150 Hz) — frontoparietal and occipital sites",
            "Bimodality index (Hartigan's dip statistic)",
            "AC1 of high-gamma envelope (500 ms pre-stimulus)",
            "Long-range gamma coherence (15–80 Hz) frontoparietal",
            "HEP amplitude (concurrent cardiac recording)",
            "Trial-by-trial detection outcome (seen / unseen)",
        ],
        "analysis": (
            "Hartigan's dip test and 2- vs. 1-component Gaussian BIC comparison; "
            "mixed-effects region × bimodality-index interaction; "
            "Kendall τ AC1 trend in detected vs. non-detected trials (10 000 permutations); "
            "point-biserial r (coherence vs. detection); "
            "Pearson r (HEP amplitude vs. coherence in seen trials)"
        ),
        "falsification_criterion": (
            "Pred 5.A: Unimodal continuous distribution of high-gamma power; dip test p > 0.20; no bistability. "
            "Pred 5.B: Uniform bimodality across all recorded regions; or intermediate-state bouts > 150 ms in "
            "≥ 50% of trials; or prevalence > 30% of trial duration. "
            "Pred 5.C: AC1 flat or decreasing before detected stimuli; no monotonic pre-ignition trend. "
            "Pred 5.D: r < 0.20 for frontoparietal coherence vs. detection; or coherence peak outside 200–400 ms; "
            "or occipital coherence statistically equivalent to frontoparietal. "
            "Note: failure of Pred 5.D criterion (3) alone reclassifies as GNW-consistent only — does not falsify "
            "Pred 5.D but removes APGI-specific confirmation (pre-specified reclassification, no amendment required)."
        ),
        "prereg_status": "Pending",
        "notes": "Ethics approval required; patients recruited via epilepsy-monitoring unit.",
    },
    "EP-7: Pharmacological Modulation": {
        "id": "EP-7",
        "title": "Pharmacological Modulation",
        "priority": 2,
        "type": "Empirical",
        "prereg_required": True,
        "depends_on": ["EP-0"],
        "status": "Not started",
        "description": (
            "Pharmacological modulation of interoceptive threshold and precision (Protocol 7). "
            "Pred 7.A: cathodal tDCS → θₜ increase; ketamine → θₜ decrease; shift ≥ 0.05 units (d ≥ |0.4|); "
            "slope unchanged (p > 0.10). "
            "Pred 7.B: atomoxetine steepens psychometric slope (increased Πᵉ, p < 0.05) and decreases threshold; "
            "propranolol selectively reduces interoceptive signal influence (β_SM shift, d > 0.4). "
            "Pred 7.C: Monotonic dose-response (Jonckheere–Terpstra p < 0.01). "
            "Pred 7.D: Active vs. placebo p < 0.01, BF₁₀ > 10."
        ),
        "platform": "OSF",
        "sample_size": "N ≥ 30 per intervention arm (within-participant crossover where feasible)",
        "measures": [
            "Detection threshold θₜ",
            "Psychometric slope",
            "Interoceptive signal influence (β_SM)",
            "Dose-response curves (propranolol 20/40/80 mg)",
        ],
        "analysis": (
            "Psychometric function fitting (mixed-effects); Jonckheere–Terpstra monotonic trend test; "
            "Bayesian factor analysis (BF₁₀ active vs. placebo); "
            "paired t / equivalence tests for slope and threshold"
        ),
        "falsification_criterion": (
            "Pred 7.A: Cathodal tDCS or ketamine fails to shift detection threshold ≥ 0.05 units in predicted "
            "direction; or effect in opposite direction (p < 0.01). "
            "Pred 7.B: Propranolol produces no significant reduction in interoceptive signal influence (p > 0.05); "
            "or atomoxetine fails to steepen slope AND fails to shift threshold (d < 0.2 for both). "
            "Pred 7.C: No monotonic dose-response (Jonckheere–Terpstra slope ≤ 0 or p > 0.05 for trend test). "
            "Pred 7.D: Placebo produces effect statistically indistinguishable from active intervention "
            "(equivalence test p < 0.05)."
        ),
        "prereg_status": "Pending",
        "notes": "",
    },
    "EP-8: Psychophysical Individual Differences": {
        "id": "EP-8",
        "title": "Psychophysical Individual Differences",
        "priority": 2,
        "type": "Empirical",
        "prereg_required": True,
        "depends_on": ["EP-0"],
        "status": "Not started",
        "description": (
            "Individual-difference mapping of APGI parameters (Protocol 8). "
            "Pred 8.A: r(Πⁱ, HEP) > 0.40; r(Πⁱ, heartbeat d′) > 0.35; r(Πⁱ, HRV/RMSSD) > 0.30. "
            "Pred 8.B: r(β_som, θ₀) < −0.25 (somatic facilitation — higher β_som lowers baseline threshold). "
            "Pred 8.C: θ₀ ICC > 0.75; Πⁱ ICC > 0.65; α ICC > 0.70 at 1-week retest. "
            "Pred 8.D: Factor analysis yields ≥ 2 interpretable components; |r|(β_som, β_ign) < 0.40. "
            "Pred 8.E: TAI × Πⁱ r < −0.25; BDI × θ₀ r > 0.25."
        ),
        "platform": "OSF",
        "sample_size": "N ≥ 100 for ICC reliability; N ≥ 30 per group for clinical correlates",
        "measures": [
            "Interoceptive precision (Πⁱ)",
            "Baseline threshold (θ₀)",
            "Learning rate (α)",
            "Somatic gain (β_som) and ignition bias (β_ign)",
            "HEP amplitude",
            "Heartbeat detection d′",
            "HRV/RMSSD",
            "TAI (trait anxiety inventory)",
            "BDI (Beck Depression Inventory)",
        ],
        "analysis": (
            "Pearson/Spearman correlations; ICC(2,1) test-retest reliability at 1-week interval; "
            "exploratory factor analysis; equivalence tests (β_som vs. β_ign independence)"
        ),
        "falsification_criterion": (
            "Pred 8.A: Πⁱ fails to correlate with HEP amplitude (r < 0.30 or p > 0.05). "
            "Pred 8.B: θ₀ and β_som show a positive relationship (r > 0, p < 0.05), contradicting somatic facilitation. "
            "Pred 8.C: Test-retest ICC for θ₀ falls below 0.60, indicating parameters too unstable for trait inference. "
            "Pred 8.D: All APGI parameters load onto a single factor; or |r| > 0.70 for all pairs (redundancy). "
            "Pred 8.E: No clinical correlations exceed |r| = 0.20 (p > 0.05 for all)."
        ),
        "prereg_status": "Pending",
        "notes": "",
    },
    # ── Computational Protocols ───────────────────────────────────────────────
    "EP-9: ML Classification": {
        "id": "EP-9",
        "title": "ML Classification (CP-1)",
        "priority": 2,
        "type": "Computational",
        "prereg_required": True,
        "depends_on": ["EP-3"],
        "status": "Not started",
        "description": (
            "APGI-generated synthetic neural signal classification (CP-1). "
            "Pred 9.A: classification accuracy 85–92%; AUC-ROC 0.90–0.95; F1-score 0.85–0.90. "
            "75–84% constitutes partial support only; < 75% falsified. "
            "Pred 9.B: APGI vs. GWTOnly cross-model confusion < 40%; HEP-present feature drives discriminability. "
            "Pred 9.C: Classifier trained on APGI synthetic data achieves > 55% accuracy on real human data "
            "(Melloni et al., 2007; Sergent et al., 2005). "
            "Pred 9.D: Standard PP (continuous, no threshold) does not match full APGI classification accuracy."
        ),
        "platform": "OSF",
        "sample_size": "N/A (computational) — synthetic signals from EP-3 simulation runs",
        "measures": [
            "Classification accuracy (ignition vs. no-ignition)",
            "AUC-ROC",
            "F1-score",
            "Cross-model confusion rate (APGI ↔ GWTOnly)",
            "Feature importance (HEP-present vs. HEP-absent)",
            "Cross-paradigm accuracy on real human datasets",
        ],
        "analysis": (
            "Supervised classifier (architecture pre-specified in pre-registration); "
            "pairwise accuracy comparisons (APGI vs. Standard PP); "
            "confusion matrix analysis (APGI vs. GWTOnly); "
            "cross-paradigm transfer to Melloni et al. 2007 and Sergent et al. 2005 datasets"
        ),
        "falsification_criterion": (
            "Pred 9.A: Classification accuracy < 75%; or 75–84% range not replicated above 85% with "
            "architectural tuning (intermediate zone = partial support pending replication). "
            "Pred 9.B: Confusion between APGI and GWTOnly > 40% (interoceptive precision feature not discriminating). "
            "Pred 9.C: Accuracy < 55% on real human data (APGI signals don't capture genuine neural signatures). "
            "Pred 9.D: Standard PP achieves equal or higher accuracy than full APGI."
        ),
        "prereg_status": "Pending",
        "notes": "",
    },
    "EP-10: Bayesian Model Comparison": {
        "id": "EP-10",
        "title": "Bayesian Model Comparison (CP-4)",
        "priority": 2,
        "type": "Computational",
        "prereg_required": True,
        "depends_on": ["EP-3"],
        "status": "Not started",
        "description": (
            "Bayesian model comparison across pre-registered consciousness datasets (CP-4). "
            "Pred 10.A: APGI achieves lowest WAIC/LOO; expected ΔLOO vs. SDT: +15 to +40; vs. GWT: +5 to +20; "
            "vs. Continuous: +25 to +70. "
            "Pred 10.B: Partial r(conscious report, Πⁱ | attention) > 0.25; 95% CI excludes zero. "
            "Pred 10.C: P3b predicted by (Sₜ − θₜ) not stimulus strength; R² improvement > 15%. "
            "Pred 10.D: RT shows U-shape around threshold (quadratic (Sₜ − θₜ)² term significant; RT minimum "
            "at |Sₜ − θₜ| > 0.3). "
            "Pred 10.E: BF₁₀ > 3 (APGI vs. GWT) across all pre-registered datasets."
        ),
        "platform": "OSF",
        "sample_size": "N/A (computational) — fit to pre-registered human IGT and consciousness datasets",
        "measures": [
            "WAIC / LOO-CV (leave-one-out cross-validation)",
            "ΔLOO vs. SDT, GWT-only, Continuous baselines",
            "Partial r(conscious report, Πⁱ | attention covariate)",
            "P3b ~ β₁(Sₜ − θₜ) + β₂(stimulus strength) regression",
            "RT quadratic fit around threshold proximity",
            "BF₁₀ (APGI vs. GWT)",
        ],
        "analysis": (
            "LOO-CV (loo package or equivalent); LRT for nested model comparison; "
            "partial regression (Πⁱ unique variance after attention covariate); "
            "polynomial regression RT ~ (Sₜ − θₜ)²; "
            "Bayesian factor computation across pre-registered datasets"
        ),
        "falsification_criterion": (
            "Pred 10.A: APGI has higher (worse) LOO than SDT or GWT by > 10 points on any pre-registered dataset. "
            "Pred 10.B: Πⁱ posterior includes zero in 80% credible interval (no unique variance beyond attention). "
            "Pred 10.C: P3b better predicted by stimulus strength alone; or R² improvement < 5% from (Sₜ − θₜ). "
            "Pred 10.D: RT does not show threshold-proximity U-shape; or RT is linear/monotonic in (Sₜ − θₜ). "
            "Pred 10.E: BF₁₀ < 3 across all datasets (interoceptive component adds no explanatory power over GWT)."
        ),
        "prereg_status": "Pending",
        "notes": "",
    },
    "EP-11: Active Inference Agent Computational": {
        "id": "EP-11",
        "title": "Active Inference Agent Computational (CP-3)",
        "priority": 2,
        "type": "Computational",
        "prereg_required": True,
        "depends_on": ["EP-3"],
        "status": "Not started",
        "description": (
            "Extended computational validation of APGI agent in volatile, high-cost environments (CP-3). "
            "Pred 11.A: APGI agent achieves > 70% optimal performance; convergence < 100 trials; "
            "outperforms Standard PP, GWTOnly, ActorCritic pairwise at σ_env = 0.6. "
            "Pred 11.B: ≥ 60% of significant behavioral shifts preceded by ignition within 200 ms window. "
            "Pred 11.C: Disabling θₜ causes > 5% performance change; θₜ converges to non-extreme values. "
            "Pred 11.D: β_SM converges outside [0.95, 1.05] and not near 0 in ≥ 80% of agents. "
            "Pred 11.E: C_metabolic vs. θₜ r > 0.2, p < 0.05; θₜ decreases with V_information (r > 0.2)."
        ),
        "platform": "OSF",
        "sample_size": "N/A (computational) — multiple independent agent seeds required",
        "measures": [
            "Final performance (% optimal)",
            "Convergence trial count",
            "Behavioral shift timing relative to ignition events",
            "θₜ value distribution (convergence check)",
            "β_SM value distribution (degeneracy check)",
            "C_metabolic vs. θₜ correlation across agent lifetimes",
        ],
        "analysis": (
            "Pairwise permutation tests (APGI vs. alternatives at σ_env = 0.6); "
            "causal timing analysis (ignition → behavioral shift within 200 ms); "
            "parameter convergence diagnostics; "
            "Pearson r (C_metabolic vs. θₜ)"
        ),
        "falsification_criterion": (
            "Pred 11.A: Fails to outperform alternatives; performance < 70% optimal or convergence > 100 trials. "
            "Pred 11.B: < 60% of significant behavioral shifts preceded by ignition within 200 ms. "
            "Pred 11.C: Disabling θₜ causes < 5% performance change; or θₜ converges to extreme values "
            "(near 0 or > 1.5 SD) in ≥ 90% of trials (mechanism degenerate). "
            "Pred 11.D: β_SM converges to [0.95, 1.05] (no somatic bias) or near 0 (complete interoceptive "
            "suppression) in ≥ 80% of agents. "
            "Pred 11.E: Correlation C_metabolic vs. θₜ < 0.2 (p > 0.05)."
        ),
        "prereg_status": "Pending",
        "notes": "",
    },
    "EP-12: Phase Transition Analysis": {
        "id": "EP-12",
        "title": "Phase Transition Analysis (CP-4-PT)",
        "priority": 2,
        "type": "Computational",
        "prereg_required": True,
        "depends_on": ["EP-3"],
        "status": "Not started",
        "description": (
            "Phase transition and critical slowing signatures at ignition threshold (CP-4-PT). "
            "Pred 12.A: Mean |dS/dt| discontinuity > 0.5 (Cohen's d > 0.8 vs. random timepoints). "
            "Pred 12.B: Variance ratio (near threshold / far from threshold) > 2.0. "
            "Pred 12.C: Autocorrelation ratio (near / far) > 1.5. "
            "Pred 12.D: Φ at ignition > 2.0× baseline (full confirmation); 1.3–2.0× = partial support "
            "(pre-specified indeterminate zone); < 1.3× = falsified. "
            "Pred 12.E: Hurst exponent H > 0.6 near threshold; H ≈ 0.5 far from threshold."
        ),
        "platform": "OSF",
        "sample_size": "N/A (computational) — derived from EP-3 simulation time series",
        "measures": [
            "Mean |dS/dt| at ignition threshold crossing",
            "Variance ratio (near threshold vs. far from threshold)",
            "Autocorrelation ratio (AC1 near / far)",
            "Integrated information Φ at ignition vs. baseline",
            "Hurst exponent H (near threshold vs. far from threshold)",
        ],
        "analysis": (
            "Permutation tests for |dS/dt| discontinuity; "
            "variance and AC1 ratio calculations; "
            "Φ estimation (per Innovation 11 threshold specification); "
            "Hurst exponent estimation (rescaled range or DFA)"
        ),
        "falsification_criterion": (
            "Pred 12.A: Discontinuity ≤ 0.5; or Cohen's d ≤ 0.5 against random timepoints. "
            "Pred 12.B: Susceptibility ratio < 1.2 (no critical fluctuations near threshold). "
            "Pred 12.C: Critical slowing ratio < 1.2 (continuous not discrete threshold behavior). "
            "Pred 12.D: Φ at ignition < 1.3× baseline; or Cohen's d for Φ discontinuity < 0.5. "
            "Results in 1.3–2.0× range = partial support (indeterminate zone, pre-specified). "
            "Pred 12.E: H near threshold ≤ 0.6; no significant difference in H between near and far states."
        ),
        "prereg_status": "Pending",
        "notes": "Pred 12.D partial-support zone (1.3–2.0×) is pre-specified; no amendment required to classify as partial.",
    },
    "EP-13: Evolutionary Emergence": {
        "id": "EP-13",
        "title": "Evolutionary Emergence (CP-5)",
        "priority": 3,
        "type": "Computational",
        "prereg_required": True,
        "depends_on": [],
        "status": "Not started",
        "description": (
            "Evolutionary simulation of APGI architectural component selection (CP-5). "
            "Pred 13.A: Selection coefficients — threshold mechanism > 0.02; interoceptive weighting > 0.02; "
            "somatic markers > 0.015; all positive across environments. "
            "Pred 13.B: > 80% of population carries all three components by generation 300. "
            "Pred 13.C: Full APGI outperforms all 1–2 component architectures (all pairwise fitness comparisons). "
            "Pred 13.D: Emergence order matches threshold → precision → interoceptive → somatic in ≥ 80% of runs. "
            "Pred 13.E: Threshold mechanism > 60% population frequency by generation 500. "
            "Pred 13.F: Interoceptive weighting shows monotonically increasing frequency (positive selection). "
            "Pred 13.G: Somatic markers > 50% frequency at evolutionary steady-state. "
            "Pred 13.H: Continuous (no-threshold) architectures achieve lower fitness than discrete-threshold APGI."
        ),
        "platform": "OSF",
        "sample_size": "N/A (computational) — ≥ 80% of independent simulation runs required for Pred 13.D",
        "measures": [
            "Selection coefficients (threshold, interoceptive, somatic components)",
            "Population frequency per component across generations",
            "Pairwise fitness comparisons (full vs. partial architectures)",
            "Emergence order sequence across independent runs",
            "Threshold mechanism frequency by generation 500",
            "Somatic marker frequency at steady-state",
        ],
        "analysis": (
            "Population genetics simulation; selection coefficient estimation; "
            "pairwise fitness comparisons (p < 0.05); "
            "sequence matching across ≥ 80% of independent runs (Pred 13.D)"
        ),
        "falsification_criterion": (
            "Pred 13.A: Any selection coefficient ≤ 0 (negative selection) for any core APGI component. "
            "Pred 13.B: < 80% of population carries all three components by generation 300. "
            "Pred 13.C: Partial architectures (1–2 components) achieve equal or higher fitness than full APGI. "
            "Pred 13.D: Emergence order differs significantly from predicted sequence across ≥ 50% of runs. "
            "Pred 13.E: Threshold mechanism frequency ≤ 60% by generation 500. "
            "Pred 13.F: Interoceptive weighting frequency decreases over generations (negative selection). "
            "Pred 13.G: Somatic markers never exceed 50% frequency at steady-state. "
            "Pred 13.H: Continuous architectures achieve equal or higher fitness than APGI."
        ),
        "prereg_status": "Pending",
        "notes": "",
    },
    "EP-14: RNN Architectures": {
        "id": "EP-14",
        "title": "RNN Architectures with APGI Inductive Biases (CP-6)",
        "priority": 2,
        "type": "Computational",
        "prereg_required": True,
        "depends_on": ["EP-3"],
        "status": "Not started",
        "description": (
            "APGI-constrained RNN vs. LSTM, Attention, and MLP baselines (CP-6). "
            "Pred 14.A: APGI RNN AUC 0.85–0.92 (vs. LSTM 0.70–0.78; Attention 0.75–0.82); superiority > 2% AUC. "
            "Pred 14.B: APGI RNN converges in 30–50% fewer epochs than comparison networks on interoceptive tasks. "
            "Pred 14.C: Learned Πⁱ increases when interoceptive information is task-relevant; Πᵉ increases "
            "when exteroceptive information is task-relevant (task × precision interaction p < 0.05). "
            "Pred 14.D: Ignition probability vs. response accuracy r > 0.5; threshold converges to non-extreme "
            "values in ≥ 90% of training runs."
        ),
        "platform": "OSF",
        "sample_size": "N/A (computational) — multiple training seeds required",
        "measures": [
            "AUC (conscious/unconscious classification)",
            "Epochs to convergence",
            "Learned precision weights (Πⁱ, Πᵉ) across task conditions",
            "Ignition probability vs. response accuracy correlation",
            "Threshold parameter convergence distribution",
        ],
        "analysis": (
            "AUC comparison across architectures; paired epoch-count comparison; "
            "task × precision interaction (Πⁱ and Πᵉ across interoceptive vs. exteroceptive task variants); "
            "Pearson r (ignition probability vs. response accuracy); "
            "threshold convergence diagnostics"
        ),
        "falsification_criterion": (
            "Pred 14.A: No AUC advantage over standard LSTM (within 2%); APGI architectural constraints "
            "provide no classification benefit. "
            "Pred 14.B: APGI RNN requires equal or more epochs to converge than comparison networks. "
            "Pred 14.C: Learned Πⁱ does not increase when interoceptive information is relevant "
            "(no significant task × Πⁱ interaction); or Πᵉ does not track exteroceptive task relevance. "
            "Note: falsification targets Πⁱ/Πᵉ precision weights — NOT β_SM. "
            "Pred 14.D: Threshold converges to extreme values (0 or ∞) in ≥ 50% of training runs (mechanism "
            "degenerate); or attention-only network achieves equal or higher AUC than APGI (explicit ignition "
            "gate unnecessary)."
        ),
        "prereg_status": "Pending",
        "notes": "",
    },
}

# Status options for dropdown
_STATUS_OPTIONS = [
    "Not started",
    "In preparation",
    "Pre-registered",
    "Data collection",
    "Analysis",
    "Completed",
    "On hold",
]
_PREREG_OPTIONS = ["Pending", "Draft ready", "Submitted", "Registered", "Not applicable"]

# Type badge colours
_TYPE_COLORS: Dict[str, str] = {
    "Empirical": "#2874a6",
    "Computational": "#155724",
    "Clinical": "#721c24",
    "Clinical-Empirical": "#5a0080",
}

# Priority labels
_PRIORITY_LABELS = {1: "Priority 1 — High", 2: "Priority 2 — Medium", 3: "Priority 3 — Long-term"}


# ── Main Application ──────────────────────────────────────────────────────────


class OSFProtocolGUI:
    """GUI for APGI Open Science Framework — EP-0 through EP-14 prediction registry management."""

    def __init__(self, root: tk.Tk, headless: bool = False) -> None:
        self.root = root
        self.headless = headless
        self.gui_queue: queue.Queue[Any] = queue.Queue()
        self._protocol_status: Dict[str, Dict[str, str]] = {
            key: {"status": meta["status"], "prereg_status": meta["prereg_status"], "notes": meta["notes"]}
            for key, meta in EP_PROTOCOLS.items()
        }
        self._selected_protocol: Optional[str] = None
        self._console_visible = True
        self._card_widgets: List[tk.Widget] = []
        self._detail_vars: Dict[str, tk.StringVar] = {}
        self.stop_event = threading.Event()
        self.running_thread: Optional[threading.Thread] = None

        if not self.headless:
            apply_apgi_theme(self.root)
            self._process_gui_queue()
            self.root.title("APGI Open Science Framework — Prediction Registry (EP-0 – EP-14)")
            self.root.geometry("1020x740")
            self.root.minsize(800, 560)
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            self._create_menu_bar()
            self._bind_shortcuts()
            self.setup_ui()

        _proj_root = os.path.dirname(os.path.abspath(__file__))
        if _proj_root not in sys.path:
            sys.path.insert(0, _proj_root)

    # ── Queue ─────────────────────────────────────────────────────────────────

    def _process_gui_queue(self) -> None:
        try:
            while True:
                fn = self.gui_queue.get_nowait()
                if callable(fn):
                    try:
                        fn()
                    except Exception as exc:
                        logger.warning("GUI queue update failed: %s", exc)
        except queue.Empty:
            pass
        try:
            self.root.after(100, self._process_gui_queue)
        except (KeyboardInterrupt, TclError):
            logger.info("GUI processing interrupted.")
            self.stop_event.set()

    # ── Menu & shortcuts ──────────────────────────────────────────────────────

    def _create_menu_bar(self) -> None:
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export Protocol Report", command=self.export_protocol_report, accelerator="Ctrl+E")
        file_menu.add_command(
            label="Generate Pre-reg Template", command=self.generate_prereg_template, accelerator="Ctrl+G"
        )
        file_menu.add_separator()
        file_menu.add_command(label="Save Status Snapshot", command=self.save_status_snapshot, accelerator="Ctrl+S")
        file_menu.add_command(label="Load Status Snapshot", command=self.load_status_snapshot)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self._on_closing, accelerator="Ctrl+Q")

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Clear Console", command=self.clear_console, accelerator="Ctrl+L")
        view_menu.add_command(label="Toggle Console", command=self._toggle_console)
        view_menu.add_command(label="Dependency Overview", command=self._show_dependency_overview)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-q>", lambda e: self._on_closing())
        self.root.bind("<Command-q>", lambda e: self._on_closing())
        self.root.bind("<Control-e>", lambda e: self.export_protocol_report())
        self.root.bind("<Control-g>", lambda e: self.generate_prereg_template())
        self.root.bind("<Control-s>", lambda e: self.save_status_snapshot())
        self.root.bind("<Control-l>", lambda e: self.clear_console())

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About",
            "APGI Open Science Framework GUI\nPrediction Registry Manager (EP-0 – EP-14)\nVersion 2.0.0",
        )

    # ── UI Construction ───────────────────────────────────────────────────────

    def setup_ui(self) -> None:
        """Build the application layout per APGI design spec."""
        # ── Row 0: Metric bar ─────────────────────────────────────────────────
        metric_bar = tk.Frame(self.root, bg=_BLUE, pady=8, padx=15)
        metric_bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        metric_bar.columnconfigure(1, weight=1)

        tk.Label(
            metric_bar,
            text="APGI OPEN SCIENCE FRAMEWORK  —  EMPIRICAL PROTOCOLS",
            bg=_BLUE,
            fg="white",
            font=("Noto Sans", 13, "bold"),
        ).grid(row=0, column=0, sticky="w")

        # Summary counters
        self._counter_var = tk.StringVar(value=self._build_counter_text())
        tk.Label(metric_bar, textvariable=self._counter_var, bg=_BLUE, fg="white", font=("Noto Sans", 10)).grid(
            row=0, column=1, padx=(20, 15), sticky="e"
        )

        # OSF status indicator
        self._osf_status_var = tk.StringVar(value="ℹ  OSF: Ready")
        tk.Label(metric_bar, textvariable=self._osf_status_var, bg=_BLUE, fg="white", font=("Noto Sans", 10)).grid(
            row=0, column=2, sticky="e"
        )

        # ── Grid weights ──────────────────────────────────────────────────────
        self.root.columnconfigure(0, weight=0, minsize=220)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)
        self.root.rowconfigure(3, weight=0, minsize=160)

        # ── Row 1, Col 0: Sidebar ─────────────────────────────────────────────
        sidebar = tk.Frame(self.root, bg=_SURFACE, width=220, bd=0)
        sidebar.grid(row=1, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(1, weight=1)

        tk.Label(
            sidebar, text="EMPIRICAL PROTOCOLS", bg=_BORDER, fg=_MUTED, font=("Noto Sans", 9, "bold"), pady=5
        ).grid(row=0, column=0, columnspan=2, sticky="ew")

        lb_frame = tk.Frame(sidebar, bg=_SURFACE)
        lb_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        lb_frame.columnconfigure(0, weight=1)
        lb_frame.rowconfigure(0, weight=1)

        self.protocol_listbox = tk.Listbox(
            lb_frame,
            font=("Noto Sans", 9),
            bg=_SURFACE,
            fg=_FG,
            selectbackground=_BLUE,
            selectforeground="white",
            borderwidth=0,
            highlightthickness=1,
            highlightcolor=_BORDER,
            highlightbackground=_BORDER,
            activestyle="none",
            cursor="hand2",
        )
        lb_scroll = ttk.Scrollbar(lb_frame, orient="vertical", command=self.protocol_listbox.yview)
        self.protocol_listbox.configure(yscrollcommand=lb_scroll.set)
        self.protocol_listbox.grid(row=0, column=0, sticky="nsew")
        lb_scroll.grid(row=0, column=1, sticky="ns")

        for key in EP_PROTOCOLS:
            ep_id = EP_PROTOCOLS[key]["id"]
            short_title = EP_PROTOCOLS[key]["title"]
            self.protocol_listbox.insert(tk.END, f"  {ep_id}  {short_title}")

        self.protocol_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        # Sidebar action buttons
        btn_area = tk.Frame(sidebar, bg=_SURFACE, pady=8, padx=8)
        btn_area.grid(row=2, column=0, columnspan=2, sticky="ew")
        btn_area.columnconfigure(0, weight=1)

        self._gen_prereg_btn = ttk.Button(
            btn_area,
            text="✎  Generate Pre-reg",
            command=self.generate_prereg_template,
            style="Primary.TButton",
            state=tk.DISABLED,
        )
        self._gen_prereg_btn.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        self._export_btn = ttk.Button(
            btn_area,
            text="Export Report",
            command=self.export_protocol_report,
            style="Secondary.TButton",
        )
        self._export_btn.grid(row=1, column=0, sticky="ew", pady=(0, 4))

        ttk.Separator(btn_area, orient="horizontal").grid(row=2, column=0, sticky="ew", pady=(0, 6))

        ttk.Button(btn_area, text="Dependency Overview", command=self._show_dependency_overview).grid(
            row=3, column=0, sticky="ew", pady=(0, 4)
        )
        ttk.Button(btn_area, text="Clear Console", command=self.clear_console).grid(row=4, column=0, sticky="ew")

        # Right border divider
        tk.Frame(self.root, bg=_BORDER, width=1).grid(row=1, column=0, sticky="nse")

        # ── Row 1, Col 1: Workspace ───────────────────────────────────────────
        workspace = ttk.Frame(self.root, padding=(15, 10, 15, 10))
        workspace.grid(row=1, column=1, sticky="nsew")
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(0, weight=0)  # cards row
        workspace.rowconfigure(1, weight=1)  # detail/notes row

        self._card_area = ttk.Frame(workspace)
        self._card_area.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self._card_area.columnconfigure(0, weight=1)
        self._show_workspace_empty_state()

        # Detail / status panel
        self._detail_frame = ttk.LabelFrame(workspace, text="PROTOCOL DETAILS & STATUS", padding=(10, 6))
        self._detail_frame.grid(row=1, column=0, sticky="nsew")
        self._detail_frame.columnconfigure(0, weight=1)
        self._detail_frame.rowconfigure(0, weight=1)

        self._detail_inner = ttk.Frame(self._detail_frame)
        self._detail_inner.grid(row=0, column=0, sticky="nsew")
        self._detail_inner.columnconfigure(0, weight=1)
        self._detail_inner.rowconfigure(0, weight=1)

        ttk.Label(
            self._detail_inner,
            text="Select a protocol from the sidebar to view details and manage pre-registration.",
            foreground=_MUTED,
            font=("Noto Sans", 10),
        ).grid(row=0, column=0, padx=20, pady=20)

        # ── Row 2: Console toggle header ──────────────────────────────────────
        console_header = tk.Frame(self.root, bg=_BORDER, pady=3)
        console_header.grid(row=2, column=0, columnspan=2, sticky="ew")

        self._console_toggle_btn = tk.Button(
            console_header,
            text="▼  OUTPUT CONSOLE",
            bg=_BORDER,
            fg=_MUTED,
            font=("Noto Sans", 9, "bold"),
            relief="flat",
            cursor="hand2",
            command=self._toggle_console,
            activebackground=_BORDER,
            activeforeground="#495057",
            bd=0,
        )
        self._console_toggle_btn.pack(side="left", padx=10)

        # ── Row 3: Console body ───────────────────────────────────────────────
        self.root.rowconfigure(3, weight=0, minsize=160)
        self._console_frame = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        self._console_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        self._console_frame.columnconfigure(0, weight=1)
        self._console_frame.rowconfigure(0, weight=1)

        self.console = scrolledtext.ScrolledText(
            self._console_frame,
            height=8,
            font=("Noto Sans Mono", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white",
            state=tk.DISABLED,
            wrap=tk.WORD,
        )
        self.console.grid(row=0, column=0, sticky="nsew")

        self._log("OSF Protocol Manager initialised. Select a protocol to begin.")
        self._log(f"{len(EP_PROTOCOLS)} protocols loaded  |  All status: Not started  |  Pre-reg: Pending")

    # ── Workspace helpers ─────────────────────────────────────────────────────

    def _show_workspace_empty_state(self) -> None:
        for w in self._card_area.winfo_children():
            w.destroy()

        frame = ttk.Frame(self._card_area, padding=30)
        frame.grid(row=0, column=0)
        tk.Canvas(frame, width=200, height=80, bg=_BG, highlightbackground=_SOFT, highlightthickness=2).pack()
        ttk.Label(
            frame,
            text="No protocol selected.  Choose one from the sidebar to view metadata and manage pre-registration.",
            wraplength=400,
            font=("Noto Sans", 11),
            foreground=_MUTED,
        ).pack(pady=(15, 0))

    def _show_protocol_cards(self, key: str) -> None:
        """Populate the card area with summary cards for the selected protocol."""
        for w in self._card_area.winfo_children():
            w.destroy()

        meta = EP_PROTOCOLS[key]
        st = self._protocol_status[key]

        cards_row = ttk.Frame(self._card_area)
        cards_row.grid(row=0, column=0, sticky="ew")
        for i in range(4):
            cards_row.columnconfigure(i, weight=1)

        APGICard(cards_row, "Protocol", meta["id"], f"Type: {meta['type']}  |  Priority: {meta['priority']}").grid(
            row=0, column=0, padx=(0, 6), pady=4, sticky="ew"
        )

        prereq_text = ", ".join(meta["depends_on"]) if meta["depends_on"] else "None"
        APGICard(cards_row, "Dependencies", prereq_text, "Must be completed before this protocol").grid(
            row=0, column=1, padx=(0, 6), pady=4, sticky="ew"
        )

        APGICard(cards_row, "Study Status", st["status"], "Update via the status panel below").grid(
            row=0, column=2, padx=(0, 6), pady=4, sticky="ew"
        )

        APGICard(
            cards_row,
            "Pre-registration",
            st["prereg_status"],
            f"Platform: {meta['platform']}  |  Required: {'Yes' if meta['prereg_required'] else 'No'}",
        ).grid(row=0, column=3, padx=0, pady=4, sticky="ew")

    def _populate_detail_panel(self, key: str) -> None:
        """Build the scrollable detail/status editor for the selected protocol."""
        for w in self._detail_inner.winfo_children():
            w.destroy()

        meta = EP_PROTOCOLS[key]
        st = self._protocol_status[key]

        # Scrollable canvas
        canvas = tk.Canvas(self._detail_inner, bg=_BG, highlightthickness=0)
        vsb = ttk.Scrollbar(self._detail_inner, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        self._detail_inner.rowconfigure(0, weight=1)
        self._detail_inner.columnconfigure(0, weight=1)

        scrollable = ttk.Frame(canvas, padding=(10, 6))
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        scrollable.columnconfigure(1, weight=1)

        def _row(r: int, label: str, value: str, mono: bool = False) -> None:
            ttk.Label(scrollable, text=label + ":", font=("Noto Sans", 9, "bold"), foreground=_MUTED).grid(
                row=r, column=0, sticky="nw", padx=(0, 12), pady=2
            )
            fnt = ("Noto Sans Mono", 9) if mono else ("Noto Sans", 10)
            ttk.Label(scrollable, text=value, font=fnt, foreground=_FG, wraplength=560, justify="left").grid(
                row=r, column=1, sticky="nw", pady=2
            )

        row_idx = 0

        # Title
        ttk.Label(scrollable, text=meta["title"], font=("Noto Sans", 13, "bold"), foreground=_FG).grid(
            row=row_idx, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )
        row_idx += 1

        _row(row_idx, "Description", meta["description"])
        row_idx += 1
        _row(row_idx, "Study Type", meta["type"])
        row_idx += 1
        _row(row_idx, "Priority", _PRIORITY_LABELS.get(meta["priority"], str(meta["priority"])))
        row_idx += 1
        _row(row_idx, "Sample Size", meta["sample_size"])
        row_idx += 1
        _row(row_idx, "Key Measures", "  •  " + "\n  •  ".join(meta["measures"]))
        row_idx += 1
        _row(row_idx, "Analysis Plan", meta["analysis"])
        row_idx += 1
        _row(row_idx, "Falsification Criterion", meta["falsification_criterion"], mono=True)
        row_idx += 1

        if meta["notes"]:
            _row(row_idx, "Notes", meta["notes"])
            row_idx += 1

        ttk.Separator(scrollable, orient="horizontal").grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=8)
        row_idx += 1

        # ── Editable status fields ────────────────────────────────────────────
        ttk.Label(scrollable, text="UPDATE STATUS", font=("Noto Sans", 10, "bold"), foreground=_MUTED).grid(
            row=row_idx, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        row_idx += 1

        # Study status
        ttk.Label(scrollable, text="Study Status:", font=("Noto Sans", 9, "bold"), foreground=_MUTED).grid(
            row=row_idx, column=0, sticky="w", pady=2
        )
        status_var = tk.StringVar(value=st["status"])
        self._detail_vars["status"] = status_var
        status_cb = ttk.Combobox(
            scrollable, textvariable=status_var, values=_STATUS_OPTIONS, state="readonly", width=28
        )
        status_cb.grid(row=row_idx, column=1, sticky="w", pady=2)
        row_idx += 1

        # Pre-reg status
        ttk.Label(scrollable, text="Pre-reg Status:", font=("Noto Sans", 9, "bold"), foreground=_MUTED).grid(
            row=row_idx, column=0, sticky="w", pady=2
        )
        prereg_var = tk.StringVar(value=st["prereg_status"])
        self._detail_vars["prereg_status"] = prereg_var
        prereg_cb = ttk.Combobox(
            scrollable, textvariable=prereg_var, values=_PREREG_OPTIONS, state="readonly", width=28
        )
        prereg_cb.grid(row=row_idx, column=1, sticky="w", pady=2)
        row_idx += 1

        # Notes
        ttk.Label(scrollable, text="Notes:", font=("Noto Sans", 9, "bold"), foreground=_MUTED).grid(
            row=row_idx, column=0, sticky="nw", pady=2
        )
        notes_text = tk.Text(
            scrollable,
            height=3,
            width=55,
            font=("Noto Sans", 10),
            wrap=tk.WORD,
            bg=_SURFACE,
            fg=_FG,
            relief="solid",
            bd=1,
        )
        notes_text.insert("1.0", st["notes"])
        notes_text.grid(row=row_idx, column=1, sticky="ew", pady=2)
        self._notes_widget = notes_text
        row_idx += 1

        # Save button
        def _save() -> None:
            self._protocol_status[key]["status"] = status_var.get()
            self._protocol_status[key]["prereg_status"] = prereg_var.get()
            self._protocol_status[key]["notes"] = notes_text.get("1.0", tk.END).strip()
            self._show_protocol_cards(key)
            self._counter_var.set(self._build_counter_text())
            self._log(f"[{meta['id']}] Status saved: {status_var.get()} | Pre-reg: {prereg_var.get()}")
            messagebox.showinfo("Saved", f"{meta['id']} status updated.")

        ttk.Button(scrollable, text="Save Changes", command=_save, style="Primary.TButton").grid(
            row=row_idx, column=1, sticky="w", pady=8
        )

    # ── Sidebar selection ─────────────────────────────────────────────────────

    def _on_listbox_select(self, _event: Any = None) -> None:
        sel = self.protocol_listbox.curselection()
        if not sel:
            return
        key = list(EP_PROTOCOLS.keys())[sel[0]]
        self._selected_protocol = key
        self._gen_prereg_btn.configure(state=tk.NORMAL)
        meta = EP_PROTOCOLS[key]
        self._log(f"Selected: {meta['id']} — {meta['title']}")
        self._show_protocol_cards(key)
        self._populate_detail_panel(key)

    # ── Counter text ──────────────────────────────────────────────────────────

    def _build_counter_text(self) -> str:
        total = len(EP_PROTOCOLS)
        registered = sum(1 for s in self._protocol_status.values() if s["prereg_status"] == "Registered")
        completed = sum(1 for s in self._protocol_status.values() if s["status"] == "Completed")
        return f"Protocols: {total}  |  Pre-registered: {registered}/{total}  |  Completed: {completed}/{total}"

    # ── Dependency Overview ───────────────────────────────────────────────────

    def _show_dependency_overview(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Dependency Overview — APGI Empirical Protocols")
        win.geometry("560x480")
        win.configure(bg=_BG)
        win.resizable(True, True)

        tk.Label(
            win, text="PROTOCOL DEPENDENCY GRAPH", bg=_BLUE, fg="white", font=("Noto Sans", 12, "bold"), pady=8
        ).pack(fill="x", padx=0)

        frame = ttk.Frame(win, padding=15)
        frame.pack(fill="both", expand=True)

        lines: List[str] = []
        lines.append("Execution Order & Dependencies\n")
        lines.append("=" * 48)
        for key, meta in EP_PROTOCOLS.items():
            dep_str = " ← requires: " + ", ".join(meta["depends_on"]) if meta["depends_on"] else " (no dependencies)"
            st = self._protocol_status[key]["status"]
            prereg = self._protocol_status[key]["prereg_status"]
            lines.append(
                f"\n  {meta['id']:6s}  {meta['title']}"
                f"\n          Type: {meta['type']}  |  Priority: {meta['priority']}"
                f"\n          Depends on:{dep_str}"
                f"\n          Status: {st}  |  Pre-reg: {prereg}"
            )
        lines.append("\n" + "=" * 48)

        text = scrolledtext.ScrolledText(
            frame, font=("Noto Sans Mono", 9), bg=_SURFACE, fg=_FG, state=tk.DISABLED, wrap=tk.WORD
        )
        text.pack(fill="both", expand=True)
        text.configure(state=tk.NORMAL)
        text.insert(tk.END, "\n".join(lines))
        text.configure(state=tk.DISABLED)

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(8, 0))

    # ── Pre-registration template ─────────────────────────────────────────────

    def generate_prereg_template(self) -> None:
        key = self._selected_protocol
        if not key:
            messagebox.showwarning("No Protocol Selected", "Please select a protocol from the sidebar first.")
            return

        meta = EP_PROTOCOLS[key]
        st = self._protocol_status[key]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        lines: List[str] = [
            "=" * 64,
            "APGI PRE-REGISTRATION TEMPLATE",
            f"Generated: {timestamp}",
            "=" * 64,
            "",
            f"Protocol ID:        {meta['id']}",
            f"Title:              {meta['title']}",
            f"Study Type:         {meta['type']}",
            f"Priority:           {_PRIORITY_LABELS.get(meta['priority'], str(meta['priority']))}",
            f"Platform:           {meta['platform']}",
            f"Current Status:     {st['status']}",
            f"Pre-reg Status:     {st['prereg_status']}",
            "",
            "─" * 64,
            "1. RESEARCH QUESTION & HYPOTHESES",
            "─" * 64,
            "",
            meta["description"],
            "",
            "─" * 64,
            "2. EXPERIMENTAL DESIGN",
            "─" * 64,
            "",
            f"Sample Size:  {meta['sample_size']}",
            "",
            "Key Measures:",
        ]
        for m in meta["measures"]:
            lines.append(f"  • {m}")

        lines += [
            "",
            "─" * 64,
            "3. ANALYSIS PLAN",
            "─" * 64,
            "",
            meta["analysis"],
            "",
            "─" * 64,
            "4. FALSIFICATION CRITERIA",
            "─" * 64,
            "",
            meta["falsification_criterion"],
            "",
            "─" * 64,
            "5. DEPENDENCIES",
            "─" * 64,
            "",
        ]

        if meta["depends_on"]:
            for dep in meta["depends_on"]:
                dep_key = next((k for k in EP_PROTOCOLS if EP_PROTOCOLS[k]["id"] == dep), None)
                dep_title = EP_PROTOCOLS[dep_key]["title"] if dep_key else dep
                lines.append(f"  Requires: {dep} — {dep_title}")
        else:
            lines.append("  None — this protocol has no upstream dependencies.")

        lines += [
            "",
            "─" * 64,
            "6. NOTES",
            "─" * 64,
            "",
            st["notes"] if st["notes"] else "[No notes recorded]",
            "",
            "=" * 64,
            "END OF PRE-REGISTRATION TEMPLATE",
            "=" * 64,
        ]

        template_text = "\n".join(lines)

        # Show in a new window and offer save
        win = tk.Toplevel(self.root)
        win.title(f"Pre-registration Template — {meta['id']}")
        win.geometry("680x560")
        win.configure(bg=_BG)

        tk.Label(
            win,
            text=f"PRE-REGISTRATION TEMPLATE  —  {meta['id']}",
            bg=_BLUE,
            fg="white",
            font=("Noto Sans", 12, "bold"),
            pady=8,
        ).pack(fill="x")

        txt = scrolledtext.ScrolledText(win, font=("Noto Sans Mono", 9), bg=_SURFACE, fg=_FG, wrap=tk.WORD)
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        txt.insert(tk.END, template_text)

        def _save_file() -> None:
            path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialfile=f"prereg_{meta['id'].replace('-', '_')}.txt",
            )
            if path:
                Path(path).write_text(template_text, encoding="utf-8")
                self._log(f"Pre-reg template saved: {path}")
                messagebox.showinfo("Saved", f"Template saved to:\n{path}")

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(btn_frame, text="Save to File…", command=_save_file, style="Primary.TButton").pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side="left")

        self._log(f"[{meta['id']}] Pre-registration template generated.")

    # ── Export report ─────────────────────────────────────────────────────────

    def export_protocol_report(self) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines: List[str] = [
            "=" * 68,
            "APGI OPEN SCIENCE FRAMEWORK — EMPIRICAL PROTOCOL STATUS REPORT",
            f"Generated: {timestamp}",
            "=" * 68,
            "",
        ]

        for key, meta in EP_PROTOCOLS.items():
            st = self._protocol_status[key]
            dep_str = ", ".join(meta["depends_on"]) if meta["depends_on"] else "None"
            lines += [
                f"{'─' * 68}",
                f"  {meta['id']}  |  {meta['title']}",
                f"  Type: {meta['type']}   Priority: {meta['priority']}   Platform: {meta['platform']}",
                f"  Depends on: {dep_str}",
                f"  Study status:    {st['status']}",
                f"  Pre-reg status:  {st['prereg_status']}",
                f"  Notes: {st['notes'] if st['notes'] else '—'}",
                "",
            ]

        lines += [
            "=" * 68,
            "END OF REPORT",
            "=" * 68,
        ]

        report_text = "\n".join(lines)
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="apgi_osf_protocol_report.txt",
        )
        if not path:
            return

        if path.endswith(".json"):
            data = {
                "generated": timestamp,
                "protocols": {key: {**EP_PROTOCOLS[key], **self._protocol_status[key]} for key in EP_PROTOCOLS},
            }
            Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        else:
            Path(path).write_text(report_text, encoding="utf-8")

        self._log(f"Protocol report exported: {path}")
        messagebox.showinfo("Exported", f"Report saved to:\n{path}")

    # ── Status snapshot (persistence) ────────────────────────────────────────

    def save_status_snapshot(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="osf_status_snapshot.json",
        )
        if not path:
            return
        Path(path).write_text(json.dumps(self._protocol_status, indent=2), encoding="utf-8")
        self._log(f"Status snapshot saved: {path}")

    def load_status_snapshot(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            for key in EP_PROTOCOLS:
                if key in data:
                    for field in ("status", "prereg_status", "notes"):
                        if field in data[key]:
                            self._protocol_status[key][field] = data[key][field]
            self._counter_var.set(self._build_counter_text())
            self._log(f"Status snapshot loaded: {path}")
            # Refresh workspace if a protocol is selected
            if self._selected_protocol:
                self._show_protocol_cards(self._selected_protocol)
                self._populate_detail_panel(self._selected_protocol)
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            messagebox.showerror("Load Failed", f"Could not load snapshot:\n{exc}")

    # ── Console ───────────────────────────────────────────────────────────────

    def _log(self, message: str) -> None:
        if self.headless:
            logger.info(message)
            return

        def _do() -> None:
            self.console.configure(state=tk.NORMAL)
            ts = datetime.now().strftime("%H:%M:%S")
            self.console.insert(tk.END, f"[{ts}]  {message}\n")
            self.console.see(tk.END)
            self.console.configure(state=tk.DISABLED)

        try:
            self.gui_queue.put_nowait(_do)
        except queue.Full:
            pass

    def clear_console(self) -> None:
        self.console.configure(state=tk.NORMAL)
        self.console.delete("1.0", tk.END)
        self.console.configure(state=tk.DISABLED)

    def _toggle_console(self) -> None:
        if self._console_visible:
            self._console_frame.grid_remove()
            self.root.rowconfigure(3, minsize=0)
            self._console_toggle_btn.configure(text="▶  OUTPUT CONSOLE")
            self._console_visible = False
        else:
            self._console_frame.grid()
            self.root.rowconfigure(3, minsize=160)
            self._console_toggle_btn.configure(text="▼  OUTPUT CONSOLE")
            self._console_visible = True

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _on_closing(self) -> None:
        self.stop_event.set()
        if self.running_thread and self.running_thread.is_alive():
            self.running_thread.join(timeout=2.0)
        self.root.destroy()


# ── Entry Point ───────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="APGI Open Science Framework GUI")
    parser.add_argument("--headless", action="store_true", help="Run without GUI (for testing)")
    args = parser.parse_args()

    if args.headless:
        _ = OSFProtocolGUI(tk.Tk(), headless=True)
        print("OSF GUI initialised in headless mode.")
        print(f"Protocols loaded: {list(EP_PROTOCOLS.keys())}")
        return

    root = tk.Tk()
    _ = OSFProtocolGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
