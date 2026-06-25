"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1

All numeric thresholds are sourced verbatim from the APGI Framework Prediction Registry.
Field names follow the pattern pred_<protocol_code>_<metric> so every value is traceable
to its prediction code (Pred 0.A, Pred 2.A, etc.).
"""

import logging
from dataclasses import asdict, dataclass
from typing import Dict

try:
    from .constants import IGNITION_THRESHOLD  # type: ignore[attr-defined]
    from .constants import ParameterBounds  # type: ignore[misc]
except ImportError:

    @dataclass
    class ParameterBounds:  # type: ignore[no-redef]
        """Fallback parameter bounds."""

        precision_min: float = 0.1
        precision_max: float = 10.0
        tau_min: float = 0.01
        tau_max: float = 100.0
        alpha_min: float = 0.0
        alpha_max: float = 15.0
        gamma_min: float = -1.0
        gamma_max: float = 1.0
        rho_min: float = 0.0
        rho_max: float = 1.0
        sigma_min: float = 0.0
        sigma_max: float = 2.0

    IGNITION_THRESHOLD = 0.5

logger = logging.getLogger(__name__)


@dataclass
class FalsificationThresholds:
    """
    Centralized falsification thresholds matching the APGI Prediction Registry.

    Each field is sourced from a specific prediction code. The docstring comment on each
    field quotes the registry criterion verbatim so values can be audited at a glance.
    """

    # ── EP-0 (Protocol 0) — HEP Proxy Validation ─────────────────────────────
    # Pred 0.A: r(HEP, interoceptive d′) independent of any EEG measure
    pred_0a_r_confirm: float = 0.35  # r(HEP, interoceptive d′) > 0.35, p < 0.01
    pred_0a_r_replicate: float = 0.25  # replicated r ≥ 0.25 in validation subsample (N ≥ 30)
    pred_0a_r_falsify: float = 0.20  # r < 0.20 across two independent samples

    # Pred 0.B: physostigmine (ACh elevation) increases HEP amplitude
    pred_0b_hep_increase_pct: float = 15.0  # HEP amplitude increases ≥ 15% vs. placebo
    pred_0b_cohens_d: float = 0.50  # Cohen's d ≥ 0.50
    pred_0b_bf10: float = 10.0  # BF₁₀ ≥ 10

    # Pred 0.C: anterior insula (aINS) BOLD tracks HEP amplitude trial-by-trial
    pred_0c_hep_ins_r: float = 0.30  # r(HEP amplitude, aINS BOLD) > 0.30 within participants

    # ── EP-1 (Protocol 1) — EEG Interoceptive Precision Gating ───────────────
    # Pred 1.A: state-level Πⁱ_eff → P3b
    pred_1a_p3b_eta2: float = 0.06  # ηₚ² ≥ 0.06 (FWE-corrected P3b condition effect)
    pred_1a_hep_p3b_partial_r: float = 0.40  # HEP predicts P3b: partial r > 0.4 (after RMSSD, pupil control)
    pred_1a_hep_p3b_r_falsify: float = 0.20  # HEP–P3b partial r < 0.20 after arousal covariate = falsify

    # Pred 1.A-trait: trait-level Πⁱ_baseline → d′
    pred_1a_trait_cohens_d: float = 0.50  # Cohen's d ≥ 0.5 (high-IA d′ > low-IA d′)

    # Pred 1.B: cardiac-phase-locked detection advantage (diastole 300–500 ms vs. systole 0–150 ms)
    pred_1b_diastole_advantage_pct: float = 5.0  # hit rate advantage < 5% across two samples = falsify

    # Pred 1.C: top-tertile IA × condition interaction on P3b
    pred_1c_p3b_eta2: float = 0.08  # accuracy × condition interaction: ηₚ² ≥ 0.08

    # ── EP-2 (Protocol 4) — Causal TMS Insula/dlPFC ──────────────────────────
    # Pred 4.A: pIC and frontoparietal TMS reduce PCI via dissociable mechanisms
    pred_4a_pic_pci_reduction_pct: float = 20.0  # pIC TMS reduces PCI ~20% vs. vertex
    pred_4a_dlpfc_pci_reduction_min_pct: float = 15.0  # dlPFC/PPC TMS: 15–25% PCI reduction
    pred_4a_dlpfc_pci_reduction_max_pct: float = 25.0

    # Pred 4.B: pIC TMS selectively disrupts HEP–P3b coupling
    pred_4b_hep_p3b_coupling_falsify: float = 0.15  # pIC TMS reduces coupling to < 0.15
    pred_4b_hep_p3b_baseline: float = 0.35  # vertex baseline > 0.35
    pred_4b_exteroceptive_p3b_within_pct: float = 10.0  # exteroceptive P3b within 10% of vertex
    pred_4b_dlpfc_bf01: float = 6.0  # dlPFC leaves HEP unaffected: BF₀₁ ≥ 6, ROPE d=[−0.15,+0.15]

    # Pred 4.C: high-baseline IA × TMS-site interaction on PCI
    pred_4c_eta2: float = 0.10  # accuracy × TMS-site: η² > 0.10

    # ── EP-3 (Protocol 2) — Active Inference Agent Simulations ───────────────
    # Pred 2.A: full APGI converges within 50–80 trials, outperforms GNWT-only and Standard PP
    pred_2a_convergence_trials_min: float = 50.0  # convergence within 50–80 trials
    pred_2a_convergence_trials_max: float = 80.0
    pred_2a_reward_ratio: float = 1.15  # APGI/β_SM-lesion cumulative reward > 1.15

    # Pred 2.B: post-ignition entropy and interoceptive dominance
    pred_2b_interoceptive_pct_min: float = 70.0  # 70–85% of ignition events satisfy Πⁱ·|zⁱ| > Πᵉ·|zᵉ|
    pred_2b_interoceptive_pct_max: float = 85.0
    pred_2b_interoceptive_pct_falsify: float = 60.0  # < 60% = interoceptive dominance independently falsified

    # Pred 2.C: somatic marker retrieval M̂ temporally precedes θₜ crossing
    pred_2c_kendall_tau: float = 0.30  # Kendall τ > 0.3 at negative lag
    pred_2c_lead_pct: float = 75.0  # M̂ leads θₜ crossing in ≥ 75% of ignition events

    # Pred 2.E: APGI generative model achieves lower BIC than alternatives
    pred_2e_bic_delta: float = 10.0  # ΔBIC ≥ 10 vs. Standard PP and vs. GNWT-only

    # ── EP-4 (Protocol 6) — Disorders of Consciousness Biomarker ─────────────
    # Pred 6.A: joint HEP + PCI model explains variance beyond either alone (primary)
    pred_6a_delta_r2: float = 0.05  # ΔR² ≥ 0.05 over best univariate model (PRIMARY criterion)
    pred_6a_auc: float = 0.80  # AUC > 0.80 (aspirational secondary metric only)

    # Pred 6.B: HEP amplitude discriminates MCS from VS/UWS
    pred_6b_cohens_d: float = 0.50  # d > 0.5 (MCS vs. VS/UWS)
    pred_6b_dmn_pci_r: float = 0.50  # DMN–PCI r > 0.5

    # Pred 6.C: interoceptive perturbation increases PCI in MCS
    pred_6c_mcs_pci_increase_pct: float = 10.0  # PCI increase ≥ 10% in MCS post-perturbation
    pred_6c_perturbation_margin_pct: float = 5.0  # ΔPCI_perturbation > ΔPCI_control + 5%
    pred_6c_hep_pci_r: float = 0.40  # PCI change correlates with HEP change: r > 0.4

    # Pred 6.D: HEP amplitude correlates with GCS-S at 3-month follow-up
    pred_6d_spearman_r: float = 0.40  # within-participant Spearman r(HEP, GCS-S) > 0.4
    pred_6d_ppv_npv_pct: float = 70.0  # joint baseline model PPV ≥ 70% and NPV ≥ 70%

    # ── EP-5 (Protocol 3) — fMRI Anticipation vs. Experience ─────────────────
    # Pred 3.A: vmPFC BOLD parametrically modulated by EV, NOT correlated with outcome SCR
    pred_3a_vmPFC_scr_r_falsify: float = 0.35  # vmPFC–SCR outcome-period r > 0.35 = falsified
    pred_3a_vmPFC_scr_r_confirm: float = 0.20  # vmPFC–SCR r < 0.20 required for confirmation
    pred_3a_pIC_scr_r_confirm: float = 0.40  # posterior insula–SCR r > 0.40 (active εⁱ control)

    # Pred 3.B: vmPFC→aINS connectivity increases; vmPFC→pIC remains flat
    pred_3b_aINS_ppi_delta_r: float = 0.30  # PPI vmPFC→aINS: Δr ≥ 0.30
    pred_3b_pIC_bf01: float = 6.0  # vmPFC→pIC flat: BF₀₁ ≥ 6, ROPE d=[−0.15,+0.15]

    # Pred 3.C: vmPFC anticipation-period activation is context-specific (EV, not sensory contrast)
    pred_3c_cohens_d: float = 0.40  # high-somatic-cost EV > low-somatic-cost EV: d > 0.4

    # Pred 3.D: removing foreperiod (0 ms ISI) abolishes vmPFC→aINS coupling
    pred_3d_bf01: float = 6.0  # BF₀₁ ≥ 6 for absence of coupling in 0 ms condition

    # ── EP-6 (Protocol 5) — iEEG All-or-None Dynamics ────────────────────────
    # Pred 5.A: frontoparietal cortex shows bimodal (not graded) high-gamma power
    pred_5a_bic_delta: float = 10.0  # 2-component Gaussian BIC lower than 1-component by > 10

    # Pred 5.B: bimodality specific to frontoparietal network; transitions sharp
    pred_5b_intermediate_bout_ms: float = 100.0  # individual intermediate-state bouts < 100 ms mean duration
    pred_5b_prevalence_pct: float = 15.0  # prevalence < 15% of trial duration

    # Pred 5.C: AC1 increases monotonically 500 ms before detected stimuli (critical slowing)
    pred_5c_ac1_kendall_tau: float = 0.30  # AC1 Kendall τ > 0.3 in detected trials (10 000 permutations)

    # Pred 5.D: long-range gamma coherence predicts seen vs. unseen trial-by-trial
    pred_5d_coherence_r: float = 0.40  # point-biserial r > 0.4, peaking 200–400 ms
    pred_5d_occipital_r_falsify: float = 0.20  # occipital coherence–detection r < 0.20 (frontoparietal specificity)
    pred_5d_hep_coherence_r: float = 0.25  # HEP–coherence r > 0.25 in seen trials (APGI-specific tier)

    # ── EP-7 (Protocol 7) — Pharmacological Modulation ───────────────────────
    # Pred 7.A: threshold-modulating interventions (cathodal tDCS, ketamine)
    pred_7a_threshold_shift: float = 0.05  # threshold shift ≥ 0.05 units
    pred_7a_cohens_d: float = 0.40  # d ≥ |0.4| in predicted direction

    # Pred 7.B: precision-modulating interventions (atomoxetine, propranolol)
    pred_7b_beta_sm_cohens_d: float = 0.40  # propranolol: significant β_SM shift, d > 0.4

    # Pred 7.C: graded pharmacological dose-response
    pred_7c_jonckheere_p: float = 0.01  # Jonckheere–Terpstra monotonic trend p < 0.01

    # Pred 7.D: active vs. placebo
    pred_7d_bf10: float = 10.0  # BF₁₀ > 10; active vs. placebo p < 0.01

    # ── EP-8 (Protocol 8) — Psychophysical Individual Differences ─────────────
    # Pred 8.A: interoceptive precision (Πⁱ) correlates with HEP, heartbeat d′, HRV
    pred_8a_pi_hep_r: float = 0.40  # r(Πⁱ, HEP) > 0.40, p < 0.01
    pred_8a_pi_heartbeat_r: float = 0.35  # r(Πⁱ, heartbeat d′) > 0.35, p < 0.01
    pred_8a_pi_hrv_r: float = 0.30  # r(Πⁱ, HRV/RMSSD) > 0.30, p < 0.05
    pred_8a_pi_hep_r_falsify: float = 0.30  # Πⁱ fails to correlate with HEP: r < 0.30 or p > 0.05

    # Pred 8.B: higher β_som predicts lower θ₀ (somatic facilitation)
    pred_8b_beta_theta_r: float = -0.25  # r(β_som, θ₀) < −0.25, p < 0.05

    # Pred 8.C: APGI parameters test-retest reliability at 1-week interval
    pred_8c_theta0_icc: float = 0.75  # θ₀ ICC > 0.75
    pred_8c_pi_icc: float = 0.65  # Πⁱ ICC > 0.65
    pred_8c_alpha_icc: float = 0.70  # α ICC > 0.70
    pred_8c_theta0_icc_falsify: float = 0.60  # θ₀ ICC < 0.60 = falsify

    # Pred 8.D: factor analysis yields ≥ 2 interpretable components; β_som and β_ign independent
    pred_8d_beta_inter_r: float = 0.40  # |r|(β_som, β_ign) < 0.40

    # Pred 8.E: clinical correlates (anxiety × Πⁱ; depression × θ₀)
    pred_8e_tai_pi_r: float = -0.25  # TAI × Πⁱ: r < −0.25, p < 0.05
    pred_8e_bdi_theta0_r: float = 0.25  # BDI × θ₀: r > 0.25, p < 0.05

    # ── EP-9 (CP-1) — ML Classification ──────────────────────────────────────
    # Pred 9.A: APGI synthetic signals yield high classification accuracy for ignition vs. no-ignition
    pred_9a_accuracy_min: float = 85.0  # 85–92% (full confirmation range)
    pred_9a_accuracy_max: float = 92.0
    pred_9a_auc_roc_min: float = 0.90  # AUC-ROC 0.90–0.95
    pred_9a_f1_min: float = 0.85  # F1-score 0.85–0.90
    pred_9a_partial_accuracy_min: float = 75.0  # 75–84% = partial support only
    pred_9a_accuracy_falsify: float = 75.0  # < 75% = falsified

    # Pred 9.B: APGI vs. GWTOnly signals distinguishable
    pred_9b_confusion_max_pct: float = 40.0  # cross-model confusion < 40%

    # Pred 9.C: classifier trained on APGI generalizes to real human data
    pred_9c_real_data_accuracy_pct: float = 55.0  # accuracy > 55% on real human data (above 50% chance)

    # ── EP-10 (CP-4) — Bayesian Model Comparison ─────────────────────────────
    # Pred 10.A: APGI achieves lowest WAIC/LOO across pre-registered datasets
    pred_10a_loo_vs_sdt_min: float = 15.0  # expected ΔLOO APGI vs. SDT: +15 to +40
    pred_10a_loo_vs_sdt_max: float = 40.0
    pred_10a_loo_vs_gwt_min: float = 5.0  # expected ΔLOO APGI vs. GWT: +5 to +20
    pred_10a_loo_vs_gwt_max: float = 20.0
    pred_10a_loo_vs_continuous_min: float = 25.0  # expected ΔLOO vs. Continuous: +25 to +70
    pred_10a_loo_vs_continuous_max: float = 70.0
    pred_10a_loo_falsify: float = 10.0  # APGI worse than SDT or GWT by > 10 = falsified

    # Pred 10.B: Πⁱ explains variance in conscious report beyond attention
    pred_10b_pi_partial_r: float = 0.25  # partial r(conscious report, Πⁱ | attention) > 0.25

    # Pred 10.C: P3b predicted by (Sₜ − θₜ) not stimulus strength alone
    pred_10c_r2_improvement_pct: float = 15.0  # R² improvement from including (Sₜ − θₜ) > 15%

    # Pred 10.E: BF for APGI vs. GWT
    pred_10e_bf10: float = 3.0  # BF₁₀ > 3 across all pre-registered datasets

    # ── EP-11 (CP-3) — Active Inference Agent Computational ──────────────────
    # Pred 11.A: APGI agent outperforms alternatives in volatile, high-cost environments
    pred_11a_performance_pct: float = 70.0  # final performance > 70% optimal
    pred_11a_convergence_trials: float = 100.0  # convergence < 100 trials

    # Pred 11.B: ignition events causally precede adaptive behavioral shifts
    pred_11b_shift_pct: float = 60.0  # ≥ 60% of significant behavioral shifts preceded by ignition (200 ms window)

    # Pred 11.C: dynamic threshold θₜ is functional
    pred_11c_performance_change_pct: float = 5.0  # disabling θₜ causes > 5% performance change

    # Pred 11.D: somatic bias (β_SM) does not default to uninformative values
    pred_11d_beta_sm_neutral_lower: float = 0.95  # β_SM must fall outside [0.95, 1.05]
    pred_11d_beta_sm_neutral_upper: float = 1.05
    pred_11d_agent_pct: float = 80.0  # ≥ 80% of agents after learning

    # Pred 11.E: θₜ is metabolically sensitive
    pred_11e_cmetabolic_theta_r: float = 0.20  # correlation C_metabolic vs. θₜ > 0.2, p < 0.05

    # ── EP-12 (CP-4-PT) — Phase Transition Analysis ───────────────────────────
    # Pred 12.A: mean |dS/dt| shows discontinuity at ignition threshold crossing
    pred_12a_discontinuity: float = 0.50  # discontinuity > 0.5
    pred_12a_cohens_d: float = 0.80  # Cohen's d > 0.8 vs. random timepoints

    # Pred 12.B: diverging susceptibility (variance ratio) near threshold
    pred_12b_variance_ratio: float = 2.0  # variance ratio (near / far) > 2.0
    pred_12b_ratio_falsify: float = 1.2  # < 1.2 = no critical fluctuations (falsify)

    # Pred 12.C: critical slowing — autocorrelation ratio near threshold
    pred_12c_autocorr_ratio: float = 1.5  # autocorrelation ratio (near / far) > 1.5
    pred_12c_ratio_falsify: float = 1.2  # < 1.2 = continuous not discrete behavior (falsify)

    # Pred 12.D: integrated information (Φ) spikes at ignition threshold crossing
    pred_12d_phi_ratio_partial: float = 1.30  # 1.3–2.0× = partial support (pre-specified indeterminate zone)
    pred_12d_phi_ratio_full: float = 2.0  # > 2.0× = full confirmation
    pred_12d_cohens_d: float = 0.80  # Cohen's d > 0.8 for Φ discontinuity
    pred_12d_phi_ratio_falsify: float = 1.30  # < 1.3× = falsified

    # Pred 12.E: Hurst exponent elevated near threshold
    pred_12e_hurst_threshold: float = 0.60  # H near threshold > 0.6 (long-range dependence)

    # ── EP-13 (CP-5) — Evolutionary Emergence ────────────────────────────────
    # Pred 13.A: APGI components are positively selected in simulated environments
    pred_13a_threshold_selection: float = 0.02  # selection coefficient: threshold mechanism > 0.02
    pred_13a_interoceptive_selection: float = 0.02  # interoceptive weighting > 0.02
    pred_13a_somatic_selection: float = 0.015  # somatic markers > 0.015

    # Pred 13.B: full APGI-like architecture reaches fixation by generation 300
    pred_13b_fixation_gen: float = 300.0  # by generation 300
    pred_13b_fixation_pct: float = 80.0  # > 80% of population carries all three components

    # Pred 13.D: components emerge in predicted order (threshold → precision → interoceptive → somatic)
    pred_13d_match_pct: float = 80.0  # ≥ 80% of independent simulation runs match predicted sequence

    # Pred 13.E: threshold mechanism reaches > 60% population frequency by generation 500
    pred_13e_frequency_gen: float = 500.0  # by generation 500
    pred_13e_frequency_pct: float = 60.0  # frequency > 60%

    # Pred 13.G: somatic markers exceed 50% frequency at evolutionary steady-state
    pred_13g_somatic_frequency_pct: float = 50.0  # frequency > 50% at steady-state

    # ── EP-14 (CP-6) — RNN Architectures with APGI Inductive Biases ──────────
    # Pred 14.A: APGI-constrained RNN achieves highest AUC vs. LSTM, Attention, MLP
    pred_14a_apgi_auc_min: float = 0.85  # APGI RNN AUC 0.85–0.92
    pred_14a_apgi_auc_max: float = 0.92
    pred_14a_lstm_auc_min: float = 0.70  # LSTM baseline 0.70–0.78
    pred_14a_lstm_auc_max: float = 0.78
    pred_14a_attention_auc_min: float = 0.75  # Attention baseline 0.75–0.82
    pred_14a_attention_auc_max: float = 0.82
    pred_14a_auc_margin_pct: float = 2.0  # superiority must exceed 2% AUC

    # Pred 14.B: APGI RNN converges faster on interoception-relevant tasks
    pred_14b_convergence_reduction_min_pct: float = 30.0  # 30–50% fewer epochs than comparison networks
    pred_14b_convergence_reduction_max_pct: float = 50.0

    # Pred 14.D: ignition probability predicts behavioral performance; threshold non-degenerate
    pred_14d_ignition_accuracy_r: float = 0.50  # r > 0.5 between ignition probability and response accuracy
    pred_14d_training_runs_pct: float = 90.0  # threshold converges to non-extreme values in ≥ 90% of runs


class ThresholdRegistry:
    """Centralized registry for APGI falsification thresholds.

    All defaults are sourced from the APGI Framework Prediction Registry.
    Field names include the prediction code (e.g. pred_0a_, pred_2b_) so
    every value is traceable to its source document entry.
    """

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.thresholds = self._load_thresholds()
        self._validate_thresholds()

    def _load_thresholds(self) -> FalsificationThresholds:
        thresholds = FalsificationThresholds()
        if self.config_manager is None:
            return thresholds

        try:
            falsif_config = self.config_manager.get_config().falsification
            for field_name in thresholds.__dataclass_fields__:
                if hasattr(falsif_config, field_name):
                    setattr(thresholds, field_name, getattr(falsif_config, field_name))
            logger.info("Loaded falsification thresholds from configuration")
        except Exception as e:
            logger.error(f"Failed to load thresholds from config: {e}")
            logger.info("Using Prediction Registry default values")

        return thresholds

    def _validate_thresholds(self) -> None:
        errors = []
        t = self.thresholds

        # Correlation bounds: must be in (0, 1)
        corr_fields = [
            "pred_0a_r_confirm",
            "pred_0a_r_replicate",
            "pred_0a_r_falsify",
            "pred_0c_hep_ins_r",
            "pred_1a_hep_p3b_partial_r",
            "pred_1a_hep_p3b_r_falsify",
            "pred_2c_kendall_tau",
            "pred_3a_vmPFC_scr_r_falsify",
            "pred_3a_pIC_scr_r_confirm",
            "pred_3b_aINS_ppi_delta_r",
            "pred_5c_ac1_kendall_tau",
            "pred_5d_coherence_r",
            "pred_5d_hep_coherence_r",
            "pred_6b_dmn_pci_r",
            "pred_6c_hep_pci_r",
            "pred_6d_spearman_r",
            "pred_8a_pi_hep_r",
            "pred_8a_pi_heartbeat_r",
            "pred_8a_pi_hrv_r",
            "pred_10b_pi_partial_r",
            "pred_11e_cmetabolic_theta_r",
            "pred_14d_ignition_accuracy_r",
        ]
        for name in corr_fields:
            val = getattr(t, name)
            if not (0.0 < abs(val) <= 1.0):
                errors.append(f"{name} must be in (0, 1], got {val}")

        # Positive-only fields
        positive_fields = [
            "pred_0b_hep_increase_pct",
            "pred_0b_cohens_d",
            "pred_0b_bf10",
            "pred_1a_p3b_eta2",
            "pred_1a_trait_cohens_d",
            "pred_1b_diastole_advantage_pct",
            "pred_1c_p3b_eta2",
            "pred_4a_pic_pci_reduction_pct",
            "pred_4a_dlpfc_pci_reduction_min_pct",
            "pred_4a_dlpfc_pci_reduction_max_pct",
            "pred_4b_dlpfc_bf01",
            "pred_4c_eta2",
            "pred_2a_convergence_trials_min",
            "pred_2a_convergence_trials_max",
            "pred_2a_reward_ratio",
            "pred_2b_interoceptive_pct_min",
            "pred_2b_interoceptive_pct_max",
            "pred_2b_interoceptive_pct_falsify",
            "pred_2c_lead_pct",
            "pred_2e_bic_delta",
            "pred_6a_delta_r2",
            "pred_6a_auc",
            "pred_6b_cohens_d",
            "pred_6c_mcs_pci_increase_pct",
            "pred_6c_perturbation_margin_pct",
            "pred_6d_ppv_npv_pct",
            "pred_3c_cohens_d",
            "pred_3d_bf01",
            "pred_5a_bic_delta",
            "pred_5b_intermediate_bout_ms",
            "pred_5b_prevalence_pct",
            "pred_7a_threshold_shift",
            "pred_7a_cohens_d",
            "pred_7b_beta_sm_cohens_d",
            "pred_7c_jonckheere_p",
            "pred_7d_bf10",
            "pred_8c_theta0_icc",
            "pred_8c_pi_icc",
            "pred_8c_alpha_icc",
            "pred_8c_theta0_icc_falsify",
            "pred_8d_beta_inter_r",
            "pred_9a_accuracy_min",
            "pred_9a_accuracy_max",
            "pred_9a_auc_roc_min",
            "pred_9a_f1_min",
            "pred_9a_partial_accuracy_min",
            "pred_9a_accuracy_falsify",
            "pred_9b_confusion_max_pct",
            "pred_9c_real_data_accuracy_pct",
            "pred_10a_loo_vs_sdt_min",
            "pred_10a_loo_vs_sdt_max",
            "pred_10a_loo_vs_gwt_min",
            "pred_10a_loo_vs_gwt_max",
            "pred_10a_loo_vs_continuous_min",
            "pred_10a_loo_vs_continuous_max",
            "pred_10a_loo_falsify",
            "pred_10c_r2_improvement_pct",
            "pred_10e_bf10",
            "pred_11a_performance_pct",
            "pred_11a_convergence_trials",
            "pred_11b_shift_pct",
            "pred_11c_performance_change_pct",
            "pred_11d_agent_pct",
            "pred_12a_discontinuity",
            "pred_12a_cohens_d",
            "pred_12b_variance_ratio",
            "pred_12b_ratio_falsify",
            "pred_12c_autocorr_ratio",
            "pred_12c_ratio_falsify",
            "pred_12d_phi_ratio_partial",
            "pred_12d_phi_ratio_full",
            "pred_12d_cohens_d",
            "pred_12e_hurst_threshold",
            "pred_13a_threshold_selection",
            "pred_13a_interoceptive_selection",
            "pred_13a_somatic_selection",
            "pred_13b_fixation_gen",
            "pred_13b_fixation_pct",
            "pred_13d_match_pct",
            "pred_13e_frequency_gen",
            "pred_13e_frequency_pct",
            "pred_13g_somatic_frequency_pct",
            "pred_14a_apgi_auc_min",
            "pred_14a_apgi_auc_max",
            "pred_14a_lstm_auc_min",
            "pred_14a_lstm_auc_max",
            "pred_14a_attention_auc_min",
            "pred_14a_attention_auc_max",
            "pred_14a_auc_margin_pct",
            "pred_14b_convergence_reduction_min_pct",
            "pred_14b_convergence_reduction_max_pct",
            "pred_14d_training_runs_pct",
        ]
        for name in positive_fields:
            val = getattr(t, name)
            if val <= 0:
                errors.append(f"{name} must be > 0, got {val}")

        # β_SM neutral bounds
        if t.pred_11d_beta_sm_neutral_lower <= 0:
            errors.append("pred_11d_beta_sm_neutral_lower must be > 0")
        if t.pred_11d_beta_sm_neutral_upper <= t.pred_11d_beta_sm_neutral_lower:
            errors.append("pred_11d_beta_sm_neutral_upper must be > pred_11d_beta_sm_neutral_lower")

        # Ordering checks
        if t.pred_2a_convergence_trials_min >= t.pred_2a_convergence_trials_max:
            errors.append("pred_2a_convergence_trials_min must be < pred_2a_convergence_trials_max")
        if t.pred_4a_dlpfc_pci_reduction_min_pct >= t.pred_4a_dlpfc_pci_reduction_max_pct:
            errors.append("pred_4a_dlpfc_pci_reduction_min_pct must be < max")
        if t.pred_12d_phi_ratio_partial >= t.pred_12d_phi_ratio_full:
            errors.append("pred_12d_phi_ratio_partial must be < pred_12d_phi_ratio_full")

        if errors:
            raise ValueError(f"Threshold validation errors: {'; '.join(errors)}")

        logger.info("All %d threshold values validated successfully", len(asdict(t)))

    def get_falsification_thresholds(self) -> FalsificationThresholds:
        """Return the current falsification threshold configuration."""
        return self.thresholds

    def get_threshold(self, name: str) -> float:
        """Return a specific threshold by its field name."""
        if not hasattr(self.thresholds, name):
            raise ValueError(f"Unknown threshold: {name}")
        return getattr(self.thresholds, name)

    def update_threshold(self, name: str, value: float) -> bool:
        """Update a specific threshold value."""
        if not hasattr(self.thresholds, name):
            raise ValueError(f"Unknown threshold: {name}")
        setattr(self.thresholds, name, value)
        logger.info(f"Updated threshold {name} to {value}")
        return True

    def get_all_thresholds(self) -> Dict[str, float]:
        """Return all thresholds as a flat dictionary."""
        return asdict(self.thresholds)

    def reset_to_defaults(self) -> None:
        """Reset all thresholds to Prediction Registry default values."""
        self.thresholds = FalsificationThresholds()
        logger.info("Reset all thresholds to Prediction Registry defaults")
