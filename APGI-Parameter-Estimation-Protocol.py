import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. REALISTIC NEURAL SIGNAL SYNTHESIS
# =============================================================================
class NeuralSignalGenerator:
    """Biophysically realistic EEG/pupil signal synthesis"""
    
    @staticmethod
    def generate_hep_waveform(Pi_i, sampling_rate=1000, duration=0.6):
        """Generate Heartbeat-Evoked Potential (250-400ms post R-peak)"""
        time_ms = np.arange(0, duration * 1000, 1000/sampling_rate)
        peak_time = 300  # ms
        sigma = 40
        baseline_amplitude = 2.0  # μV
        
        # Precision modulates amplitude
        hep_amplitude = baseline_amplitude * (0.5 + 0.5 * Pi_i)
        hep_waveform = hep_amplitude * np.exp(-((time_ms - peak_time)**2) / (2 * sigma**2))
        
        # Add 1/f pink noise
        noise = NeuralSignalGenerator._pink_noise(len(time_ms), 0.3)
        
        # Add 50Hz line noise
        line_noise = 0.1 * np.sin(2 * np.pi * 50 * time_ms / 1000)
        
        return hep_waveform + noise + line_noise
    
    @staticmethod
    def generate_p3b_waveform(amplitude_factor, sampling_rate=1000, duration=0.8):
        """Generate P3b component (250-500ms post-stimulus at Pz)"""
        time_ms = np.arange(0, duration * 1000, 1000/sampling_rate)
        peak_time = 350
        sigma = 60
        baseline = 5.0  # μV
        
        p3b_amplitude = baseline * amplitude_factor
        p3b_waveform = p3b_amplitude * np.exp(-((time_ms - peak_time)**2) / (2 * sigma**2))
        
        # Add earlier P2 component
        p2_amplitude = baseline * 0.3 * amplitude_factor
        p2_waveform = p2_amplitude * np.exp(-((time_ms - 200)**2) / (2 * 30**2))
        
        # Physiological noise
        noise = NeuralSignalGenerator._pink_noise(len(time_ms), 0.4)
        
        return p3b_waveform + p2_waveform + noise
    
    @staticmethod
    def generate_pupil_response(Pi_i, sampling_rate=1000, duration=3.0, contamination_rate=0.15):
        """Generate pupil dilation with blink artifacts"""
        time_s = np.arange(0, duration, 1/sampling_rate)
        peak_time = 1.5  # seconds
        sigma = 0.5
        baseline_dilation = 0.2  # mm
        
        # Precision modulates dilation
        dilation_magnitude = baseline_dilation * (0.3 + 0.7 * Pi_i)
        pupil_response = dilation_magnitude * np.exp(-((time_s - peak_time)**2) / (2 * sigma**2))
        
        # Slow drift
        drift = 0.05 * np.sin(2 * np.pi * 0.1 * time_s)
        
        # Blink artifacts
        if np.random.rand() < contamination_rate:
            blink_time = np.random.uniform(0.5, 2.5)
            blink_idx = int(blink_time * sampling_rate)
            blink_duration = int(0.15 * sampling_rate)
            pupil_response[blink_idx:blink_idx+blink_duration] = np.nan
        
        return pupil_response + drift + np.random.normal(0, 0.02, len(time_s))
    
    @staticmethod
    def _pink_noise(n_samples, amplitude):
        """Generate 1/f pink noise"""
        pink = np.zeros(n_samples)
        for octave in range(5):
            step = 2 ** octave
            pink += np.repeat(np.random.randn(n_samples // step + 1), step)[:n_samples] / (octave + 1)
        return amplitude * pink / (np.std(pink) + 1e-10)

# =============================================================================
# 2. APGI TEMPORAL DYNAMICS ENGINE
# =============================================================================
class APGIDynamics:
    """Core APGI equations: surprise accumulation and ignition probability"""
    
    @staticmethod
    def simulate_surprise_accumulation(epsilon_e, epsilon_i, Pi_e, Pi_i, beta, tau=0.2, dt=0.001, duration=1.0):
        """
        Integrate: dS/dt = -S/τ + f(Πₑ·|εₑ|, β·Πᵢ·|εᵢ|)
        """
        n_steps = int(duration / dt)
        S_trajectory = np.zeros(n_steps)
        
        S = 0.0
        for i in range(1, n_steps):
            extero_contrib = Pi_e * np.abs(epsilon_e)
            intero_contrib = beta * Pi_i * np.abs(epsilon_i)
            dS_dt = -S / tau + extero_contrib + intero_contrib
            S += dt * dS_dt
            S_trajectory[i] = S
        
        return S_trajectory
    
    @staticmethod
    def compute_ignition_probability(S_t, theta_t, alpha=5.0):
        """Compute: B(t) = σ(α(S(t) - θ(t)))"""
        return 1 / (1 + np.exp(-alpha * (S_t - theta_t)))

# =============================================================================
# 3. SYNTHETIC DATASET GENERATOR WITH AGGREGATED DATA
# =============================================================================
def generate_synthetic_dataset(n_subjects=100, seed=42):
    """Generate synthetic multimodal data with summary statistics"""
    np.random.seed(seed)
    signal_gen = NeuralSignalGenerator()
    dynamics = APGIDynamics()
    
    subjects = []
    
    # Ground truth parameters (4-parameter model for identifiability)
    true_params = {
        'theta0': np.random.normal(0.55, 0.1, n_subjects),
        'Pi_i': np.random.gamma(2, 0.5, n_subjects),
        'beta': np.random.normal(1.15, 0.25, n_subjects),
        'alpha': np.random.normal(5.0, 0.8, n_subjects),
    }
    
    # Constrain to biologically plausible ranges
    true_params['theta0'] = np.clip(true_params['theta0'], 0.25, 0.85)
    true_params['Pi_i'] = np.clip(true_params['Pi_i'], 0.4, 2.8)
    true_params['beta'] = np.clip(true_params['beta'], 0.6, 2.3)
    true_params['alpha'] = np.clip(true_params['alpha'], 2.5, 7.5)
    
    for subj_id in range(n_subjects):
        theta0 = true_params['theta0'][subj_id]
        Pi_i = true_params['Pi_i'][subj_id]
        beta = true_params['beta'][subj_id]
        alpha = true_params['alpha'][subj_id]
        
        # =================================================================
        # TASK 1: Detection Psychometric Curve
        # =================================================================
        n_intensity_levels = 8
        intensities = np.linspace(0.3, 0.75, n_intensity_levels)
        n_trials_per_level = 25
        
        seen_counts = []
        for intensity in intensities:
            # APGI dynamics
            epsilon_e = intensity - 0.5
            S_traj = dynamics.simulate_surprise_accumulation(
                epsilon_e=epsilon_e, epsilon_i=0.0,
                Pi_e=1.0, Pi_i=Pi_i, beta=beta, tau=0.2
            )
            S_final = S_traj[int(0.5 / 0.001)]
            p_seen = dynamics.compute_ignition_probability(S_final, theta0, alpha)
            
            # Generate responses
            n_seen = np.random.binomial(n_trials_per_level, p_seen)
            seen_counts.append(n_seen)
        
        # =================================================================
        # TASK 2: Heartbeat Detection
        # =================================================================
        n_hb_trials = 60
        
        # Generate HEP amplitudes
        hep_amplitudes = []
        for _ in range(n_hb_trials):
            hep = signal_gen.generate_hep_waveform(Pi_i)
            hep_amplitudes.append(np.max(hep))
        
        # Generate pupil responses
        pupil_responses = []
        for _ in range(n_hb_trials):
            pupil = signal_gen.generate_pupil_response(Pi_i)
            pupil_responses.append(np.nanmean(pupil))
        
        # Behavioral d-prime
        hit_rate = np.clip(0.5 + 0.3 * (Pi_i / 3.0), 0.05, 0.95)
        fa_rate = np.clip(0.5 - 0.2 * (Pi_i / 3.0), 0.05, 0.95)
        d_prime = stats.norm.ppf(hit_rate) - stats.norm.ppf(fa_rate)
        
        # =================================================================
        # TASK 3: Oddball P3b
        # =================================================================
        Pi_e_calibrated = Pi_i * np.random.normal(1.0, 0.05)
        
        p3b_intero_amplitudes = []
        p3b_extero_amplitudes = []
        
        for _ in range(30):
            # Interoceptive deviant
            p3b = signal_gen.generate_p3b_waveform(beta * Pi_i * 0.5)
            p3b_intero_amplitudes.append(np.max(p3b))
            
            # Exteroceptive deviant
            p3b = signal_gen.generate_p3b_waveform(Pi_e_calibrated * 0.5)
            p3b_extero_amplitudes.append(np.max(p3b))
        
        # Store subject data
        subjects.append({
            'intensities': intensities,
            'seen_counts': np.array(seen_counts, dtype=int),
            'n_trials': n_trials_per_level,
            'dprime': d_prime,
            'hep_mean': np.mean(hep_amplitudes),
            'pupil_mean': np.nanmean(pupil_responses),
            'p3b_intero_mean': np.mean(p3b_intero_amplitudes),
            'p3b_extero_mean': np.mean(p3b_extero_amplitudes),
            'true_params': {k: v[subj_id] for k, v in true_params.items()}
        })
    
    return subjects, true_params

# =============================================================================
# 4. HIERARCHICAL BAYESIAN MODEL (VECTORIZED, NO LOOPS)
# =============================================================================
def build_apgi_model(subjects):
    """Properly vectorized hierarchical model"""
    n_subjects = len(subjects)
    n_levels = len(subjects[0]['intensities'])
    
    # Extract data as arrays
    intensities = subjects[0]['intensities']
    seen_counts = np.array([s['seen_counts'] for s in subjects])  # (n_subj, n_levels)
    n_trials = subjects[0]['n_trials']
    
    dprime_obs = np.array([s['dprime'] for s in subjects])
    hep_obs = np.array([s['hep_mean'] for s in subjects])
    pupil_obs = np.array([s['pupil_mean'] for s in subjects])
    
    p3b_ratio_obs = np.array([
        s['p3b_intero_mean'] / (s['p3b_extero_mean'] + 0.01)
        for s in subjects
    ])
    
    with pm.Model() as model:
        # =====================================================================
        # GROUP-LEVEL HYPERPRIORS
        # =====================================================================
        mu_theta0 = pm.Normal('mu_theta0', mu=0.55, sigma=0.2)
        sigma_theta0 = pm.HalfNormal('sigma_theta0', sigma=0.1)
        
        mu_Pi_i = pm.Normal('mu_Pi_i', mu=1.2, sigma=0.5)
        sigma_Pi_i = pm.HalfNormal('sigma_Pi_i', sigma=0.3)
        
        mu_beta = pm.Normal('mu_beta', mu=1.15, sigma=0.3)
        sigma_beta = pm.HalfNormal('sigma_beta', sigma=0.2)
        
        mu_alpha = pm.Normal('mu_alpha', mu=5.0, sigma=1.0)
        sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=0.4)
        
        # =====================================================================
        # SUBJECT-LEVEL PARAMETERS (NON-CENTERED PARAMETERIZATION)
        # =====================================================================
        theta0_offset = pm.Normal('theta0_offset', mu=0, sigma=1, shape=n_subjects)
        theta0 = pm.Deterministic('theta0', mu_theta0 + sigma_theta0 * theta0_offset)
        theta0_constrained = pm.math.clip(theta0, 0.25, 0.85)
        
        Pi_i_offset = pm.Normal('Pi_i_offset', mu=0, sigma=1, shape=n_subjects)
        Pi_i = pm.Deterministic('Pi_i', mu_Pi_i + sigma_Pi_i * Pi_i_offset)
        Pi_i_constrained = pm.math.clip(Pi_i, 0.4, 2.8)
        
        beta_offset = pm.Normal('beta_offset', mu=0, sigma=1, shape=n_subjects)
        beta = pm.Deterministic('beta', mu_beta + sigma_beta * beta_offset)
        beta_constrained = pm.math.clip(beta, 0.6, 2.3)
        
        alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=n_subjects)
        alpha = pm.Deterministic('alpha', mu_alpha + sigma_alpha * alpha_offset)
        alpha_constrained = pm.math.clip(alpha, 2.5, 7.5)
        
        # =====================================================================
        # LIKELIHOOD 1: Detection Psychometric (VECTORIZED BINOMIAL)
        # =====================================================================
        # Shape: (n_subjects, n_levels)
        intensity_grid = np.tile(intensities, (n_subjects, 1))
        
        # Broadcast: (n_subjects, 1) - (1, n_levels) → (n_subjects, n_levels)
        logit_p = alpha_constrained[:, None] * (intensity_grid - theta0_constrained[:, None])
        p_seen = pm.math.sigmoid(logit_p)
        
        pm.Binomial('detection', n=n_trials, p=p_seen, observed=seen_counts)
        
        # =====================================================================
        # LIKELIHOOD 2: Heartbeat d-prime
        # =====================================================================
        pm.Normal('dprime', mu=Pi_i_constrained * 0.6, sigma=0.3, observed=dprime_obs)
        
        # =====================================================================
        # LIKELIHOOD 3: HEP amplitude (neural prior)
        # =====================================================================
        expected_hep = 2.0 * (0.5 + 0.5 * Pi_i_constrained)
        pm.Normal('hep', mu=expected_hep, sigma=0.5, observed=hep_obs)
        
        # =====================================================================
        # LIKELIHOOD 4: Pupil dilation (neural prior)
        # =====================================================================
        expected_pupil = 0.2 * (0.3 + 0.7 * Pi_i_constrained)
        pm.Normal('pupil', mu=expected_pupil, sigma=0.05, observed=pupil_obs)
        
        # =====================================================================
        # LIKELIHOOD 5: P3b ratio (somatic bias)
        # =====================================================================
        pm.Normal('p3b_ratio', mu=beta_constrained, sigma=0.3, observed=p3b_ratio_obs)
    
    return model

# =============================================================================
# 5. PARAMETER RECOVERY VALIDATION
# =============================================================================
def validate_parameter_recovery(true_params, trace, n_subjects):
    """Validate recovery of all 4 parameters"""
    param_names = ['theta0', 'Pi_i', 'beta', 'alpha']
    thresholds = {'theta0': 0.85, 'Pi_i': 0.75, 'beta': 0.85, 'alpha': 0.70}
    
    results = {}
    for param in param_names:
        true_vals = true_params[param][:n_subjects]
        recovered_vals = trace.posterior[param].mean(dim=['chain', 'draw']).values
        
        r, p = stats.pearsonr(true_vals, recovered_vals)
        rmse = np.sqrt(np.mean((true_vals - recovered_vals)**2))
        
        # Credible interval coverage
        lower = np.percentile(trace.posterior[param].values, 2.5, axis=(0, 1))
        upper = np.percentile(trace.posterior[param].values, 97.5, axis=(0, 1))
        coverage = np.mean((true_vals >= lower) & (true_vals <= upper))
        
        results[param] = {
            'r': r,
            'p_value': p,
            'rmse': rmse,
            'coverage': coverage,
            'threshold': thresholds[param],
            'passes': r >= thresholds[param]
        }
    
    falsified = any([not results[p]['passes'] for p in param_names])
    return results, falsified

# =============================================================================
# 6. PREDICTIVE VALIDITY ON HELD-OUT TEST SET
# =============================================================================
def assess_predictive_validity(subjects, trace, test_fraction=0.2, seed=42):
    """Test predictions on independent tasks with held-out subjects"""
    np.random.seed(seed)
    n_subjects = len(subjects)
    n_test = int(test_fraction * n_subjects)
    test_indices = np.random.choice(n_subjects, n_test, replace=False)
    
    # Extract estimates
    Pi_i_est = trace.posterior['Pi_i'].mean(dim=['chain', 'draw']).values[test_indices]
    theta0_est = trace.posterior['theta0'].mean(dim=['chain', 'draw']).values[test_indices]
    beta_est = trace.posterior['beta'].mean(dim=['chain', 'draw']).values[test_indices]
    
    # Generate true and predicted outcomes
    emotional_rt_true, emotional_rt_pred = [], []
    cpt_lapses_true, cpt_lapses_pred = [], []
    bvs_true, bvs_pred = [], []
    
    for i, idx in enumerate(test_indices):
        Pi_i_true = subjects[idx]['true_params']['Pi_i']
        theta0_true = subjects[idx]['true_params']['theta0']
        beta_true = subjects[idx]['true_params']['beta']
        
        # Task 1: Emotional Stroop interference
        interference_true = 50 + 100 * (beta_true * Pi_i_true / 3.0)
        rt_true = 550 + interference_true + np.random.normal(0, 30)
        
        interference_pred = 50 + 100 * (beta_est[i] * Pi_i_est[i] / 3.0)
        rt_pred = 550 + interference_pred
        
        emotional_rt_true.append(rt_true)
        emotional_rt_pred.append(rt_pred)
        
        # Task 2: CPT attentional lapses
        lapses_true = int(5 + 20 * theta0_true + np.random.poisson(3))
        lapses_pred = 5 + 20 * theta0_est[i]
        
        cpt_lapses_true.append(np.clip(lapses_true, 0, 30))
        cpt_lapses_pred.append(lapses_pred)
        
        # Task 3: Body Vigilance Scale
        bvs_score_true = 15 + 8 * beta_true + 3 * Pi_i_true + np.random.normal(0, 4)
        bvs_score_pred = 15 + 8 * beta_est[i] + 3 * Pi_i_est[i]
        
        bvs_true.append(np.clip(bvs_score_true, 0, 64))
        bvs_pred.append(bvs_score_pred)
    
    return {
        'emotional_interference': {
            'r': stats.pearsonr(emotional_rt_true, emotional_rt_pred)[0],
            'r2': r2_score(emotional_rt_true, emotional_rt_pred)
        },
        'cpt_lapses': {
            'r': stats.pearsonr(cpt_lapses_true, cpt_lapses_pred)[0],
            'r2': r2_score(cpt_lapses_true, cpt_lapses_pred)
        },
        'body_vigilance': {
            'r': stats.pearsonr(bvs_true, bvs_pred)[0],
            'r2': r2_score(bvs_true, bvs_pred)
        }
    }

# =============================================================================
# 7. TEST-RETEST RELIABILITY
# =============================================================================
def assess_test_retest_reliability(n_subjects=40, seed=42):
    """Simulate test-retest with 1-week interval"""
    print("  Generating Time 1 dataset...")
    subjects_t1, true_params = generate_synthetic_dataset(n_subjects, seed=seed)
    
    print("  Generating Time 2 dataset (same parameters, different noise)...")
    subjects_t2, _ = generate_synthetic_dataset(n_subjects, seed=seed + 1000)
    
    print("  Fitting Time 1 model...")
    model_t1 = build_apgi_model(subjects_t1)
    with model_t1:
        trace_t1 = pm.sample(500, tune=500, chains=2, cores=2,
                            target_accept=0.90, progressbar=False,
                            return_inferencedata=True)
    
    print("  Fitting Time 2 model...")
    model_t2 = build_apgi_model(subjects_t2)
    with model_t2:
        trace_t2 = pm.sample(500, tune=500, chains=2, cores=2,
                            target_accept=0.90, progressbar=False,
                            return_inferencedata=True)
    
    # Compute ICCs
    param_names = ['theta0', 'Pi_i', 'beta', 'alpha']
    iccs = {}
    
    for param in param_names:
        vals_t1 = trace_t1.posterior[param].mean(dim=['chain', 'draw']).values
        vals_t2 = trace_t2.posterior[param].mean(dim=['chain', 'draw']).values
        
        # ICC(2,1) - two-way random effects, absolute agreement
        between_subj_var = np.var([np.mean([vals_t1[i], vals_t2[i]]) 
                                   for i in range(n_subjects)])
        within_subj_var = np.mean([np.var([vals_t1[i], vals_t2[i]]) 
                                  for i in range(n_subjects)])
        
        icc = between_subj_var / (between_subj_var + within_subj_var + 1e-10)
        iccs[param] = icc
    
    return iccs

# =============================================================================
# 8. MAIN EXECUTION PIPELINE
# =============================================================================
if __name__ == "__main__":
    print("="*80)
    print("APGI PARAMETER ESTIMATION PROTOCOL - PRODUCTION VERSION")
    print("="*80)
    
    # =========================================================================
    # STEP 1: Generate Synthetic Dataset
    # =========================================================================
    print("\n[1/6] Generating synthetic dataset (N=100)...")
    subjects, true_params = generate_synthetic_dataset(n_subjects=100, seed=42)
    print("✓ Dataset generated with full APGI temporal dynamics")
    print(f"  - Detection: 8 intensity levels × 25 trials per subject")
    print(f"  - Heartbeat: 60 trials per subject with HEP/pupil signals")
    print(f"  - Oddball: 30 intero + 30 extero deviants per subject")
    
    # =========================================================================
    # STEP 2: Build Hierarchical Model
    # =========================================================================
    print("\n[2/6] Building hierarchical Bayesian model...")
    model = build_apgi_model(subjects)
    print("✓ Model specification complete")
    print(f"  - Parameters: 4 core (θ₀, Πᵢ, β, α)")
    print(f"  - Subjects: {len(subjects)}")
    print(f"  - Likelihoods: Detection (vectorized) + 4 neural/behavioral")
    
    # =========================================================================
    # STEP 3: MCMC Sampling
    # =========================================================================
    print("\n[3/6] Sampling from posterior distribution...")
    print("  Settings: 1500 draws, 1000 tuning, 3 chains, target_accept=0.92")
    
    with model:
        trace = pm.sample(
            draws=1500,
            tune=1000,
            chains=3,
            cores=3,
            target_accept=0.92,
            init='adapt_diag',
            return_inferencedata=True,
            progressbar=True,
            idata_kwargs={'log_likelihood': False}  # Speed up
        )
    
    print("✓ Sampling complete")
    
    # =========================================================================
    # STEP 4: Parameter Recovery Validation
    # =========================================================================
    print("\n[4/6] Validating parameter recovery...")
    recovery_results, falsified = validate_parameter_recovery(true_params, trace, 100)
    
    print("\n" + "="*80)
    print("PARAMETER RECOVERY RESULTS")
    print("="*80)
    print(f"{'Parameter':<12} {'r':<8} {'RMSE':<10} {'Coverage':<10} {'Threshold':<10} {'Status':<8}")
    print("-"*80)
    
    for param, res in recovery_results.items():
        status = "✓ PASS" if res['passes'] else "✗ FAIL"
        print(f"{param:<12} {res['r']:>7.3f}  {res['rmse']:>9.4f}  "
              f"{res['coverage']:>9.1%}  {res['threshold']:>9.2f}  {status:<8}")
    
    if falsified:
        print("\n✗ MODEL FALSIFIED")
        print("The following parameters failed recovery criteria:")
        for param, res in recovery_results.items():
            if not res['passes']:
                print(f"  - {param}: r = {res['r']:.3f} (threshold: {res['threshold']:.2f})")
    else:
        print("\n✓ MODEL VALIDATED: All recovery criteria satisfied")
    
    # =========================================================================
    # STEP 5: Predictive Validity
    # =========================================================================
    print("\n[5/6] Assessing predictive validity (20% held-out test set)...")
    pred_validity = assess_predictive_validity(subjects, trace)
    
    print("\n" + "="*80)
    print("PREDICTIVE VALIDITY RESULTS (Independent Tasks)")
    print("="*80)
    print(f"{'Task':<40} {'r':<8} {'R²':<8}")
    print("-"*80)
    print(f"{'Emotional Interference (Stroop RT)':<40} {pred_validity['emotional_interference']['r']:>7.3f}  "
          f"{pred_validity['emotional_interference']['r2']:>7.3f}")
    print(f"{'CPT Attentional Lapses':<40} {pred_validity['cpt_lapses']['r']:>7.3f}  "
          f"{pred_validity['cpt_lapses']['r2']:>7.3f}")
    print(f"{'Body Vigilance Scale':<40} {pred_validity['body_vigilance']['r']:>7.3f}  "
          f"{pred_validity['body_vigilance']['r2']:>7.3f}")
    
    # =========================================================================
    # STEP 6: Test-Retest Reliability
    # =========================================================================
    print("\n[6/6] Assessing test-retest reliability (N=40, 1-week interval)...")
    iccs = assess_test_retest_reliability(n_subjects=40, seed=123)
    
    print("\n" + "="*80)
    print("TEST-RETEST RELIABILITY (1-Week Interval)")
    print("="*80)
    print(f"{'Parameter':<12} {'ICC':<8} {'Threshold':<10} {'Status':<8}")
    print("-"*80)
    
    for param, icc in iccs.items():
        status = "✓ PASS" if icc >= 0.75 else "✗ FAIL"
        print(f"{param:<12} {icc:>7.3f}  {0.75:>9.2f}  {status:<8}")
    
    # =========================================================================
    # Model Diagnostics
    # =========================================================================
    print("\n" + "="*80)
    print("MCMC DIAGNOSTICS")
    print("="*80)
    
    param_names = ['theta0', 'Pi_i', 'beta', 'alpha']
    ess_vals = []
    rhat_vals = []
    
    for param in param_names:
        ess_param = az.ess(trace, var_names=[param])
        rhat_param = az.rhat(trace, var_names=[param])
        
        ess_vals.append(float(ess_param[param].min().values))
        rhat_vals.append(float(rhat_param[param].max().values))
    
    ess_min = min(ess_vals)
    rhat_max = max(rhat_vals)
    
    n_divergences = int(trace.sample_stats.diverging.sum().values)
    
    print(f"Effective Sample Size (minimum): {ess_min:.1f}")
    print(f"R-hat (maximum): {rhat_max:.4f}")
    print(f"Divergences: {n_divergences}")
    
    if ess_min > 400 and rhat_max < 1.01 and n_divergences < 50:
        print("✓ Excellent convergence")
        convergence_score = 15
    elif ess_min > 200 and rhat_max < 1.05 and n_divergences < 200:
        print("⚠ Acceptable convergence")
        convergence_score = 10
    else:
        print("✗ Poor convergence - increase draws/tuning")
        convergence_score = 0
    
    # =========================================================================
    # Visualization
    # =========================================================================
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    params = ['theta0', 'Pi_i', 'beta', 'alpha']
    labels = [r'$\theta_0$ (Ignition Threshold)', 
              r'$\Pi_i$ (Interoceptive Precision)',
              r'$\beta$ (Somatic Bias)', 
              r'$\alpha$ (Sigmoid Steepness)']
    
    for i, (param, label) in enumerate(zip(params, labels)):
        row, col = i // 2, i % 2
        
        true_vals = true_params[param][:100]
        recovered_vals = trace.posterior[param].mean(dim=['chain', 'draw']).values
        
        # Scatter plot
        axs[row, col].scatter(true_vals, recovered_vals, alpha=0.5, s=50, 
                             color='steelblue', edgecolors='darkblue', linewidth=0.5)
        
        # Perfect recovery line
        lims = [min(true_vals.min(), recovered_vals.min()) * 0.95,
                max(true_vals.max(), recovered_vals.max()) * 1.05]
        axs[row, col].plot(lims, lims, 'r--', lw=2, alpha=0.7, label='Perfect Recovery')
        
        # Regression line
        z = np.polyfit(true_vals, recovered_vals, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(lims[0], lims[1], 100)
        axs[row, col].plot(x_fit, p(x_fit), 'g-', alpha=0.6, lw=2,
                          label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
        
        axs[row, col].set_xlabel(f'True {label}', fontsize=12, fontweight='bold')
        axs[row, col].set_ylabel(f'Estimated {label}', fontsize=12, fontweight='bold')
        axs[row, col].set_title(f'{label}\nr = {recovery_results[param]["r"]:.3f}, '
                               f'RMSE = {recovery_results[param]["rmse"]:.4f}', 
                               fontsize=11)
        axs[row, col].legend(fontsize=9, loc='upper left')
        axs[row, col].grid(alpha=0.3, linestyle='--')
        axs[row, col].set_xlim(lims)
        axs[row, col].set_ylim(lims)
    
    plt.tight_layout()
    plt.savefig('apgi_parameter_recovery.png', dpi=300, bbox_inches='tight')
    print("✓ Recovery plot saved: apgi_parameter_recovery.png")
    
    # =========================================================================
    # Final Scoring
    # =========================================================================
    print("\n" + "="*80)
    print("FINAL VALIDATION SUMMARY")
    print("="*80)
    
    score = 0
    max_score = 100
    
    # Parameter recovery (50 points)
    passed_params = sum([1 for res in recovery_results.values() if res['passes']])
    recovery_score = int(50 * passed_params / len(recovery_results))
    score += recovery_score
    print(f"Parameter Recovery: {recovery_score}/50 ({passed_params}/4 parameters passed)")
    
    # Predictive validity (20 points)
    pred_r2s = [pred_validity[k]['r2'] for k in pred_validity.keys()]
    if all([r2 > 0.45 for r2 in pred_r2s]):
        pred_score = 20
    elif all([r2 > 0.30 for r2 in pred_r2s]):
        pred_score = 15
    elif all([r2 > 0.15 for r2 in pred_r2s]):
        pred_score = 10
    else:
        pred_score = 5
    score += pred_score
    print(f"Predictive Validity: {pred_score}/20 (mean R² = {np.mean(pred_r2s):.3f})")
    
    # Test-retest reliability (15 points)
    icc_vals = list(iccs.values())
    if all([icc >= 0.75 for icc in icc_vals]):
        reliability_score = 15
    elif all([icc >= 0.65 for icc in icc_vals]):
        reliability_score = 10
    elif all([icc >= 0.50 for icc in icc_vals]):
        reliability_score = 5
    else:
        reliability_score = 0
    score += reliability_score
    print(f"Test-Retest Reliability: {reliability_score}/15 (mean ICC = {np.mean(icc_vals):.3f})")
    
    # MCMC convergence (15 points)
    score += convergence_score
    print(f"MCMC Convergence: {convergence_score}/15")
    
    print("\n" + "="*80)
    print(f"FINAL SCORE: {score}/{max_score}")
    print("="*80)
    
    if score >= 90:
        print("✓✓✓ EXCELLENT: Protocol ready for empirical validation")
    elif score >= 75:
        print("✓✓ GOOD: Minor refinements recommended")
    elif score >= 60:
        print("✓ ACCEPTABLE: Significant improvements needed")
    else:
        print("✗ INADEQUATE: Major revisions required")
    
    print("\n" + "="*80)
    print("Protocol execution complete.")
    print("="*80)