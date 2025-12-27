"""
APGI Protocol 1: Synthetic Neural Data Generation and Machine Learning Classification
=====================================================================================

Complete implementation of falsifiable predictions for the APGI framework through
synthetic data generation and multi-model comparison using deep learning.

Author: APGI Research Team
Date: 2025
Version: 1.0 (Production)

Dependencies:
    numpy, scipy, torch, sklearn, matplotlib, seaborn, tqdm
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from scipy import signal, stats
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
from tqdm import tqdm
import json
from collections import defaultdict

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# =============================================================================
# PART 1: APGI DYNAMICAL SYSTEM & MEASUREMENT EQUATIONS
# =============================================================================

class APGIDynamicalSystem:
    """Core APGI equations for surprise accumulation and ignition"""
    
    def __init__(self, tau: float = 0.2, alpha: float = 5.0):
        """
        Args:
            tau: Surprise decay time constant (seconds)
            alpha: Sigmoid steepness for ignition probability
        """
        self.tau = tau
        self.alpha = alpha
    
    def simulate_surprise_accumulation(
        self,
        epsilon_e: float,
        epsilon_i: float,
        Pi_e: float,
        Pi_i: float,
        beta: float,
        theta_t: float,
        dt: float = 0.001,
        duration: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Simulate APGI surprise accumulation dynamics
        
        Equation: dS/dt = -S/τ + Π_e·|ε_e| + β·Π_i·|ε_i|
        
        Returns:
            S_trajectory: Surprise over time
            B_trajectory: Ignition probability over time
            ignition_occurred: Whether ignition threshold was crossed
        """
        n_steps = int(duration / dt)
        S_trajectory = np.zeros(n_steps)
        B_trajectory = np.zeros(n_steps)
        
        S = 0.0
        ignition_occurred = False
        
        for i in range(1, n_steps):
            # APGI core equation
            extero_contrib = Pi_e * np.abs(epsilon_e)
            intero_contrib = beta * Pi_i * np.abs(epsilon_i)
            
            dS_dt = -S / self.tau + extero_contrib + intero_contrib
            S += dt * dS_dt
            S = np.clip(S, 0, 10)  # Physiological bounds
            
            S_trajectory[i] = S
            
            # Ignition probability
            B_trajectory[i] = self._sigmoid(S - theta_t)
            
            # Check for ignition (threshold crossing)
            if S > theta_t and not ignition_occurred:
                ignition_occurred = True
        
        return S_trajectory, B_trajectory, ignition_occurred
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid with steepness α"""
        return 1.0 / (1.0 + np.exp(-self.alpha * x))


class APGISyntheticSignalGenerator:
    """Generate biophysically realistic neural signals from APGI dynamics"""
    
    def __init__(self, fs: int = 1000):
        """
        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
    
    def generate_P3b_waveform(
        self,
        S_t: float,
        theta_t: float,
        ignition: bool,
        duration: float = 0.8
    ) -> np.ndarray:
        """
        Generate P3b component (350-600ms post-stimulus)
        
        Measurement Equation:
            P3b_amplitude = c_0 + c_1 * max(S_t - θ_t, 0) if ignition
            P3b_amplitude = c_0 * 0.3 if no ignition
        
        Args:
            S_t: Surprise signal value at stimulus time
            theta_t: Ignition threshold
            ignition: Whether ignition occurred
            duration: Waveform duration in seconds
        
        Returns:
            ERP waveform (μV)
        """
        c_0, c_1 = 2.0, 8.0  # Baseline and scaling (μV)
        
        n_samples = int(duration * self.fs)
        t = np.linspace(0, duration, n_samples)
        
        if ignition:
            amplitude = c_0 + c_1 * max(S_t - theta_t, 0)
            peak_time = 0.45  # 450ms
            sigma = 0.08
        else:
            amplitude = c_0 * 0.3
            peak_time = 0.45
            sigma = 0.08
        
        # Gaussian peak for P3b
        waveform = amplitude * np.exp(-((t - peak_time)**2) / (2 * sigma**2))
        
        # Add earlier P2 component (200ms)
        if ignition:
            p2_amp = amplitude * 0.2
        else:
            p2_amp = amplitude * 0.4
        waveform += p2_amp * np.exp(-((t - 0.20)**2) / (2 * 0.03**2))
        
        # Add N2 component (200-250ms) - larger for ignited trials
        if ignition:
            n2_amp = -amplitude * 0.15
        else:
            n2_amp = -amplitude * 0.25
        waveform += n2_amp * np.exp(-((t - 0.22)**2) / (2 * 0.04**2))
        
        # Add physiological noise
        waveform += self._pink_noise(n_samples, 0.5)
        
        # Add alpha oscillation (8-13 Hz)
        alpha_freq = np.random.uniform(8, 13)
        waveform += 0.3 * np.sin(2 * np.pi * alpha_freq * t)
        
        return waveform
    
    def generate_HEP_waveform(
        self,
        Pi_i: float,
        epsilon_i: float,
        duration: float = 0.6
    ) -> np.ndarray:
        """
        Generate Heartbeat-Evoked Potential (250-400ms post R-peak)
        
        Measurement Equation:
            HEP_amplitude = a_0 + a_1 * Π_i * |ε_i|
        
        Args:
            Pi_i: Interoceptive precision
            epsilon_i: Interoceptive prediction error
            duration: Waveform duration in seconds
        
        Returns:
            HEP waveform (μV)
        """
        a_0, a_1 = 1.5, 2.5  # Baseline and scaling
        
        amplitude = a_0 + a_1 * Pi_i * np.abs(epsilon_i)
        
        n_samples = int(duration * self.fs)
        t = np.linspace(0, duration, n_samples)
        
        peak_time = 0.32  # 320ms
        sigma = 0.05
        
        waveform = amplitude * np.exp(-((t - peak_time)**2) / (2 * sigma**2))
        
        # Add cardiac field artifact (QRS complex)
        qrs_time = 0.05
        qrs_amp = 3.0
        waveform += qrs_amp * np.exp(-((t - qrs_time)**2) / (2 * 0.015**2))
        
        # Physiological noise
        waveform += self._pink_noise(n_samples, 0.3)
        
        return waveform
    
    def generate_gamma_burst(
        self,
        ignition: bool,
        S_t: float,
        duration: float = 1.0
    ) -> np.ndarray:
        """
        Generate late gamma synchronization (30-100 Hz)
        
        Prediction: Gamma burst during ignition (400-600ms)
        
        Args:
            ignition: Whether ignition occurred
            S_t: Surprise signal magnitude
            duration: Signal duration in seconds
        
        Returns:
            Gamma band signal
        """
        n_samples = int(duration * self.fs)
        t = np.linspace(0, duration, n_samples)
        
        if ignition:
            # Gamma burst envelope centered at 500ms
            envelope = np.exp(-((t - 0.5)**2) / (2 * 0.1**2))
            
            # Frequency scales with surprise (40-70 Hz)
            gamma_freq = 40 + min(20 * S_t / 3.0, 30)
            
            # Amplitude scales with surprise
            amplitude = 2.0 + S_t
            
            gamma = amplitude * envelope * np.sin(2 * np.pi * gamma_freq * t)
        else:
            # Background gamma (low amplitude, constant)
            gamma = 0.2 * np.sin(2 * np.pi * 40 * t)
        
        # Add noise
        gamma += self._pink_noise(n_samples, 0.3)
        
        return gamma
    
    def generate_pupil_response(
        self,
        Pi_i: float,
        ignition: bool,
        duration: float = 3.0,
        blink_prob: float = 0.15
    ) -> np.ndarray:
        """
        Generate pupil dilation response
        
        Args:
            Pi_i: Interoceptive precision (modulates dilation)
            ignition: Whether ignition occurred
            duration: Signal duration in seconds
            blink_prob: Probability of blink artifact
        
        Returns:
            Pupil diameter change (mm)
        """
        n_samples = int(duration * self.fs)
        t = np.linspace(0, duration, n_samples)
        
        if ignition:
            peak_time = 1.5
            sigma = 0.5
            baseline_dilation = 0.2
            
            # Precision modulates dilation magnitude
            dilation_magnitude = baseline_dilation * (0.3 + 0.7 * Pi_i / 2.5)
            
            pupil = dilation_magnitude * np.exp(-((t - peak_time)**2) / (2 * sigma**2))
        else:
            pupil = 0.05 * np.exp(-((t - 1.5)**2) / (2 * 0.5**2))
        
        # Slow drift
        pupil += 0.05 * np.sin(2 * np.pi * 0.1 * t)
        
        # Add noise
        pupil += np.random.normal(0, 0.02, n_samples)
        
        # Random blink artifacts
        if np.random.rand() < blink_prob:
            blink_time = np.random.uniform(0.5, 2.5)
            blink_idx = int(blink_time * self.fs)
            blink_duration = int(0.15 * self.fs)
            end_idx = min(blink_idx + blink_duration, n_samples)
            pupil[blink_idx:end_idx] = np.nan
        
        return pupil
    
    def generate_multi_channel_eeg(
        self,
        S_t: float,
        theta_t: float,
        ignition: bool,
        n_channels: int = 64,
        duration: float = 1.0
    ) -> np.ndarray:
        """
        Generate multi-channel EEG with realistic spatial patterns
        
        Args:
            S_t: Surprise signal
            theta_t: Threshold
            ignition: Ignition state
            n_channels: Number of EEG channels
            duration: Duration in seconds
        
        Returns:
            EEG array (channels × timepoints)
        """
        n_samples = int(duration * self.fs)
        eeg = np.zeros((n_channels, n_samples))
        
        # Generate P3b (strongest at Pz - channel 31)
        p3b = self.generate_P3b_waveform(S_t, theta_t, ignition, duration)
        
        # Generate gamma
        gamma = self.generate_gamma_burst(ignition, S_t, duration)
        
        # Spatial distribution (simplified topography)
        for ch in range(n_channels):
            # Distance from Pz (centro-parietal)
            dist_from_pz = np.abs(ch - 31) / 31.0
            
            # P3b falloff
            p3b_weight = np.exp(-2 * dist_from_pz)
            
            # Gamma more distributed
            gamma_weight = np.exp(-0.5 * dist_from_pz)
            
            # Combine components
            eeg[ch] = p3b_weight * p3b + 0.3 * gamma_weight * gamma
            
            # Add channel-specific noise
            eeg[ch] += self._pink_noise(n_samples, 1.0)
            
            # Add alpha (8-13 Hz) - stronger in occipital
            if ch > 48:  # Posterior channels
                alpha_amp = 2.0
            else:
                alpha_amp = 0.5
            
            alpha_freq = np.random.uniform(8, 13)
            t = np.linspace(0, duration, n_samples)
            eeg[ch] += alpha_amp * np.sin(2 * np.pi * alpha_freq * t)
        
        return eeg
    
    def _pink_noise(self, n_samples: int, amplitude: float) -> np.ndarray:
        """Generate 1/f pink noise"""
        pink = np.zeros(n_samples)
        for octave in range(5):
            step = 2 ** octave
            n_octave_samples = n_samples // step + 1
            pink += np.repeat(
                np.random.randn(n_octave_samples), step
            )[:n_samples] / (octave + 1)
        
        # Normalize
        if np.std(pink) > 1e-10:
            pink = amplitude * pink / np.std(pink)
        
        return pink


# =============================================================================
# PART 2: COMPETING MODEL IMPLEMENTATIONS
# =============================================================================

class StandardPredictiveProcessingGenerator:
    """Continuous processing without threshold ignition"""
    
    def __init__(self, fs: int = 1000):
        self.fs = fs
        self.signal_gen = APGISyntheticSignalGenerator(fs)
    
    def generate_trial(
        self,
        epsilon_e: float,
        epsilon_i: float,
        Pi_e: float,
        Pi_i: float
    ) -> Dict[str, np.ndarray]:
        """
        Generate signals without ignition mechanism
        
        Key difference: Continuous, graded response
        No P3b signature, only early components
        """
        # Continuous response amplitude
        response_amp = Pi_e * np.abs(epsilon_e) + Pi_i * np.abs(epsilon_i)
        
        duration = 0.8
        n_samples = int(duration * self.fs)
        t = np.linspace(0, duration, n_samples)
        
        # Only early components (N1 at 100ms, P2 at 200ms)
        n1_amp = -response_amp * 0.5
        p2_amp = response_amp * 0.3
        
        erp = (n1_amp * np.exp(-((t - 0.10)**2) / (2 * 0.02**2)) +
               p2_amp * np.exp(-((t - 0.20)**2) / (2 * 0.03**2)))
        
        # No late P3b component
        erp += self.signal_gen._pink_noise(n_samples, 0.5)
        
        # Generate multi-channel (broadcast early components)
        n_channels = 64
        eeg = np.tile(erp, (n_channels, 1))
        
        # Add channel noise
        for ch in range(n_channels):
            eeg[ch] += self.signal_gen._pink_noise(n_samples, 1.0)
        
        # HEP present (interoception still processed)
        hep = self.signal_gen.generate_HEP_waveform(Pi_i, epsilon_i)
        
        # Minimal pupil response (no ignition)
        pupil = 0.05 * np.exp(-((t - 1.0)**2) / (2 * 0.5**2))
        pupil += np.random.normal(0, 0.02, len(pupil))
        
        return {
            'eeg': eeg,
            'hep': hep,
            'pupil': pupil,
            'ignition': False,  # Never ignites
            'S_t': response_amp,
            'model': 'StandardPP'
        }


class GlobalWorkspaceOnlyGenerator:
    """Ignition without interoceptive precision weighting"""
    
    def __init__(self, fs: int = 1000):
        self.fs = fs
        self.signal_gen = APGISyntheticSignalGenerator(fs)
        self.apgi_system = APGIDynamicalSystem()
    
    def generate_trial(
        self,
        epsilon_e: float,
        Pi_e: float,
        theta_t: float
    ) -> Dict[str, np.ndarray]:
        """
        Generate signals with ignition but no somatic bias
        
        Key difference: No β·Π_i term in surprise equation
        P3b present, but HEP not modulated by ignition
        """
        # Only exteroceptive signals
        S_traj, B_traj, ignition = self.apgi_system.simulate_surprise_accumulation(
            epsilon_e=epsilon_e,
            epsilon_i=0.0,  # No interoceptive contribution
            Pi_e=Pi_e,
            Pi_i=0.0,
            beta=0.0,  # No somatic bias
            theta_t=theta_t
        )
        
        S_final = S_traj[int(0.5 / 0.001)]
        
        # Generate EEG with P3b if ignition
        eeg = self.signal_gen.generate_multi_channel_eeg(
            S_final, theta_t, ignition
        )
        
        # HEP present but NOT modulated by ignition
        # (fixed low amplitude)
        hep = self.signal_gen.generate_HEP_waveform(
            Pi_i=0.5,  # Fixed low value
            epsilon_i=0.1
        )
        
        # Pupil response if ignition
        pupil = self.signal_gen.generate_pupil_response(
            Pi_i=1.0,
            ignition=ignition
        )
        
        return {
            'eeg': eeg,
            'hep': hep,
            'pupil': pupil,
            'ignition': ignition,
            'S_t': S_final,
            'model': 'GWTOnly'
        }


class ContinuousIntegrationGenerator:
    """Graded consciousness without phase transition"""
    
    def __init__(self, fs: int = 1000):
        self.fs = fs
        self.signal_gen = APGISyntheticSignalGenerator(fs)
    
    def generate_trial(
        self,
        epsilon_e: float,
        epsilon_i: float,
        Pi_e: float,
        Pi_i: float,
        beta: float
    ) -> Dict[str, np.ndarray]:
        """
        Generate signals with continuous integration
        
        Key difference: No discrete threshold
        Response scales smoothly with surprise
        """
        # Continuous accumulation (no threshold)
        S = Pi_e * np.abs(epsilon_e) + beta * Pi_i * np.abs(epsilon_i)
        
        # Soft saturation (no sharp ignition)
        response_strength = np.tanh(S / 2.0)
        
        duration = 0.8
        n_samples = int(duration * self.fs)
        t = np.linspace(0, duration, n_samples)
        
        # Spread activation (no localized P3b peak)
        # Response builds gradually
        envelope = response_strength * (1 - np.exp(-t / 0.2))
        
        # Generate distributed ERP
        n_channels = 64
        eeg = np.zeros((n_channels, n_samples))
        
        for ch in range(n_channels):
            eeg[ch] = 3.0 * envelope
            eeg[ch] += self.signal_gen._pink_noise(n_samples, 1.0)
        
        # HEP with graded modulation
        hep = self.signal_gen.generate_HEP_waveform(
            Pi_i * response_strength, epsilon_i
        )
        
        # Graded pupil response
        pupil_mag = 0.2 * response_strength
        pupil = pupil_mag * np.exp(-((t - 1.5)**2) / (2 * 0.5**2))
        pupil += np.random.normal(0, 0.02, len(pupil))
        
        return {
            'eeg': eeg,
            'hep': hep,
            'pupil': pupil,
            'ignition': False,  # No discrete ignition
            'S_t': S,
            'model': 'Continuous'
        }


# =============================================================================
# PART 3: DATASET GENERATION
# =============================================================================

@dataclass
class TrialParameters:
    """Parameters for a single trial"""
    epsilon_e: float
    epsilon_i: float
    Pi_e: float
    Pi_i: float
    beta: float
    theta_t: float
    model_name: str


class APGIDatasetGenerator:
    """Generate complete synthetic dataset with all models"""
    
    def __init__(self, fs: int = 1000):
        self.fs = fs
        
        # Initialize all generators
        self.apgi_gen = APGISyntheticSignalGenerator(fs)
        self.apgi_system = APGIDynamicalSystem()
        
        self.generators = {
            'APGI': self._generate_apgi_trial,
            'StandardPP': StandardPredictiveProcessingGenerator(fs),
            'GWTOnly': GlobalWorkspaceOnlyGenerator(fs),
            'Continuous': ContinuousIntegrationGenerator(fs)
        }
    
    def sample_physiological_parameters(self) -> TrialParameters:
        """Sample realistic parameter ranges"""
        return TrialParameters(
            epsilon_e=np.random.uniform(-0.5, 0.5),
            epsilon_i=np.random.uniform(-0.3, 0.3),
            Pi_e=np.random.gamma(2.0, 0.5),  # Typically 0.5-3.0
            Pi_i=np.random.gamma(2.0, 0.5),
            beta=np.random.normal(1.15, 0.25),
            theta_t=np.random.normal(0.55, 0.15),
            model_name=''
        )
    
    def _generate_apgi_trial(self, params: TrialParameters) -> Dict:
        """Generate APGI trial with full dynamics"""
        # Run APGI dynamics
        S_traj, B_traj, ignition = self.apgi_system.simulate_surprise_accumulation(
            epsilon_e=params.epsilon_e,
            epsilon_i=params.epsilon_i,
            Pi_e=params.Pi_e,
            Pi_i=params.Pi_i,
            beta=params.beta,
            theta_t=params.theta_t
        )
        
        S_final = S_traj[int(0.5 / 0.001)]
        
        # Generate signals
        eeg = self.apgi_gen.generate_multi_channel_eeg(
            S_final, params.theta_t, ignition
        )
        
        hep = self.apgi_gen.generate_HEP_waveform(
            params.Pi_i, params.epsilon_i
        )
        
        pupil = self.apgi_gen.generate_pupil_response(
            params.Pi_i, ignition
        )
        
        return {
            'eeg': eeg,
            'hep': hep,
            'pupil': pupil,
            'ignition': ignition,
            'S_t': S_final,
            'theta_t': params.theta_t,
            'Pi_i': params.Pi_i,
            'beta': params.beta,
            'model': 'APGI'
        }
    
    def generate_dataset(
        self,
        n_trials_per_model: int = 5000,
        save_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate complete dataset with all models
        
        Args:
            n_trials_per_model: Number of trials per model
            save_path: Optional path to save dataset
        
        Returns:
            Dictionary with all data arrays
        """
        print(f"Generating dataset: {n_trials_per_model} trials × 4 models = "
              f"{4 * n_trials_per_model} total trials")
        
        dataset = {
            'eeg': [],
            'hep': [],
            'pupil': [],
            'ignition_labels': [],
            'S_values': [],
            'model_labels': [],
            'model_names': []
        }
        
        model_names = ['APGI', 'StandardPP', 'GWTOnly', 'Continuous']
        
        for model_idx, model_name in enumerate(model_names):
            print(f"\nGenerating {model_name} trials...")
            
            for trial_idx in tqdm(range(n_trials_per_model)):
                # Sample parameters
                params = self.sample_physiological_parameters()
                params.model_name = model_name
                
                # Generate trial
                if model_name == 'APGI':
                    trial_data = self._generate_apgi_trial(params)
                elif model_name == 'StandardPP':
                    trial_data = self.generators['StandardPP'].generate_trial(
                        params.epsilon_e, params.epsilon_i,
                        params.Pi_e, params.Pi_i
                    )
                elif model_name == 'GWTOnly':
                    trial_data = self.generators['GWTOnly'].generate_trial(
                        params.epsilon_e, params.Pi_e, params.theta_t
                    )
                else:  # Continuous
                    trial_data = self.generators['Continuous'].generate_trial(
                        params.epsilon_e, params.epsilon_i,
                        params.Pi_e, params.Pi_i, params.beta
                    )
                
                # Store data
                dataset['eeg'].append(trial_data['eeg'])
                dataset['hep'].append(trial_data['hep'])
                dataset['pupil'].append(trial_data['pupil'])
                dataset['ignition_labels'].append(int(trial_data['ignition']))
                dataset['S_values'].append(trial_data['S_t'])
                dataset['model_labels'].append(model_idx)
                dataset['model_names'].append(model_name)
        
        # Convert to arrays
        dataset['eeg'] = np.array(dataset['eeg'])
        dataset['hep'] = np.array(dataset['hep'])
        dataset['pupil'] = np.array(dataset['pupil'])
        dataset['ignition_labels'] = np.array(dataset['ignition_labels'])
        dataset['S_values'] = np.array(dataset['S_values'])
        dataset['model_labels'] = np.array(dataset['model_labels'])
        
        print(f"\nDataset generated:")
        print(f"  EEG shape: {dataset['eeg'].shape}")
        print(f"  HEP shape: {dataset['hep'].shape}")
        print(f"  Pupil shape: {dataset['pupil'].shape}")
        print(f"  Ignition distribution: {np.bincount(dataset['ignition_labels'])}")
        
        if save_path:
            np.savez_compressed(save_path, **dataset)
            print(f"  Saved to: {save_path}")
        
        return dataset


# =============================================================================
# PART 4: PYTORCH DATASETS
# =============================================================================

class IgnitionClassificationDataset(Dataset):
    """Dataset for Task 1A: Binary ignition classification"""
    
    def __init__(self, eeg_data: np.ndarray, labels: np.ndarray):
        self.eeg = torch.FloatTensor(eeg_data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.eeg[idx], self.labels[idx]


class ModelIdentificationDataset(Dataset):
    """Dataset for Task 1B: Multi-class model identification"""
    
    def __init__(
        self,
        eeg_data: np.ndarray,
        hep_data: np.ndarray,
        pupil_data: np.ndarray,
        model_labels: np.ndarray
    ):
        self.eeg = torch.FloatTensor(eeg_data)
        self.hep = torch.FloatTensor(hep_data)
        self.pupil = torch.FloatTensor(pupil_data)
        self.labels = torch.LongTensor(model_labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'eeg': self.eeg[idx],
            'hep': self.hep[idx],
            'pupil': self.pupil[idx]
        }, self.labels[idx]


# =============================================================================
# PART 5: NEURAL NETWORK ARCHITECTURES
# =============================================================================

class IgnitionClassifier(nn.Module):
    """
    1D CNN for EEG-based ignition classification
    
    Architecture:
        - Multi-scale temporal convolutions
        - Spatial attention across channels
        - Temporal attention
        - Binary classification head
    """
    
    def __init__(
        self,
        n_channels: int = 64,
        n_timepoints: int = 1000,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        
        # Multi-scale temporal convolutions
        self.temporal_conv1 = nn.Conv1d(n_channels, 64, kernel_size=25, padding=12)
        self.temporal_conv2 = nn.Conv1d(64, 128, kernel_size=15, padding=7)
        self.temporal_conv3 = nn.Conv1d(128, 256, kernel_size=7, padding=3)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Binary classification
        )
    
    def forward(self, x):
        # x: (batch, channels, timepoints)
        
        # Temporal convolutions
        x = F.relu(self.bn1(self.temporal_conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.temporal_conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.temporal_conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # x: (batch, 256, reduced_time)
        
        # Temporal attention
        x_permuted = x.permute(0, 2, 1)  # (batch, time, features)
        attention_weights = F.softmax(self.attention(x_permuted), dim=1)
        x_attended = torch.sum(x_permuted * attention_weights, dim=1)
        
        # Classification
        out = self.fc(x_attended)
        
        return out


class MultiModalFusionNetwork(nn.Module):
    """
    Multi-modal fusion network for model identification
    
    Inputs: EEG, HEP, Pupil
    Output: 4-class classification (APGI, StandardPP, GWTOnly, Continuous)
    """
    
    def __init__(
        self,
        n_eeg_channels: int = 64,
        n_eeg_time: int = 1000,
        n_hep_time: int = 600,
        n_pupil_time: int = 3000,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # EEG encoder
        self.eeg_conv1 = nn.Conv1d(n_eeg_channels, 64, kernel_size=25, padding=12)
        self.eeg_conv2 = nn.Conv1d(64, 128, kernel_size=15, padding=7)
        self.eeg_pool = nn.MaxPool1d(4)
        self.eeg_bn1 = nn.BatchNorm1d(64)
        self.eeg_bn2 = nn.BatchNorm1d(128)
        
        # HEP encoder (1D signal)
        self.hep_conv1 = nn.Conv1d(1, 32, kernel_size=15, padding=7)
        self.hep_conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.hep_pool = nn.MaxPool1d(3)
        self.hep_bn1 = nn.BatchNorm1d(32)
        self.hep_bn2 = nn.BatchNorm1d(64)
        
        # Pupil encoder (1D signal)
        self.pupil_conv1 = nn.Conv1d(1, 32, kernel_size=15, padding=7)
        self.pupil_conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.pupil_pool = nn.MaxPool1d(5)
        self.pupil_bn1 = nn.BatchNorm1d(32)
        self.pupil_bn2 = nn.BatchNorm1d(64)
        
        self.dropout = nn.Dropout(dropout)
        
        # Fusion layers
        # Compute output sizes after convolutions
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64 + 64, 256),  # Concatenated features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4)  # 4 models
        )
    
    def forward(self, eeg, hep, pupil):
        # EEG processing
        x_eeg = F.relu(self.eeg_bn1(self.eeg_conv1(eeg)))
        x_eeg = self.eeg_pool(x_eeg)
        x_eeg = F.relu(self.eeg_bn2(self.eeg_conv2(x_eeg)))
        x_eeg = self.eeg_pool(x_eeg)
        x_eeg = torch.mean(x_eeg, dim=2)  # Global average pooling
        
        # HEP processing
        hep = hep.unsqueeze(1)  # Add channel dimension
        x_hep = F.relu(self.hep_bn1(self.hep_conv1(hep)))
        x_hep = self.hep_pool(x_hep)
        x_hep = F.relu(self.hep_bn2(self.hep_conv2(x_hep)))
        x_hep = self.hep_pool(x_hep)
        x_hep = torch.mean(x_hep, dim=2)  # Global average pooling
        
        # Pupil processing
        pupil = pupil.unsqueeze(1)  # Add channel dimension
        # Handle NaN values from blinks
        pupil = torch.nan_to_num(pupil, nan=0.0)
        x_pupil = F.relu(self.pupil_bn1(self.pupil_conv1(pupil)))
        x_pupil = self.pupil_pool(x_pupil)
        x_pupil = F.relu(self.pupil_bn2(self.pupil_conv2(x_pupil)))
        x_pupil = self.pupil_pool(x_pupil)
        x_pupil = torch.mean(x_pupil, dim=2)  # Global average pooling
        
        # Concatenate all modalities
        x_fused = torch.cat([x_eeg, x_hep, x_pupil], dim=1)
        
        # Classification
        out = self.fusion(x_fused)
        
        return out


# =============================================================================
# PART 6: TRAINING FUNCTIONS
# =============================================================================

def train_ignition_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """Train binary ignition classifier"""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"Training on {device}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_eeg, batch_labels in train_loader:
            batch_eeg = batch_eeg.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_eeg)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_labels.size(0)
            train_correct += predicted.eq(batch_labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_eeg, batch_labels in val_loader:
                batch_eeg = batch_eeg.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_eeg)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_labels.size(0)
                val_correct += predicted.eq(batch_labels).sum().item()
                
                # Store for AUC calculation
                probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(batch_labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Calculate AUC
        val_auc = roc_auc_score(all_labels, all_probs)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                  f"Val AUC: {val_auc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


def train_model_identifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """Train multi-class model identifier"""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_data, batch_labels in train_loader:
            eeg = batch_data['eeg'].to(device)
            hep = batch_data['hep'].to(device)
            pupil = batch_data['pupil'].to(device)
            labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(eeg, hep, pupil)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                eeg = batch_data['eeg'].to(device)
                hep = batch_data['hep'].to(device)
                pupil = batch_data['pupil'].to(device)
                labels = batch_labels.to(device)
                
                outputs = model(eeg, hep, pupil)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


# =============================================================================
# PART 7: EVALUATION & FALSIFICATION
# =============================================================================

def evaluate_ignition_classifier(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """Comprehensive evaluation of ignition classifier"""
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_eeg, batch_labels in test_loader:
            batch_eeg = batch_eeg.to(device)
            outputs = model(batch_eeg)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='binary')
    auc = roc_auc_score(all_labels, all_probs)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'auc_roc': auc,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probs
    }


def evaluate_model_identifier(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """Comprehensive evaluation of model identifier"""
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            eeg = batch_data['eeg'].to(device)
            hep = batch_data['hep'].to(device)
            pupil = batch_data['pupil'].to(device)
            
            outputs = model(eeg, hep, pupil)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Per-class metrics
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Classification report
    model_names = ['APGI', 'StandardPP', 'GWTOnly', 'Continuous']
    report = classification_report(
        all_labels, all_predictions,
        target_names=model_names,
        output_dict=True
    )
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_predictions,
        'labels': all_labels
    }


class FalsificationChecker:
    """Check all falsification criteria for Protocol 1"""
    
    def __init__(self):
        self.criteria = {
            'F1.1': {
                'description': 'APGI ignition classification accuracy < 75%',
                'threshold': 0.75,
                'comparison': 'less_than'
            },
            'F1.2': {
                'description': 'APGI-GWT confusion > 40%',
                'threshold': 0.40,
                'comparison': 'greater_than'
            },
            'F1.3': {
                'description': 'Generalization to real data < 55%',
                'threshold': 0.55,
                'comparison': 'less_than'
            },
            'F1.4': {
                'description': 'StandardPP accuracy >= APGI accuracy',
                'threshold': None,
                'comparison': 'greater_equal'
            }
        }
    
    def check_F1_1(
        self,
        results_by_model: Dict[str, Dict]
    ) -> Tuple[bool, float]:
        """F1.1: APGI ignition classification < 75%"""
        apgi_accuracy = results_by_model['APGI']['accuracy']
        falsified = apgi_accuracy < self.criteria['F1.1']['threshold']
        return falsified, apgi_accuracy
    
    def check_F1_2(
        self,
        confusion_matrix: np.ndarray
    ) -> Tuple[bool, float]:
        """F1.2: APGI-GWT confusion > 40%"""
        # Extract confusion between APGI (0) and GWTOnly (2)
        apgi_to_gwt = confusion_matrix[0, 2] / confusion_matrix[0].sum()
        gwt_to_apgi = confusion_matrix[2, 0] / confusion_matrix[2].sum()
        avg_confusion = (apgi_to_gwt + gwt_to_apgi) / 2
        
        falsified = avg_confusion > self.criteria['F1.2']['threshold']
        return falsified, avg_confusion
    
    def check_F1_3(
        self,
        real_data_accuracy: float
    ) -> Tuple[bool, float]:
        """F1.3: Generalization to real data < 55%"""
        falsified = real_data_accuracy < self.criteria['F1.3']['threshold']
        return falsified, real_data_accuracy
    
    def check_F1_4(
        self,
        results_by_model: Dict[str, Dict]
    ) -> Tuple[bool, Tuple[float, float]]:
        """F1.4: StandardPP >= APGI accuracy"""
        apgi_acc = results_by_model['APGI']['accuracy']
        pp_acc = results_by_model['StandardPP']['accuracy']
        
        falsified = pp_acc >= apgi_acc
        return falsified, (apgi_acc, pp_acc)
    
    def generate_report(
        self,
        results_task_1a: Dict[str, Dict],
        results_task_1b: Dict,
        real_data_accuracy: float = None
    ) -> Dict:
        """Generate comprehensive falsification report"""
        
        report = {
            'falsified_criteria': [],
            'passed_criteria': [],
            'overall_falsified': False
        }
        
        # Check F1.1
        f1_1_falsified, apgi_acc = self.check_F1_1(results_task_1a)
        criterion_result = {
            'code': 'F1.1',
            'description': self.criteria['F1.1']['description'],
            'falsified': f1_1_falsified,
            'value': apgi_acc,
            'threshold': self.criteria['F1.1']['threshold']
        }
        
        if f1_1_falsified:
            report['falsified_criteria'].append(criterion_result)
        else:
            report['passed_criteria'].append(criterion_result)
        
        # Check F1.2
        f1_2_falsified, confusion_val = self.check_F1_2(
            results_task_1b['confusion_matrix']
        )
        criterion_result = {
            'code': 'F1.2',
            'description': self.criteria['F1.2']['description'],
            'falsified': f1_2_falsified,
            'value': confusion_val,
            'threshold': self.criteria['F1.2']['threshold']
        }
        
        if f1_2_falsified:
            report['falsified_criteria'].append(criterion_result)
        else:
            report['passed_criteria'].append(criterion_result)
        
        # Check F1.3 (if real data available)
        if real_data_accuracy is not None:
            f1_3_falsified, real_acc = self.check_F1_3(real_data_accuracy)
            criterion_result = {
                'code': 'F1.3',
                'description': self.criteria['F1.3']['description'],
                'falsified': f1_3_falsified,
                'value': real_acc,
                'threshold': self.criteria['F1.3']['threshold']
            }
            
            if f1_3_falsified:
                report['falsified_criteria'].append(criterion_result)
            else:
                report['passed_criteria'].append(criterion_result)
        
        # Check F1.4
        f1_4_falsified, (apgi_acc, pp_acc) = self.check_F1_4(results_task_1a)
        criterion_result = {
            'code': 'F1.4',
            'description': self.criteria['F1.4']['description'],
            'falsified': f1_4_falsified,
            'value': {'APGI': apgi_acc, 'StandardPP': pp_acc},
            'threshold': 'StandardPP < APGI'
        }
        
        if f1_4_falsified:
            report['falsified_criteria'].append(criterion_result)
        else:
            report['passed_criteria'].append(criterion_result)
        
        # Overall verdict
        report['overall_falsified'] = len(report['falsified_criteria']) > 0
        
        return report


# =============================================================================
# PART 8: VISUALIZATION
# =============================================================================

def plot_classification_results(
    results_task_1a: Dict[str, Dict],
    results_task_1b: Dict,
    save_path: str = 'protocol1_results.png'
):
    """Generate comprehensive visualization of results"""
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    model_names = ['APGI', 'StandardPP', 'GWTOnly', 'Continuous']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    # =========================================================================
    # Row 1: Task 1A - Per-model ignition classification
    # =========================================================================
    
    # Accuracy comparison
    ax1 = fig.add_subplot(gs[0, 0])
    accuracies = [results_task_1a[m]['accuracy'] * 100 for m in model_names]
    bars = ax1.bar(model_names, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=75, color='red', linestyle='--', linewidth=2, label='F1.1 Threshold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Task 1A: Ignition Classification Accuracy', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # F1 scores
    ax2 = fig.add_subplot(gs[0, 1])
    f1_scores = [results_task_1a[m]['f1_score'] for m in model_names]
    bars = ax2.bar(model_names, f1_scores, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('Task 1A: F1 Scores', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # AUC-ROC
    ax3 = fig.add_subplot(gs[0, 2])
    aucs = [results_task_1a[m]['auc_roc'] for m in model_names]
    bars = ax3.bar(model_names, aucs, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax3.set_title('Task 1A: AUC-ROC Scores', fontsize=13, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # =========================================================================
    # Row 2: Task 1B - Model identification confusion matrix
    # =========================================================================
    
    ax4 = fig.add_subplot(gs[1, :])
    cm = results_task_1b['confusion_matrix']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=model_names, yticklabels=model_names,
                ax=ax4, cbar_kws={'label': 'Proportion'})
    ax4.set_ylabel('True Model', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Predicted Model', fontsize=12, fontweight='bold')
    ax4.set_title('Task 1B: Model Identification Confusion Matrix', 
                  fontsize=13, fontweight='bold')
    
    # Add text box with APGI-GWT confusion
    apgi_gwt = (cm_normalized[0, 2] + cm_normalized[2, 0]) / 2
    textstr = f'APGI ↔ GWT Confusion: {apgi_gwt:.1%}\n(F1.2 Threshold: 40%)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax4.text(0.02, 0.98, textstr, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # =========================================================================
    # Row 3: ROC curves and additional metrics
    # =========================================================================
    
    # ROC curves for APGI model
    ax5 = fig.add_subplot(gs[2, 0])
    apgi_results = results_task_1a['APGI']
    fpr, tpr, _ = roc_curve(apgi_results['labels'], apgi_results['probabilities'])
    ax5.plot(fpr, tpr, color='#2E86AB', linewidth=2, 
            label=f"APGI (AUC={apgi_results['auc_roc']:.3f})")
    ax5.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance')
    ax5.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax5.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax5.set_title('ROC Curve - APGI Model', fontsize=12, fontweight='bold')
    ax5.legend(loc='lower right')
    ax5.grid(alpha=0.3)
    
    # Per-model confusion matrices (APGI only)
    ax6 = fig.add_subplot(gs[2, 1])
    apgi_cm = results_task_1a['APGI']['confusion_matrix']
    apgi_cm_norm = apgi_cm.astype('float') / apgi_cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(apgi_cm_norm, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=['No Ignition', 'Ignition'],
                yticklabels=['No Ignition', 'Ignition'],
                ax=ax6, cbar=False)
    ax6.set_title('APGI Ignition Confusion Matrix', fontsize=12, fontweight='bold')
    
    # Summary metrics table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('tight')
    ax7.axis('off')
    
    table_data = []
    table_data.append(['Model', 'Acc', 'F1', 'AUC'])
    for model in model_names:
        acc = f"{results_task_1a[model]['accuracy']:.3f}"
        f1 = f"{results_task_1a[model]['f1_score']:.3f}"
        auc = f"{results_task_1a[model]['auc_roc']:.3f}"
        table_data.append([model, acc, f1, auc])
    
    table = ax7.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax7.set_title('Summary Metrics', fontsize=12, fontweight='bold', pad=20)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Results visualization saved to: {save_path}")
    
    plt.show()


def print_falsification_report(report: Dict):
    """Print formatted falsification report"""
    
    print("\n" + "="*80)
    print("PROTOCOL 1 FALSIFICATION REPORT")
    print("="*80)
    
    print(f"\nOVERALL STATUS: ", end="")
    if report['overall_falsified']:
        print("❌ MODEL FALSIFIED")
    else:
        print("✅ MODEL VALIDATED")
    
    print(f"\nCriteria Passed: {len(report['passed_criteria'])}/4")
    print(f"Criteria Failed: {len(report['falsified_criteria'])}/4")
    
    if report['passed_criteria']:
        print("\n" + "-"*80)
        print("PASSED CRITERIA:")
        print("-"*80)
        for criterion in report['passed_criteria']:
            print(f"\n✅ {criterion['code']}: {criterion['description']}")
            if isinstance(criterion['value'], dict):
                for k, v in criterion['value'].items():
                    print(f"   {k}: {v:.3f}")
            else:
                print(f"   Value: {criterion['value']:.3f}")
            print(f"   Threshold: {criterion['threshold']}")
    
    if report['falsified_criteria']:
        print("\n" + "-"*80)
        print("FAILED CRITERIA (FALSIFICATIONS):")
        print("-"*80)
        for criterion in report['falsified_criteria']:
            print(f"\n❌ {criterion['code']}: {criterion['description']}")
            if isinstance(criterion['value'], dict):
                for k, v in criterion['value'].items():
                    print(f"   {k}: {v:.3f}")
            else:
                print(f"   Value: {criterion['value']:.3f}")
            print(f"   Threshold: {criterion['threshold']}")
    
    print("\n" + "="*80)


# =============================================================================
# PART 9: MAIN EXECUTION PIPELINE
# =============================================================================

def main():
    """Main execution pipeline for Protocol 1"""
    
    print("="*80)
    print("APGI PROTOCOL 1: SYNTHETIC DATA GENERATION & ML CLASSIFICATION")
    print("="*80)
    
    # Configuration
    config = {
        'n_trials_per_model': 5000,  # 5000 × 4 = 20,000 total trials
        'batch_size': 32,
        'epochs_task_1a': 50,
        'epochs_task_1b': 50,
        'learning_rate': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # =========================================================================
    # STEP 1: Generate Dataset
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: GENERATING SYNTHETIC DATASET")
    print("="*80)
    
    generator = APGIDatasetGenerator(fs=1000)
    dataset = generator.generate_dataset(
        n_trials_per_model=config['n_trials_per_model'],
        save_path='apgi_protocol1_dataset.npz'
    )
    
    # =========================================================================
    # STEP 2: Task 1A - Ignition Classification (Per Model)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: TASK 1A - BINARY IGNITION CLASSIFICATION")
    print("="*80)
    
    results_task_1a = {}
    model_names = ['APGI', 'StandardPP', 'GWTOnly', 'Continuous']
    
    for model_idx, model_name in enumerate(model_names):
        print(f"\n--- Training classifier for {model_name} ---")
        
        # Filter data for this model
        model_mask = dataset['model_labels'] == model_idx
        eeg_model = dataset['eeg'][model_mask]
        labels_model = dataset['ignition_labels'][model_mask]
        
        # Create dataset
        full_dataset = IgnitionClassificationDataset(eeg_model, labels_model)
        
        # Split train/val/test (60/20/20)
        n_total = len(full_dataset)
        n_train = int(0.6 * n_total)
        n_val = int(0.2 * n_total)
        n_test = n_total - n_train - n_val
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [n_train, n_val, n_test]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                               shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                                shuffle=False, num_workers=0)
        
        # Train classifier
        classifier = IgnitionClassifier(
            n_channels=64,
            n_timepoints=1000,
            dropout=0.5
        )
        
        trained_model, history = train_ignition_classifier(
            classifier, train_loader, val_loader,
            epochs=config['epochs_task_1a'],
            lr=config['learning_rate'],
            device=config['device']
        )
        
        # Evaluate on test set
        results = evaluate_ignition_classifier(
            trained_model, test_loader, device=config['device']
        )
        
        results_task_1a[model_name] = results
        
        print(f"\n{model_name} Results:")
        print(f"  Accuracy: {results['accuracy']:.3f}")
        print(f"  F1 Score: {results['f1_score']:.3f}")
        print(f"  AUC-ROC: {results['auc_roc']:.3f}")
    
    # =========================================================================
    # STEP 3: Task 1B - Model Identification
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: TASK 1B - MULTI-MODAL MODEL IDENTIFICATION")
    print("="*80)
    
    # Create multi-modal dataset
    full_dataset = ModelIdentificationDataset(
        dataset['eeg'],
        dataset['hep'],
        dataset['pupil'],
        dataset['model_labels']
    )
    
    # Split
    n_total = len(full_dataset)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    n_test = n_total - n_train - n_val
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [n_train, n_val, n_test]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=0)
    
    # Train model identifier
    model_identifier = MultiModalFusionNetwork(
        n_eeg_channels=64,
        n_eeg_time=1000,
        n_hep_time=600,
        n_pupil_time=3000,
        dropout=0.5
    )
    
    trained_identifier, history = train_model_identifier(
        model_identifier, train_loader, val_loader,
        epochs=config['epochs_task_1b'],
        lr=config['learning_rate'],
        device=config['device']
    )
    
    # Evaluate
    results_task_1b = evaluate_model_identifier(
        trained_identifier, test_loader, device=config['device']
    )
    
    print(f"\nModel Identification Results:")
    print(f"  Overall Accuracy: {results_task_1b['accuracy']:.3f}")
    print(f"\nPer-class F1 scores:")
    for model_name in model_names:
        f1 = results_task_1b['classification_report'][model_name]['f1-score']
        print(f"  {model_name}: {f1:.3f}")
    
    # =========================================================================
    # STEP 4: Falsification Analysis
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: FALSIFICATION ANALYSIS")
    print("="*80)
    
    checker = FalsificationChecker()
    falsification_report = checker.generate_report(
        results_task_1a,
        results_task_1b,
        real_data_accuracy=None  # Would need real data
    )
    
    print_falsification_report(falsification_report)
    
    # =========================================================================
    # STEP 5: Visualization
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_classification_results(
        results_task_1a,
        results_task_1b,
        save_path='protocol1_results.png'
    )
    
    # =========================================================================
    # STEP 6: Save Results
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 6: SAVING RESULTS")
    print("="*80)
    
    # Save comprehensive results
    results_summary = {
        'config': config,
        'task_1a': {
            model: {
                'accuracy': float(results['accuracy']),
                'f1_score': float(results['f1_score']),
                'auc_roc': float(results['auc_roc'])
            }
            for model, results in results_task_1a.items()
        },
        'task_1b': {
            'accuracy': float(results_task_1b['accuracy']),
            'classification_report': results_task_1b['classification_report']
        },
        'falsification': falsification_report
    }
    
    with open('protocol1_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("✅ Results saved to: protocol1_results.json")
    
    print("\n" + "="*80)
    print("PROTOCOL 1 EXECUTION COMPLETE")
    print("="*80)
    
    return results_summary


if __name__ == "__main__":
    main()
