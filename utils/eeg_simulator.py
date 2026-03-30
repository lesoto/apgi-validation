"""
EEG Waveform Simulator for APGI Validation

Implements cardiac-phase aligned epochs, computes HEP amplitude (200–600ms post-R-wave),
and P3b amplitude/latency (300–600ms). This is essential for the core mechanism test
(Πⁱ → HEP → P3b coupling) to be testable in silico.
"""

import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class EEGSimulator:
    """
    Minimal EEG waveform simulator for APGI validation.

    Generates cardiac-phase aligned epochs, computes HEP amplitude (200–600ms post-R-wave),
    and P3b amplitude/latency (300–600ms).
    """

    def __init__(
        self,
        sampling_rate: float = 1000.0,  # Hz
        eeg_duration: float = 1.5,  # seconds
        cardiac_cycle_duration: float = 0.9,  # seconds (average heart rate ~67 bpm)
        r_wave_duration: float = 0.05,  # seconds
        hep_window_start: float = 0.2,  # seconds post-R-wave (200ms)
        hep_window_end: float = 0.6,  # seconds post-R-wave (600ms)
        p3b_window_start: float = 0.3,  # seconds post-stimulus (300ms)
        p3b_window_end: float = 0.6,  # seconds post-stimulus (600ms)
        noise_level: float = 0.1,
    ):
        """
        Initialize EEG simulator.

        Args:
            sampling_rate: EEG sampling rate in Hz
            eeg_duration: Duration of each EEG epoch in seconds
            cardiac_cycle_duration: Duration of cardiac cycle in seconds
            r_wave_duration: Duration of R-wave in seconds
            hep_window_start: HEP analysis window start post-R-wave (seconds)
            hea_window_end: HEP analysis window end post-R-wave (seconds)
            p3b_window_start: P3b analysis window start post-stimulus (seconds)
            p3b_window_end: P3b analysis window end post-stimulus (seconds)
            noise_level: Noise level (standard deviation of background noise)
        """
        self.sampling_rate = sampling_rate
        self.eeg_duration = eeg_duration
        self.cardiac_cycle_duration = cardiac_cycle_duration
        self.r_wave_duration = r_wave_duration
        self.hep_window_start = hep_window_start
        self.hep_window_end = hep_window_end
        self.p3b_window_start = p3b_window_start
        self.p3b_window_end = p3b_window_end
        self.noise_level = noise_level

        # Convert to samples
        self.n_samples = int(sampling_rate * eeg_duration)
        self.r_wave_samples = int(sampling_rate * r_wave_duration)
        self.hep_start_sample = int(sampling_rate * hep_window_start)
        self.hep_end_sample = int(sampling_rate * hep_window_end)
        self.p3b_start_sample = int(sampling_rate * p3b_window_start)
        self.p3b_end_sample = int(sampling_rate * p3b_window_end)

        logger.info(
            f"EEGSimulator initialized: {sampling_rate}Hz, "
            f"{eeg_duration}s epochs, "
            f"HEP window: {hep_window_start}-{hep_window_end}s, "
            f"P3b window: {p3b_window_start}-{p3b_window_end}s"
        )

    def generate_r_wave(
        self,
        amplitude: float = 1.0,
        timing_jitter: float = 0.01,
    ) -> np.ndarray:
        """
        Generate R-wave waveform.

        Args:
            amplitude: Peak amplitude of R-wave
            timing_jitter: Random timing jitter in seconds

        Returns:
            R-wave waveform array (aligned to full EEG duration)
        """
        t = np.linspace(0, self.r_wave_duration, self.r_wave_samples)

        # Add timing jitter
        jitter_samples = int(timing_jitter * self.sampling_rate)
        if jitter_samples > 0:
            jitter = np.random.randint(-jitter_samples, jitter_samples + 1)
            t = t + jitter / self.sampling_rate
            t = np.clip(t, 0, self.r_wave_duration - 1 / self.sampling_rate)

        # R-wave shape (Gaussian-like)
        r_wave = amplitude * np.exp(-((t - self.r_wave_duration / 2) ** 2 / (0.01**2)))

        # Pad to full EEG duration (R-wave at time 0)
        r_wave_full = np.zeros(self.n_samples)
        r_wave_end_sample = len(r_wave)

        if r_wave_end_sample <= self.n_samples:
            r_wave_full[0:r_wave_end_sample] = r_wave

        return r_wave_full

    def generate_hep(
        self,
        r_wave_time: float,
        amplitude: float = 0.15,
        modulation: float = 1.0,
    ) -> np.ndarray:
        """
        Generate Heartbeat-Evoked Potential (HEP) 200-600ms post-R-wave.

        Args:
            r_wave_time: Time of R-wave in seconds
            amplitude: Base HEP amplitude
            modulation: Modulation factor (e.g., by interoceptive precision Πⁱ)

        Returns:
            HEP waveform array (aligned to full EEG duration)
        """
        # HEP onset delay after R-wave
        hep_onset = r_wave_time + 0.15  # 150ms delay
        hep_duration = 0.45  # 450ms duration

        hep_samples = int(self.sampling_rate * hep_duration)
        t_hep = np.linspace(0, hep_duration, hep_samples)

        # HEP shape (slow positive deflection)
        hep = amplitude * modulation * np.exp(-((t_hep - 0.15) ** 2 / (0.15**2)))

        # Pad to full EEG duration
        hep_full = np.zeros(self.n_samples)
        hep_start_sample = int(hep_onset * self.sampling_rate)
        hep_end_sample = hep_start_sample + len(hep)

        if hep_end_sample <= self.n_samples:
            hep_full[hep_start_sample:hep_end_sample] = hep

        return hep_full

    def generate_p3b(
        self,
        stimulus_time: float,
        amplitude: float = 0.3,
        latency: float = 0.35,
        modulation: float = 1.0,
    ) -> np.ndarray:
        """
        Generate P3b ERP 300-600ms post-stimulus.

        Args:
            stimulus_time: Time of stimulus presentation in seconds
            amplitude: Base P3b amplitude
            latency: P3b latency in seconds
            modulation: Modulation factor (e.g., by precision cueing)

        Returns:
            P3b waveform array (aligned to full EEG duration)
        """
        # P3b onset delay after stimulus
        p3b_onset = stimulus_time + latency
        p3b_duration = 0.25  # 250ms duration

        p3b_samples = int(self.sampling_rate * p3b_duration)
        t_p3b = np.linspace(0, p3b_duration, p3b_samples)

        # P3b shape (slow positive deflection)
        p3b = amplitude * modulation * np.exp(-((t_p3b - 0.1) ** 2 / (0.12**2)))

        # Pad to full EEG duration
        p3b_full = np.zeros(self.n_samples)
        p3b_start_sample = int(p3b_onset * self.sampling_rate)
        p3b_end_sample = p3b_start_sample + len(p3b)

        if p3b_end_sample <= self.n_samples:
            p3b_full[p3b_start_sample:p3b_end_sample] = p3b

        return p3b_full

    def generate_background_eeg(
        self,
        alpha_power: float = 0.5,
        beta_power: float = 0.3,
        theta_power: float = 0.2,
    ) -> np.ndarray:
        """
        Generate background EEG with oscillatory components.

        Args:
            alpha_power: Power in alpha band (8-12 Hz)
            beta_power: Power in beta band (13-30 Hz)
            theta_power: Power in theta band (4-7 Hz)

        Returns:
            Background EEG array
        """
        t = np.linspace(0, self.eeg_duration, self.n_samples)

        # Generate oscillatory components
        alpha = alpha_power * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
        beta = beta_power * np.sin(2 * np.pi * 20 * t)  # 20 Hz beta
        theta = theta_power * np.sin(2 * np.pi * 6 * t)  # 6 Hz theta

        # Combine with random noise
        noise = np.random.normal(0, self.noise_level, self.n_samples)
        background = alpha + beta + theta + noise

        return background

    def generate_epoch(
        self,
        r_wave_amplitude: float = 1.0,
        hep_amplitude: float = 0.15,
        hep_modulation: float = 1.0,
        stimulus_time: Optional[float] = None,
        p3b_amplitude: float = 0.3,
        p3b_latency: float = 0.35,
        p3b_modulation: float = 1.0,
        alpha_power: float = 0.5,
        beta_power: float = 0.3,
        theta_power: float = 0.2,
    ) -> Dict[str, np.ndarray]:
        """
        Generate a complete EEG epoch with R-wave, HEP, P3b, and background.

        Args:
            r_wave_amplitude: R-wave amplitude
            hep_amplitude: HEP base amplitude
            hep_modulation: HEP modulation factor (by Πⁱ)
            stimulus_time: Stimulus presentation time (None if no stimulus)
            p3b_amplitude: P3b base amplitude
            p3b_latency: P3b latency
            p3b_modulation: P3b modulation factor (by precision cueing)
            alpha_power: Alpha band power
            beta_power: Beta band power
            theta_power: Theta band power

        Returns:
            Dictionary with:
                - 'eeg': Full EEG signal
                - 'r_wave': R-wave component
                - 'hep': HEP component
                - 'p3b': P3b component (if stimulus_time provided)
                - 'background': Background EEG
        """
        # Generate components
        r_wave = self.generate_r_wave(r_wave_amplitude)
        hep = self.generate_hep(0.0, hep_amplitude, hep_modulation)
        background = self.generate_background_eeg(alpha_power, beta_power, theta_power)

        # Combine
        eeg = r_wave + hep + background

        result = {
            "eeg": eeg,
            "r_wave": r_wave,
            "hep": hep,
            "background": background,
        }

        # Add P3b if stimulus provided
        if stimulus_time is not None:
            p3b = self.generate_p3b(
                stimulus_time, p3b_amplitude, p3b_latency, p3b_modulation
            )
            eeg = eeg + p3b
            result["eeg"] = eeg
            result["p3b"] = p3b

        return result

    def compute_hep_amplitude(
        self,
        eeg_epoch: np.ndarray,
        r_wave_time: float = 0.0,
    ) -> float:
        """
        Compute HEP amplitude from 200-600ms post-R-wave.

        Args:
            eeg_epoch: EEG epoch data
            r_wave_time: Time of R-wave in seconds

        Returns:
            Mean HEP amplitude in the analysis window
        """
        hep_window = eeg_epoch[self.hep_start_sample : self.hep_end_sample]
        hep_amplitude = np.mean(np.abs(hep_window))

        return hep_amplitude

    def compute_p3b_amplitude(
        self,
        eeg_epoch: np.ndarray,
        stimulus_time: float,
    ) -> float:
        """
        Compute P3b amplitude from 300-600ms post-stimulus.

        Args:
            eeg_epoch: EEG epoch data
            stimulus_time: Time of stimulus presentation in seconds

        Returns:
            Mean P3b amplitude in the analysis window
        """
        p3b_window = eeg_epoch[self.p3b_start_sample : self.p3b_end_sample]
        p3b_amplitude = np.mean(np.abs(p3b_window))

        return p3b_amplitude

    def compute_hep_p3b_coupling(
        self,
        hep_amplitude: float,
        p3b_amplitude: float,
        baseline_hep: float = 0.15,
        baseline_p3b: float = 0.3,
    ) -> Dict[str, float]:
        """
        Compute HEP-P3b coupling metrics.

        Args:
            hep_amplitude: Measured HEP amplitude
            p3b_amplitude: Measured P3b amplitude
            baseline_hep: Baseline HEP amplitude (no modulation)
            baseline_p3b: Baseline P3b amplitude (no modulation)

        Returns:
            Dictionary with coupling metrics:
                - 'hep_modulation': HEP modulation ratio
                - 'p3b_modulation': P3b modulation ratio
                - 'hep_p3b_ratio': HEP:P3b amplitude ratio
                - 'coupling_strength': Overall coupling strength
        """
        hep_modulation = hep_amplitude / baseline_hep if baseline_hep > 0 else 0
        p3b_modulation = p3b_amplitude / baseline_p3b if baseline_p3b > 0 else 0
        hep_p3b_ratio = hep_amplitude / p3b_amplitude if p3b_amplitude > 0 else 0
        coupling_strength = hep_modulation * p3b_modulation

        return {
            "hep_modulation": hep_modulation,
            "p3b_modulation": p3b_modulation,
            "hep_p3b_ratio": hep_p3b_ratio,
            "coupling_strength": coupling_strength,
        }

    def simulate_cardiac_phase_alignment(
        self,
        n_epochs: int = 100,
        cardiac_cycle_jitter: float = 0.05,
    ) -> np.ndarray:
        """
        Simulate cardiac phase alignment across epochs.

        Args:
            n_epochs: Number of epochs to generate
            cardiac_cycle_jitter: Random jitter in cardiac cycle duration

        Returns:
            Array of R-wave times for each epoch
        """
        r_wave_times = []
        current_time = 0.0

        for i in range(n_epochs):
            # Add jitter to cardiac cycle duration
            jitter = np.random.uniform(-cardiac_cycle_jitter, cardiac_cycle_jitter)
            cycle_duration = self.cardiac_cycle_duration + jitter

            current_time += cycle_duration
            r_wave_times.append(current_time)

        return np.array(r_wave_times)


def create_default_simulator() -> EEGSimulator:
    """
    Create EEG simulator with default parameters.

    Returns:
        EEGSimulator instance with default parameters
    """
    return EEGSimulator(
        sampling_rate=1000.0,
        eeg_duration=1.5,
        cardiac_cycle_duration=0.9,
        r_wave_duration=0.05,
        hep_window_start=0.2,
        hep_window_end=0.6,
        p3b_window_start=0.3,
        p3b_window_end=0.6,
        noise_level=0.1,
    )


if __name__ == "__main__":
    # Test the simulator
    simulator = create_default_simulator()

    # Generate an epoch with both HEP and P3b
    epoch = simulator.generate_epoch(
        r_wave_amplitude=1.0,
        hep_amplitude=0.15,
        hep_modulation=1.2,
        stimulus_time=0.5,
        p3b_amplitude=0.3,
        p3b_latency=0.35,
        p3b_modulation=1.5,
    )

    print(f"EEG shape: {epoch['eeg'].shape}")
    print(f"HEP amplitude: {simulator.compute_hep_amplitude(epoch['eeg']):.4f}")
    print(f"P3b amplitude: {simulator.compute_p3b_amplitude(epoch['eeg'], 0.5):.4f}")

    coupling = simulator.compute_hep_p3b_coupling(
        simulator.compute_hep_amplitude(epoch["eeg"]),
        simulator.compute_p3b_amplitude(epoch["eeg"], 0.5),
    )
    print(f"HEP-P3b coupling: {coupling}")
