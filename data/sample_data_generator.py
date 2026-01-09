#!/usr/bin/env python3
"""
Sample Data Generator for APGI Framework
======================================

Generates realistic sample datasets for testing and demonstration purposes.
Includes EEG, pupil, EDA, and other physiological signals.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path


class SampleDataGenerator:
    """Generate realistic sample data for APGI framework testing."""

    def __init__(self, sampling_rate=1000, duration=60):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.n_samples = int(sampling_rate * duration)
        self.time_vector = np.linspace(0, duration, self.n_samples)

    def generate_eeg_data(self, include_artifacts=True):
        """Generate realistic EEG data with P300 components."""
        # Base EEG signal (1/f noise + alpha rhythm)
        frequencies = np.array([1, 2, 4, 8, 10, 20, 40])
        amplitudes = np.array([2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.2])

        eeg_signal = np.zeros(self.n_samples)
        for freq, amp in zip(frequencies, amplitudes):
            phase = np.random.uniform(0, 2 * np.pi)
            eeg_signal += amp * np.sin(2 * np.pi * freq * self.time_vector + phase)

        # Add 1/f noise
        pink_noise = self._generate_pink_noise(self.n_samples) * 0.5
        eeg_signal += pink_noise

        # Add P300 events (every 10-15 seconds)
        p300_events = []
        event_time = 10.0
        while event_time < self.duration - 5:
            event_idx = int(event_time * self.sampling_rate)
            if event_idx < self.n_samples - 300:  # Ensure space for P300
                # P300 waveform (positive peak around 300ms)
                p300_template = self._generate_p300_waveform()
                p300_start = event_idx
                p300_end = min(p300_start + len(p300_template), self.n_samples)
                eeg_signal[p300_start:p300_end] += p300_template[
                    : p300_end - p300_start
                ]
                p300_events.append(event_time)

                # Next event after random interval
                event_time += np.random.uniform(10, 15)

        # Add artifacts if requested
        if include_artifacts:
            # Eye blinks
            for _ in range(np.random.randint(5, 15)):
                blink_idx = np.random.randint(0, self.n_samples - 100)
                blink_template = self._generate_blink_waveform()
                blink_end = min(blink_idx + len(blink_template), self.n_samples)
                eeg_signal[blink_idx:blink_end] += blink_template[
                    : blink_end - blink_idx
                ]

            # Muscle artifacts
            for _ in range(np.random.randint(3, 8)):
                artifact_idx = np.random.randint(0, self.n_samples - 200)
                artifact_duration = np.random.randint(50, 200)
                artifact_end = min(artifact_idx + artifact_duration, self.n_samples)
                eeg_signal[artifact_idx:artifact_end] += np.random.normal(
                    0, 5, artifact_end - artifact_idx
                )

        return eeg_signal, p300_events

    def generate_pupil_data(self, task_related=True):
        """Generate realistic pupil diameter data."""
        # Base pupil diameter (3-8mm typical range)
        base_diameter = 5.0
        pupil_signal = np.full(self.n_samples, base_diameter)

        # Task-related dilation
        if task_related:
            # Dilation events (similar timing to P300)
            for event_time in np.arange(
                10, self.duration - 5, np.random.uniform(10, 15)
            ):
                event_idx = int(event_time * self.sampling_rate)
                if event_idx < self.n_samples - 1000:
                    # Pupil dilation response (peaks around 1-2 seconds)
                    dilation_template = self._generate_pupil_dilation()
                    dilation_start = event_idx
                    dilation_end = min(
                        dilation_start + len(dilation_template), self.n_samples
                    )
                    pupil_signal[dilation_start:dilation_end] += dilation_template[
                        : dilation_end - dilation_start
                    ]

        # Spontaneous fluctuations
        spontaneous = self._generate_pink_noise(self.n_samples) * 0.3
        pupil_signal += spontaneous

        # Blink-related drops
        for _ in range(np.random.randint(10, 20)):
            blink_idx = np.random.randint(0, self.n_samples - 100)
            blink_duration = np.random.randint(100, 300)
            blink_end = min(blink_idx + blink_duration, self.n_samples)
            pupil_signal[blink_idx:blink_end] *= 0.1  # Pupil closes during blink

        # Ensure positive values
        pupil_signal = np.maximum(pupil_signal, 1.0)

        return pupil_signal

    def generate_eda_data(self, responsive=True):
        """Generate realistic electrodermal activity data."""
        # Base EDA level (0.5-3 microsiemens typical)
        base_eda = 1.0
        eda_signal = np.full(self.n_samples, base_eda)

        # Tonic component (slow changes)
        tonic = self._generate_pink_noise(self.n_samples) * 0.2
        eda_signal += tonic

        # Phasic responses (SCR - skin conductance responses)
        if responsive:
            # SCR events (peaks around 1-4 seconds after stimulus)
            for event_time in np.arange(
                15, self.duration - 5, np.random.uniform(12, 20)
            ):
                event_idx = int(event_time * self.sampling_rate)
                if event_idx < self.n_samples - 4000:
                    scr_template = self._generate_scr_waveform()
                    scr_start = event_idx + np.random.randint(500, 1500)  # Delay
                    scr_end = min(scr_start + len(scr_template), self.n_samples)
                    eda_signal[scr_start:scr_end] += scr_template[: scr_end - scr_start]

        # Add noise
        noise = np.random.normal(0, 0.05, self.n_samples)
        eda_signal += noise

        # Ensure positive values
        eda_signal = np.maximum(eda_signal, 0.1)

        return eda_signal

    def generate_heart_rate_data(self):
        """Generate realistic heart rate data."""
        # Base heart rate (60-80 BPM typical)
        base_hr = 70.0
        hr_signal = np.full(self.n_samples, base_hr, dtype=float)

        # Respiratory sinus arrhythmia
        respiratory_freq = 0.25  # 15 breaths per minute
        respiratory_variation = 5.0 * np.sin(
            2 * np.pi * respiratory_freq * self.time_vector
        )
        hr_signal += respiratory_variation

        # Task-related increases
        for event_time in np.arange(10, self.duration - 5, np.random.uniform(10, 15)):
            event_idx = int(event_time * self.sampling_rate)
            if event_idx < self.n_samples - 5000:
                # Heart rate increase response
                hr_increase = self._generate_hr_increase()
                hr_start = event_idx
                hr_end = min(hr_start + len(hr_increase), self.n_samples)
                hr_signal[hr_start:hr_end] += hr_increase[: hr_end - hr_start]

        # Add noise
        noise = np.random.normal(0, 2, self.n_samples)
        hr_signal += noise

        # Ensure reasonable range
        hr_signal = np.clip(hr_signal, 40, 120)

        return hr_signal

    def _generate_p300_waveform(self):
        """Generate realistic P300 ERP component."""
        duration_ms = 800
        n_points = int(duration_ms * self.sampling_rate / 1000)
        time_ms = np.linspace(0, duration_ms, n_points)

        # P300 typically peaks at 300-400ms
        peak_time = 350
        sigma = 80
        amplitude = 5.0  # 5 microvolts typical

        p300 = amplitude * np.exp(-((time_ms - peak_time) ** 2) / (2 * sigma**2))

        # Add earlier P2 component
        p2_time = 200
        p2_sigma = 50
        p2_amplitude = 2.0
        p2 = p2_amplitude * np.exp(-((time_ms - p2_time) ** 2) / (2 * p2_sigma**2))

        return p2 + p300

    def _generate_blink_waveform(self):
        """Generate realistic eye blink artifact."""
        duration_ms = 200
        n_points = int(duration_ms * self.sampling_rate / 1000)
        time_ms = np.linspace(0, duration_ms, n_points)

        # Blink has characteristic shape
        peak_time = 100
        sigma = 30
        amplitude = 50.0  # Large amplitude artifact

        blink = amplitude * np.exp(-((time_ms - peak_time) ** 2) / (2 * sigma**2))

        return blink

    def _generate_pupil_dilation(self):
        """Generate realistic pupil dilation response."""
        duration_ms = 2000
        n_points = int(duration_ms * self.sampling_rate / 1000)
        time_ms = np.linspace(0, duration_ms, n_points)

        # Pupil dilation peaks around 1-2 seconds
        peak_time = 1200
        sigma = 300
        amplitude = 1.5  # 1.5mm typical dilation

        dilation = amplitude * np.exp(-((time_ms - peak_time) ** 2) / (2 * sigma**2))

        return dilation

    def _generate_scr_waveform(self):
        """Generate realistic skin conductance response."""
        duration_ms = 4000
        n_points = int(duration_ms * self.sampling_rate / 1000)
        time_ms = np.linspace(0, duration_ms, n_points)

        # SCR has characteristic shape with slow rise and slower decay
        rise_time = 1000
        decay_time = 3000
        amplitude = 0.5  # 0.5 microsiemens typical

        scr = np.zeros(n_points)
        for i, t in enumerate(time_ms):
            if t < rise_time:
                scr[i] = amplitude * (t / rise_time)
            else:
                scr[i] = amplitude * np.exp(-(t - rise_time) / decay_time)

        return scr

    def _generate_hr_increase(self):
        """Generate realistic heart rate increase response."""
        duration_ms = 5000
        n_points = int(duration_ms * self.sampling_rate / 1000)
        time_ms = np.linspace(0, duration_ms, n_points)

        # HR increase peaks around 2-3 seconds
        peak_time = 2500
        sigma = 800
        amplitude = 15.0  # 15 BPM typical increase

        hr_increase = amplitude * np.exp(-((time_ms - peak_time) ** 2) / (2 * sigma**2))

        return hr_increase

    def _generate_pink_noise(self, n_samples):
        """Generate pink noise (1/f noise)."""
        # Simple approximation of pink noise
        white_noise = np.random.normal(0, 1, n_samples)

        # Apply 1/f filter in frequency domain
        fft = np.fft.fft(white_noise)
        freqs = np.fft.fftfreq(n_samples)

        # Avoid division by zero
        freqs[0] = 1e-10

        # Apply 1/f scaling
        fft_scaled = fft / np.sqrt(np.abs(freqs))

        # Transform back to time domain
        pink_noise = np.real(np.fft.ifft(fft_scaled))

        # Normalize
        pink_noise = pink_noise / np.std(pink_noise)

        return pink_noise

    def create_multimodal_dataset(self, subject_id="sub_01", session_id="sess_01"):
        """Create a complete multimodal dataset."""
        print(f"Generating multimodal dataset for {subject_id}, {session_id}...")

        # Generate all signals
        eeg_signal, p300_events = self.generate_eeg_data()
        pupil_signal = self.generate_pupil_data()
        eda_signal = self.generate_eda_data()
        hr_signal = self.generate_heart_rate_data()

        # Create timestamps
        start_time = datetime.now()
        timestamps = [start_time + timedelta(seconds=t) for t in self.time_vector]

        # Create DataFrame
        data = {
            "timestamp": timestamps,
            "time_seconds": self.time_vector,
            "subject_id": [subject_id] * self.n_samples,
            "session_id": [session_id] * self.n_samples,
            "eeg_fz": eeg_signal,
            "eeg_pz": eeg_signal * 0.8
            + np.random.normal(0, 0.5, self.n_samples),  # Slightly different signal
            "pupil_diameter": pupil_signal,
            "eda": eda_signal,
            "heart_rate": hr_signal,
            "event_marker": np.zeros(self.n_samples, dtype=int),
        }

        # Mark P300 events
        for event_time in p300_events:
            event_idx = int(event_time * self.sampling_rate)
            if event_idx < self.n_samples:
                end_idx = min(event_idx + 10, self.n_samples)
                data["event_marker"][event_idx:end_idx] = 1  # 10ms event marker

        df = pd.DataFrame(data)

        # Add metadata
        metadata = {
            "subject_id": subject_id,
            "session_id": session_id,
            "sampling_rate": self.sampling_rate,
            "duration": self.duration,
            "n_samples": self.n_samples,
            "p300_events": p300_events,
            "creation_time": datetime.now().isoformat(),
            "data_quality": "good" if not np.any(np.isnan(eeg_signal)) else "poor",
        }

        return df, metadata

    def save_dataset(self, df, metadata, output_dir="data"):
        """Save dataset in multiple formats."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        base_filename = f"{metadata['subject_id']}_{metadata['session_id']}"

        # Save as CSV
        csv_file = output_path / f"{base_filename}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved CSV: {csv_file}")

        # Save as JSON
        json_file = output_path / f"{base_filename}.json"
        json_data = {"metadata": metadata, "data": df.to_dict("records")}
        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=2, default=str)
        print(f"Saved JSON: {json_file}")

        # Save metadata separately
        meta_file = output_path / f"{base_filename}_metadata.json"
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Saved metadata: {meta_file}")

        return csv_file, json_file, meta_file


def main():
    """Generate sample datasets for testing."""
    print("APGI Framework - Sample Data Generator")
    print("=" * 50)

    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Generate multiple subjects
    generator = SampleDataGenerator(sampling_rate=1000, duration=60)

    subjects = ["sub_01", "sub_02", "sub_03"]
    sessions = ["sess_01", "sess_02"]

    for subject in subjects:
        for session in sessions:
            df, metadata = generator.create_multimodal_dataset(subject, session)
            generator.save_dataset(df, metadata, "data")

    # Generate a shorter demo dataset
    demo_generator = SampleDataGenerator(
        sampling_rate=250, duration=30
    )  # Lower sampling rate for demo
    demo_df, demo_metadata = demo_generator.create_multimodal_dataset("demo", "demo")
    demo_generator.save_dataset(demo_df, demo_metadata, "data")

    print("\nSample datasets generated successfully!")
    print("Available files in data/ directory:")

    data_path = Path("data")
    for file_path in sorted(data_path.glob("*")):
        if file_path.is_file():
            size_kb = file_path.stat().st_size / 1024
            print(f"  {file_path.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
