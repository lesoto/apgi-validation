"""
APGI Multimodal Integration Module
==================================

Integrates multiple physiological modalities (EEG, pupil, EDA) for APGI analysis.
Provides core integration logic and batch processing capabilities.
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class ModalityData:
    """Container for single modality data."""

    name: str
    data: np.ndarray
    sampling_rate: float
    units: str


@dataclass
class IntegrationResult:
    """Result of multimodal integration."""

    integrated_signal: np.ndarray
    confidence_score: float
    modality_contributions: Dict[str, float]
    processing_time: float
    warnings: List[str]


class APGICoreIntegration:
    """Core multimodal integration engine for APGI analysis."""

    def __init__(self):
        """Initialize the integration engine."""
        self.modalities = {
            "exteroceptive": ["P3b_amplitude", "pupil_diameter"],
            "interoceptive": ["SCR", "heart_rate"],
            "somatic": ["beta_power"],
        }

    def integrate_modalities(
        self, modality_data: Dict[str, ModalityData]
    ) -> IntegrationResult:
        """
        Integrate multiple physiological modalities.

        Args:
            modality_data: Dictionary of modality data

        Returns:
            IntegrationResult with integrated signal and metadata
        """
        import time

        start_time = time.time()

        # Validate input data
        self._validate_modality_data(modality_data)

        # Normalize modalities
        normalized_data = self._normalize_modalities(modality_data)

        # Compute integration weights based on modality reliability
        weights = self._compute_integration_weights(normalized_data)

        # Perform weighted integration
        integrated_signal = self._weighted_integration(normalized_data, weights)

        # Compute confidence score
        confidence_score = self._compute_confidence_score(normalized_data, weights)

        # Calculate modality contributions
        contributions = self._calculate_contributions(weights)

        processing_time = time.time() - start_time

        return IntegrationResult(
            integrated_signal=integrated_signal,
            confidence_score=confidence_score,
            modality_contributions=contributions,
            processing_time=processing_time,
            warnings=[],
        )

    def _validate_modality_data(self, modality_data: Dict[str, ModalityData]) -> None:
        """Validate that modality data meets requirements."""
        if not modality_data:
            raise ValueError("No modality data provided")

        # Check that we have at least one modality from each category
        categories_present = set()
        for modality in modality_data.values():
            for category, mods in self.modalities.items():
                if modality.name in mods:
                    categories_present.add(category)

        if len(categories_present) < 2:
            raise ValueError("Need at least two modality categories for integration")

        # Check data consistency
        lengths = [len(mod.data) for mod in modality_data.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All modalities must have the same number of samples")

    def _normalize_modalities(
        self, modality_data: Dict[str, ModalityData]
    ) -> Dict[str, np.ndarray]:
        """Normalize modality data to common scale."""
        normalized = {}

        for name, mod_data in modality_data.items():
            # Z-score normalization
            data = mod_data.data
            if np.std(data) > 0:
                normalized[name] = (data - np.mean(data)) / np.std(data)
            else:
                normalized[name] = data - np.mean(data)  # Handle constant signals

        return normalized

    def _compute_integration_weights(
        self, normalized_data: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Compute integration weights based on signal quality and reliability."""
        weights = {}

        for name, data in normalized_data.items():
            # Base weight on signal-to-noise ratio and temporal stability
            if len(data) > 1:
                # Compute temporal stability (inverse of derivative variance)
                derivatives = np.diff(data)
                stability = 1.0 / (1.0 + np.var(derivatives))

                # Compute signal-to-noise ratio approximation
                signal_power = np.var(data)
                noise_estimate = np.mean(np.abs(derivatives)) * 0.1
                snr = signal_power / (noise_estimate + 1e-10)

                # Combine metrics
                weight = min(1.0, (stability * snr) / 10.0)
            else:
                weight = 0.5  # Default weight for single-sample data

            weights[name] = max(0.1, min(1.0, weight))  # Clamp to [0.1, 1.0]

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def _weighted_integration(
        self, normalized_data: Dict[str, np.ndarray], weights: Dict[str, float]
    ) -> np.ndarray:
        """Perform weighted integration of normalized modalities."""
        integrated = np.zeros(len(next(iter(normalized_data.values()))))

        for name, data in normalized_data.items():
            integrated += weights[name] * data

        return integrated

    def _compute_confidence_score(
        self, normalized_data: Dict[str, np.ndarray], weights: Dict[str, float]
    ) -> float:
        """Compute overall confidence score for the integration."""
        # Confidence based on weight distribution and signal consistency
        weight_entropy = -sum(w * np.log(w + 1e-10) for w in weights.values())
        max_entropy = np.log(len(weights))

        # Lower entropy (more balanced weights) = higher confidence
        balance_score = 1.0 - (weight_entropy / max_entropy)

        # Consistency score based on correlation between modalities
        correlations = []
        modality_names = list(normalized_data.keys())
        for i in range(len(modality_names)):
            for j in range(i + 1, len(modality_names)):
                corr = np.corrcoef(
                    normalized_data[modality_names[i]],
                    normalized_data[modality_names[j]],
                )[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

        consistency_score = np.mean(correlations) if correlations else 0.5

        # Combine scores
        confidence = 0.5 * balance_score + 0.5 * consistency_score
        return max(0.0, min(1.0, confidence))

    def _calculate_contributions(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate percentage contribution of each modality."""
        total_weight = sum(weights.values())
        if total_weight == 0:
            return {name: 1.0 / len(weights) for name in weights.keys()}

        return {name: weight / total_weight for name, weight in weights.items()}


class APGIBatchProcessor:
    """Batch processor for multimodal data integration."""

    def __init__(
        self,
        integration_engine: APGICoreIntegration,
        normalization_config: Dict[str, Dict[str, float]],
    ):
        """
        Initialize batch processor.

        Args:
            integration_engine: Core integration engine
            normalization_config: Configuration for data normalization
        """
        self.integration_engine = integration_engine
        self.normalization_config = normalization_config

    def process_subject(self, subject_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Process multimodal data for a single subject.

        Args:
            subject_data: Dictionary with modality data arrays

        Returns:
            Processing results
        """
        try:
            # Convert to ModalityData objects
            modality_data = {}
            for name, data in subject_data.items():
                if isinstance(data, np.ndarray):
                    modality_data[name] = ModalityData(
                        name=name,
                        data=data,
                        sampling_rate=100.0,  # Assume 100 Hz
                        units=self._infer_units(name),
                    )

            # Perform integration
            result = self.integration_engine.integrate_modalities(modality_data)

            # Format result for output
            return {
                "status": "success",
                "integrated_signal": result.integrated_signal.tolist(),
                "confidence_score": result.confidence_score,
                "modality_contributions": result.modality_contributions,
                "processing_time": result.processing_time,
                "metadata": {
                    "n_samples": len(result.integrated_signal),
                    "modalities_used": list(modality_data.keys()),
                    "integration_method": "weighted_average",
                },
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Integration failed: {str(e)}",
                "error_type": type(e).__name__,
            }

    def _infer_units(self, modality_name: str) -> str:
        """Infer units for a modality based on its name."""
        unit_map = {
            "P3b_amplitude": "µV",
            "pupil_diameter": "mm",
            "SCR": "µS",
            "heart_rate": "bpm",
            "beta_power": "dB",
        }
        return unit_map.get(modality_name, "AU")  # AU = Arbitrary Units

    def process_batch(
        self, subjects_data: List[Dict[str, np.ndarray]]
    ) -> List[Dict[str, Any]]:
        """
        Process multimodal data for multiple subjects.

        Args:
            subjects_data: List of subject data dictionaries

        Returns:
            List of processing results
        """
        results = []
        for i, subject_data in enumerate(subjects_data):
            try:
                result = self.process_subject(subject_data)
                result["subject_index"] = i
                results.append(result)
            except Exception as e:
                results.append(
                    {
                        "subject_index": i,
                        "status": "error",
                        "message": f"Batch processing failed for subject {i}: {str(e)}",
                    }
                )

        return results
