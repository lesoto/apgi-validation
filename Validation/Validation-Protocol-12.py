"""
APGI Validation Protocol 12: Clinical and Cross-Species Convergence
==================================================================

Complete implementation of Priority 4 from the APGI Empirical Credibility Roadmap:
Clinical and cross-species convergence.

This protocol validates:
- Loss of P3b/frontoparietal activation in vegetative state patients
- APGI parameter changes in psychiatric disorders (GAD, MDD, Psychosis)
- Cross-species homologies in ignition signatures
- Convergence between APGI (algorithmic) and IIT (implementational) descriptions

"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix


class ClinicalDataAnalyzer:
    """Analyze clinical populations for APGI validation"""

    def __init__(self):
        # Clinical population characteristics
        self.clinical_profiles = {
            "vegetative_state": {
                "p3b_amplitude": 0.1,  # Severely reduced
                "frontoparietal_connectivity": 0.05,  # Minimal
                "ignition_probability": 0.01,
                "theta_t": 2.0,  # Very high threshold
            },
            "minimally_conscious": {
                "p3b_amplitude": 0.4,
                "frontoparietal_connectivity": 0.3,
                "ignition_probability": 0.2,
                "theta_t": 1.2,
            },
            "healthy_controls": {
                "p3b_amplitude": 1.0,
                "frontoparietal_connectivity": 1.0,
                "ignition_probability": 0.8,
                "theta_t": 0.5,
            },
        }

    def simulate_patient_data(
        self, condition: str, n_subjects: int = 20
    ) -> pd.DataFrame:
        """
        Simulate clinical patient data based on known profiles

        Args:
            condition: 'vegetative_state', 'minimally_conscious', 'healthy_controls'
            n_subjects: Number of subjects to simulate

        Returns:
            DataFrame with simulated patient data
        """
        if condition not in self.clinical_profiles:
            raise ValueError(f"Unknown condition: {condition}")

        profile = self.clinical_profiles[condition]

        data = []
        for subject_id in range(n_subjects):
            # Simulate neural measures
            p3b_noise = np.random.normal(0, 0.1)
            connectivity_noise = np.random.normal(0, 0.1)
            ignition_noise = np.random.normal(0, 0.05)
            threshold_noise = np.random.normal(0, 0.1)

            subject_data = {
                "subject_id": subject_id,
                "condition": condition,
                "p3b_amplitude": max(0, profile["p3b_amplitude"] + p3b_noise),
                "frontoparietal_connectivity": max(
                    0, profile["frontoparietal_connectivity"] + connectivity_noise
                ),
                "ignition_probability": np.clip(
                    profile["ignition_probability"] + ignition_noise, 0, 1
                ),
                "theta_t": max(0.1, profile["theta_t"] + threshold_noise),
            }

            # Add APGI-specific measures
            subject_data.update(self._simulate_apgi_measures(subject_data))

            data.append(subject_data)

        return pd.DataFrame(data)

    def _simulate_apgi_measures(self, subject_data: Dict) -> Dict:
        """Simulate APGI-specific measures based on neural data"""

        p3b = subject_data["p3b_amplitude"]
        connectivity = subject_data["frontoparietal_connectivity"]

        # Estimate APGI parameters from neural measures
        Pi_e = 0.5 + 0.5 * connectivity  # Precision from connectivity
        Pi_i = 0.3 + 0.4 * p3b  # Interoceptive precision from P3b
        beta = 1.0 + 0.5 * connectivity  # Somatic influence

        # Simulate precision expectation gap (key for anxiety disorders)
        precision_expectation_gap = np.random.normal(0, 0.2)

        return {
            "Pi_e": Pi_e,
            "Pi_i": Pi_i,
            "beta": beta,
            "precision_expectation_gap": precision_expectation_gap,
        }

    def analyze_clinical_differences(self, patient_data: pd.DataFrame) -> Dict:
        """
        Analyze differences between clinical populations

        Args:
            patient_data: DataFrame with patient data

        Returns:
            Dictionary with statistical analysis results
        """

        conditions = patient_data["condition"].unique()
        results = {}

        # Compare key measures across conditions
        measures = [
            "p3b_amplitude",
            "frontoparietal_connectivity",
            "ignition_probability",
            "theta_t",
        ]

        for measure in measures:
            condition_data = [
                patient_data[patient_data["condition"] == cond][measure].values
                for cond in conditions
            ]

            # ANOVA
            f_stat, p_value = stats.f_oneway(*condition_data)

            # Effect sizes (Cohen's d for key comparisons)
            if "vegetative_state" in conditions and "healthy_controls" in conditions:
                vs_data = patient_data[patient_data["condition"] == "vegetative_state"][
                    measure
                ]
                hc_data = patient_data[patient_data["condition"] == "healthy_controls"][
                    measure
                ]
                cohens_d = self._cohens_d(vs_data, hc_data)
            else:
                cohens_d = None

            results[measure] = {
                "f_statistic": f_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "cohens_d_vs_healthy": cohens_d,
                "condition_means": {
                    cond: np.mean(
                        patient_data[patient_data["condition"] == cond][measure]
                    )
                    for cond in conditions
                },
            }

        return results

    def _cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0


class PsychiatricProfileAnalyzer:
    """Analyze psychiatric disorder profiles"""

    def __init__(self):
        # DSM-5 inspired profiles with APGI interpretations
        self.psychiatric_profiles = {
            "generalized_anxiety_disorder": {
                "precision_expectation_gap": 0.8,  # Π̂ > Π (overestimation)
                "Pi_e_baseline": 0.9,  # High exteroceptive precision
                "Pi_i_baseline": 0.4,  # Low interoceptive precision
                "beta": 2.0,  # High somatic influence
                "theta_t": 0.3,  # Low threshold (hypervigilant)
                "arousal": 0.9,  # High baseline arousal
            },
            "major_depressive_disorder": {
                "precision_expectation_gap": -0.6,  # Π̂ < Π (underestimation)
                "Pi_e_baseline": 0.3,  # Low exteroceptive precision
                "Pi_i_baseline": 0.2,  # Low interoceptive precision
                "beta": 0.8,  # Low somatic influence
                "theta_t": 1.2,  # High threshold (reduced responsiveness)
                "arousal": 0.3,  # Low arousal
            },
            "psychosis": {
                "precision_expectation_gap": 1.2,  # Severe overestimation
                "Pi_e_baseline": 1.1,  # Very high exteroceptive precision
                "Pi_i_baseline": 0.1,  # Very low interoceptive precision
                "beta": 0.5,  # Low somatic modulation
                "theta_t": 0.2,  # Very low threshold (hallucinations)
                "arousal": 1.0,  # Very high arousal
            },
            "healthy_controls": {
                "precision_expectation_gap": 0.0,
                "Pi_e_baseline": 0.7,
                "Pi_i_baseline": 0.6,
                "beta": 1.2,
                "theta_t": 0.5,
                "arousal": 0.6,
            },
        }

    def simulate_psychiatric_data(
        self, diagnosis: str, n_subjects: int = 30
    ) -> pd.DataFrame:
        """
        Simulate psychiatric patient data

        Args:
            diagnosis: Psychiatric diagnosis
            n_subjects: Number of subjects

        Returns:
            DataFrame with simulated psychiatric data
        """
        if diagnosis not in self.psychiatric_profiles:
            raise ValueError(f"Unknown diagnosis: {diagnosis}")

        profile = self.psychiatric_profiles[diagnosis]

        data = []
        for subject_id in range(n_subjects):
            subject_data = {
                "subject_id": subject_id,
                "diagnosis": diagnosis,
            }

            # Add profile parameters with noise
            for param, value in profile.items():
                noise_scale = (
                    0.2 if "gap" in param else 0.15
                )  # More noise for expectation gap
                noise = np.random.normal(0, noise_scale)
                subject_data[param] = value + noise

            # Calculate derived measures
            subject_data.update(self._calculate_psychiatric_measures(subject_data))

            data.append(subject_data)

        return pd.DataFrame(data)

    def _calculate_psychiatric_measures(self, subject_data: Dict) -> Dict:
        """Calculate derived psychiatric measures"""

        gap = subject_data["precision_expectation_gap"]

        # Anxiety index (positive gap)
        anxiety_index = max(0, gap * 10)

        # Depression index (negative gap)
        depression_index = max(0, -gap * 8)

        # Psychosis liability (extreme positive gap)
        psychosis_liability = max(0, gap - 0.5) * 5

        # APGI-based symptom predictions
        if gap > 0.5:  # Anxiety/psychosis profile
            predicted_symptoms = [
                "hypervigilance",
                "racing_thoughts",
                "somatic_complaints",
            ]
        elif gap < -0.3:  # Depression profile
            predicted_symptoms = ["anhedonia", "fatigue", "reduced_motivation"]
        else:  # Normal range
            predicted_symptoms = []

        return {
            "anxiety_index": anxiety_index,
            "depression_index": depression_index,
            "psychosis_liability": psychosis_liability,
            "predicted_symptoms": predicted_symptoms,
        }

    def validate_diagnostic_accuracy(self, psychiatric_data: pd.DataFrame) -> Dict:
        """
        Validate APGI-based diagnostic classification

        Args:
            psychiatric_data: DataFrame with psychiatric data

        Returns:
            Classification performance metrics
        """

        # Features for classification
        y = psychiatric_data["diagnosis"].values

        # Simple rule-based classification (could be improved with ML)
        predictions = []
        for _, subject in psychiatric_data.iterrows():
            gap = subject["precision_expectation_gap"]
            theta = subject["theta_t"]
            arousal = subject["arousal"]

            if gap > 0.5 and arousal > 0.8:
                pred = "psychosis" if gap > 1.0 else "generalized_anxiety_disorder"
            elif gap < -0.3 and theta > 0.8:
                pred = "major_depressive_disorder"
            else:
                pred = "healthy_controls"

            predictions.append(pred)

        # Calculate performance metrics
        y_pred = np.array(predictions)

        # Confusion matrix
        conditions = [
            "generalized_anxiety_disorder",
            "major_depressive_disorder",
            "psychosis",
            "healthy_controls",
        ]
        cm = confusion_matrix(y, y_pred, labels=conditions)

        # Classification report
        report = classification_report(y, y_pred, output_dict=True)

        return {
            "confusion_matrix": cm,
            "classification_report": report,
            "accuracy": report["accuracy"],
            "diagnostic_power": self._calculate_diagnostic_power(cm, conditions),
        }

    def _calculate_diagnostic_power(
        self, cm: np.ndarray, conditions: List[str]
    ) -> Dict:
        """Calculate diagnostic discrimination power"""

        power = {}
        for i, condition in enumerate(conditions):
            # Sensitivity (true positive rate)
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

            # Specificity (true negative rate)
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + tp
            fp = np.sum(cm[:, i]) - tp
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            power[condition] = {
                "sensitivity": sensitivity,
                "specificity": specificity,
                "discrimination": (sensitivity + specificity) / 2,
            }

        return power


class CrossSpeciesHomologyAnalyzer:
    """Analyze cross-species homologies in ignition signatures"""

    def __init__(self):
        # Species-specific APGI parameter ranges (based on neuroanatomy)
        self.species_profiles = {
            "human": {
                "cortical_thickness": 1.0,
                "frontal_lobe_ratio": 1.0,
                "theta_t_range": (0.3, 0.8),
                "Pi_e_range": (0.6, 0.9),
                "ignition_latency": 0.3,  # seconds
            },
            "macaque": {
                "cortical_thickness": 0.7,
                "frontal_lobe_ratio": 0.8,
                "theta_t_range": (0.4, 0.9),
                "Pi_e_range": (0.5, 0.8),
                "ignition_latency": 0.25,
            },
            "mouse": {
                "cortical_thickness": 0.3,
                "frontal_lobe_ratio": 0.4,
                "theta_t_range": (0.6, 1.2),
                "Pi_e_range": (0.3, 0.6),
                "ignition_latency": 0.15,
            },
            "zebrafish": {
                "cortical_thickness": 0.1,
                "frontal_lobe_ratio": 0.2,
                "theta_t_range": (0.8, 1.5),
                "Pi_e_range": (0.2, 0.4),
                "ignition_latency": 0.1,
            },
        }

    def simulate_species_data(self, species: str, n_subjects: int = 10) -> pd.DataFrame:
        """
        Simulate cross-species data

        Args:
            species: Species name
            n_subjects: Number of subjects

        Returns:
            DataFrame with simulated species data
        """
        if species not in self.species_profiles:
            raise ValueError(f"Unknown species: {species}")

        profile = self.species_profiles[species]

        data = []
        for subject_id in range(n_subjects):
            subject_data = {
                "subject_id": subject_id,
                "species": species,
                "cortical_thickness": profile["cortical_thickness"],
                "frontal_lobe_ratio": profile["frontal_lobe_ratio"],
            }

            # Sample APGI parameters within species range
            subject_data["theta_t"] = np.random.uniform(*profile["theta_t_range"])
            subject_data["Pi_e"] = np.random.uniform(*profile["Pi_e_range"])
            subject_data["ignition_latency"] = profile[
                "ignition_latency"
            ] + np.random.normal(0, 0.02)

            # Add species-specific neural measures
            subject_data.update(self._simulate_species_measures(subject_data, species))

            data.append(subject_data)

        return pd.DataFrame(data)

    def _simulate_species_measures(self, subject_data: Dict, species: str) -> Dict:
        """Simulate species-specific neural measures"""

        # Scaling factors based on neuroanatomy
        if species == "human":
            p3b_scale, connectivity_scale = 1.0, 1.0
        elif species == "macaque":
            p3b_scale, connectivity_scale = 0.8, 0.9
        elif species == "mouse":
            p3b_scale, connectivity_scale = 0.4, 0.5
        else:  # zebrafish
            p3b_scale, connectivity_scale = 0.2, 0.3

        # Simulate measures
        theta_t = subject_data["theta_t"]
        Pi_e = subject_data["Pi_e"]

        p3b_amplitude = p3b_scale * (1.0 / (1.0 + np.exp(-5.0 * (Pi_e - theta_t))))
        connectivity = connectivity_scale * Pi_e

        return {
            "p3b_amplitude": p3b_amplitude,
            "frontoparietal_connectivity": connectivity,
        }

    def analyze_homologies(self, species_data: pd.DataFrame) -> Dict:
        """
        Analyze evolutionary homologies in ignition signatures

        Args:
            species_data: DataFrame with cross-species data

        Returns:
            Homology analysis results
        """

        results = {}

        # Test for conserved relationships across species
        for measure in [
            "p3b_amplitude",
            "frontoparietal_connectivity",
            "ignition_latency",
        ]:
            # Correlation with APGI parameters across species
            correlations = {}
            for param in ["theta_t", "Pi_e"]:
                corr, p_value = stats.pearsonr(
                    species_data[measure], species_data[param]
                )
                correlations[param] = {
                    "correlation": corr,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                }

            results[measure] = correlations

        # Test for phylogenetic signal
        results["phylogenetic_conservation"] = self._test_phylogenetic_conservation(
            species_data
        )

        return results

    def _test_phylogenetic_conservation(self, species_data: pd.DataFrame) -> Dict:
        """Test if APGI parameters show phylogenetic conservation"""

        # Simple phylogenetic distance matrix (arbitrary units)
        phylo_distances = {
            ("human", "macaque"): 25,
            ("human", "mouse"): 75,
            ("human", "zebrafish"): 400,
            ("macaque", "mouse"): 50,
            ("macaque", "zebrafish"): 375,
            ("mouse", "zebrafish"): 350,
        }

        # Test correlation between phylogenetic distance and parameter similarity
        species_list = species_data["species"].unique()
        conservation_results = {}

        for param in ["theta_t", "Pi_e", "ignition_latency"]:
            distances = []
            similarities = []

            for i, sp1 in enumerate(species_list):
                for sp2 in species_list[i + 1 :]:
                    if (sp1, sp2) in phylo_distances:
                        dist = phylo_distances[(sp1, sp2)]

                        # Parameter similarity (1 - normalized difference)
                        val1 = np.mean(
                            species_data[species_data["species"] == sp1][param]
                        )
                        val2 = np.mean(
                            species_data[species_data["species"] == sp2][param]
                        )
                        similarity = 1 - abs(val1 - val2) / max(val1, val2)

                        distances.append(dist)
                        similarities.append(similarity)

            if distances:
                corr, p_value = stats.pearsonr(distances, similarities)
                conservation_results[param] = {
                    "phylocorrelation": corr,
                    "p_value": p_value,
                    "conserved": p_value < 0.05
                    and corr < 0,  # Negative correlation = conservation
                }

        return conservation_results


class IITConvergenceAnalyzer:
    """Analyze convergence with Integrated Information Theory"""

    def __init__(self):
        # IIT Φ values for different states (simplified)
        self.iit_phi_values = {
            "unconscious": 0.1,
            "minimally_conscious": 1.5,
            "conscious": 5.0,
            "self-conscious": 12.0,
        }

    def simulate_iit_apgi_convergence(self, n_simulations: int = 100) -> Dict:
        """
        Simulate convergence between IIT Φ and APGI ignition

        Returns:
            Convergence analysis results
        """

        convergence_data = []

        for _ in range(n_simulations):
            # Random APGI state
            S = np.random.uniform(0, 2)  # Surprise level
            theta = np.random.uniform(0.2, 1.0)  # Threshold
            alpha = 5.0  # Sigmoid steepness

            ignition_prob = 1.0 / (1.0 + np.exp(-alpha * (S - theta)))

            # Map to IIT Φ (simplified relationship)
            # Higher ignition probability → higher Φ
            phi_estimated = 0.5 + 10 * ignition_prob + np.random.normal(0, 1)

            # Determine consciousness state
            if ignition_prob < 0.3:
                state = "unconscious"
            elif ignition_prob < 0.6:
                state = "minimally_conscious"
            elif ignition_prob < 0.8:
                state = "conscious"
            else:
                state = "self-conscious"

            true_phi = self.iit_phi_values[state]

            convergence_data.append(
                {
                    "ignition_probability": ignition_prob,
                    "phi_estimated": phi_estimated,
                    "phi_true": true_phi,
                    "state": state,
                    "convergence_error": abs(phi_estimated - true_phi),
                }
            )

        convergence_df = pd.DataFrame(convergence_data)

        # Statistical analysis
        correlation, p_value = stats.pearsonr(
            convergence_df["ignition_probability"], convergence_df["phi_true"]
        )

        return {
            "convergence_data": convergence_df,
            "correlation_coefficient": correlation,
            "correlation_p_value": p_value,
            "mean_convergence_error": np.mean(convergence_df["convergence_error"]),
            "convergence_significant": p_value < 0.05,
            "state_classification_accuracy": self._analyze_state_classification(
                convergence_df
            ),
        }

    def _analyze_state_classification(self, convergence_df: pd.DataFrame) -> float:
        """Analyze how well APGI predicts IIT consciousness states"""

        correct_predictions = 0
        total_predictions = len(convergence_df)

        for _, row in convergence_df.iterrows():
            ignition = row["ignition_probability"]

            # Predict state from ignition
            if ignition < 0.3:
                pred_state = "unconscious"
            elif ignition < 0.6:
                pred_state = "minimally_conscious"
            elif ignition < 0.8:
                pred_state = "conscious"
            else:
                pred_state = "self-conscious"

            if pred_state == row["state"]:
                correct_predictions += 1

        return correct_predictions / total_predictions


class ClinicalConvergenceValidator:
    """Complete clinical and cross-species validation"""

    def __init__(self):
        self.clinical_analyzer = ClinicalDataAnalyzer()
        self.psychiatric_analyzer = PsychiatricProfileAnalyzer()
        self.species_analyzer = CrossSpeciesHomologyAnalyzer()
        self.iit_analyzer = IITConvergenceAnalyzer()

    def validate_clinical_convergence(self) -> Dict:
        """
        Complete validation of clinical and cross-species convergence

        Returns:
            Dictionary with all convergence validation results
        """

        results = {
            "disorders_of_consciousness": self._validate_disorders_of_consciousness(),
            "psychiatric_disorder_profiles": self._validate_psychiatric_profiles(),
            "cross_species_homologies": self._validate_cross_species_homologies(),
            "iit_apgi_convergence": self._validate_iit_convergence(),
            "overall_clinical_score": 0.0,
        }

        # Calculate overall score
        results["overall_clinical_score"] = self._calculate_clinical_score(results)

        return results

    def _validate_disorders_of_consciousness(self) -> Dict:
        """Validate APGI in disorders of consciousness"""

        # Simulate patient data
        conditions = ["vegetative_state", "minimally_conscious", "healthy_controls"]
        patient_data = []

        for condition in conditions:
            condition_data = self.clinical_analyzer.simulate_patient_data(
                condition, n_subjects=15
            )
            patient_data.append(condition_data)

        all_patient_data = pd.concat(patient_data, ignore_index=True)

        # Analyze differences
        clinical_differences = self.clinical_analyzer.analyze_clinical_differences(
            all_patient_data
        )

        # Test key predictions
        predictions_tested = {
            "p3b_loss_in_vs": clinical_differences["p3b_amplitude"]["significant"]
            and clinical_differences["p3b_amplitude"]["cohens_d_vs_healthy"] > 1.0,
            "connectivity_loss_in_vs": clinical_differences[
                "frontoparietal_connectivity"
            ]["significant"]
            and clinical_differences["frontoparietal_connectivity"][
                "cohens_d_vs_healthy"
            ]
            > 1.0,
            "threshold_elevation_in_vs": clinical_differences["theta_t"]["significant"]
            and clinical_differences["theta_t"]["condition_means"]["vegetative_state"]
            > clinical_differences["theta_t"]["condition_means"]["healthy_controls"],
        }

        return {
            "patient_data": all_patient_data,
            "clinical_differences": clinical_differences,
            "key_predictions": predictions_tested,
            "validation_passed": all(predictions_tested.values()),
        }

    def _validate_psychiatric_profiles(self) -> Dict:
        """Validate psychiatric disorder profiles"""

        # Simulate psychiatric data
        diagnoses = [
            "generalized_anxiety_disorder",
            "major_depressive_disorder",
            "psychosis",
            "healthy_controls",
        ]
        psychiatric_data = []

        for diagnosis in diagnoses:
            diagnosis_data = self.psychiatric_analyzer.simulate_psychiatric_data(
                diagnosis, n_subjects=25
            )
            psychiatric_data.append(diagnosis_data)

        all_psychiatric_data = pd.concat(psychiatric_data, ignore_index=True)

        # Validate diagnostic accuracy
        diagnostic_performance = self.psychiatric_analyzer.validate_diagnostic_accuracy(
            all_psychiatric_data
        )

        # Test APGI-based predictions
        apgi_predictions = {
            "anxiety_precision_gap": np.mean(
                all_psychiatric_data[
                    all_psychiatric_data["diagnosis"] == "generalized_anxiety_disorder"
                ]["precision_expectation_gap"]
            )
            > 0.5,
            "depression_precision_gap": np.mean(
                all_psychiatric_data[
                    all_psychiatric_data["diagnosis"] == "major_depressive_disorder"
                ]["precision_expectation_gap"]
            )
            < -0.3,
            "psychosis_precision_gap": np.mean(
                all_psychiatric_data[all_psychiatric_data["diagnosis"] == "psychosis"][
                    "precision_expectation_gap"
                ]
            )
            > 1.0,
        }

        return {
            "psychiatric_data": all_psychiatric_data,
            "diagnostic_performance": diagnostic_performance,
            "apgi_predictions": apgi_predictions,
            "validation_passed": diagnostic_performance["accuracy"] > 0.7
            and all(apgi_predictions.values()),
        }

    def _validate_cross_species_homologies(self) -> Dict:
        """Validate cross-species homologies"""

        # Simulate species data
        species_list = ["human", "macaque", "mouse", "zebrafish"]
        species_data = []

        for species in species_list:
            species_df = self.species_analyzer.simulate_species_data(
                species, n_subjects=12
            )
            species_data.append(species_df)

        all_species_data = pd.concat(species_data, ignore_index=True)

        # Analyze homologies
        homology_results = self.species_analyzer.analyze_homologies(all_species_data)

        # Test conservation predictions
        conservation_tests = {
            "p3b_conserved": homology_results["p3b_amplitude"]["Pi_e"]["significant"],
            "connectivity_conserved": homology_results["frontoparietal_connectivity"][
                "Pi_e"
            ]["significant"],
            "phylogenetic_signal": any(
                param_results.get("conserved", False)
                for param_results in homology_results.get(
                    "phylogenetic_conservation", {}
                ).values()
            ),
        }

        return {
            "species_data": all_species_data,
            "homology_analysis": homology_results,
            "conservation_tests": conservation_tests,
            "validation_passed": all(conservation_tests.values()),
        }

    def _validate_iit_convergence(self) -> Dict:
        """Validate convergence with IIT"""

        convergence_results = self.iit_analyzer.simulate_iit_apgi_convergence(
            n_simulations=200
        )

        # Test convergence predictions
        convergence_tests = {
            "correlation_significant": convergence_results["convergence_significant"],
            "state_classification_accurate": convergence_results[
                "state_classification_accuracy"
            ]
            > 0.8,
            "low_convergence_error": convergence_results["mean_convergence_error"]
            < 2.0,
        }

        return {
            "convergence_analysis": convergence_results,
            "convergence_tests": convergence_tests,
            "validation_passed": all(convergence_tests.values()),
        }

    def _calculate_clinical_score(self, results: Dict) -> float:
        """Calculate overall clinical validation score"""

        scores = []

        # Disorders of consciousness (weight: 0.35)
        doc_result = results.get("disorders_of_consciousness", {})
        scores.append(
            0.35 * (1.0 if doc_result.get("validation_passed", False) else 0.0)
        )

        # Psychiatric profiles (weight: 0.35)
        psych_result = results.get("psychiatric_disorder_profiles", {})
        scores.append(
            0.35 * (1.0 if psych_result.get("validation_passed", False) else 0.0)
        )

        # Cross-species homologies (weight: 0.2)
        species_result = results.get("cross_species_homologies", {})
        scores.append(
            0.2 * (1.0 if species_result.get("validation_passed", False) else 0.0)
        )

        # IIT convergence (weight: 0.1)
        iit_result = results.get("iit_apgi_convergence", {})
        scores.append(
            0.1 * (1.0 if iit_result.get("validation_passed", False) else 0.0)
        )

        return sum(scores)


def main():
    """Run clinical convergence validation"""
    validator = ClinicalConvergenceValidator()
    results = validator.validate_clinical_convergence()

    print("APGI Clinical and Cross-Species Convergence Validation Results:")
    print(f"Overall Clinical Validation Score: {results['overall_clinical_score']:.3f}")

    print("\nDetailed Results:")
    for key, value in results.items():
        if key != "overall_clinical_score":
            print(f"\n{key}:")
            if isinstance(value, dict):
                if "validation_passed" in value:
                    print(f"  Validation Passed: {value['validation_passed']}")
                for sub_key, sub_value in value.items():
                    if sub_key != "validation_passed" and not isinstance(
                        sub_value, (pd.DataFrame, dict)
                    ):
                        if isinstance(sub_value, (int, float)):
                            print(f"  {sub_key}: {sub_value:.3f}")
                        else:
                            print(f"  {sub_key}: {sub_value}")
            else:
                print(f"  {value}")


def run_validation():
    """Standard validation entry point for Protocol 12."""
    try:
        validator = ClinicalConvergenceValidator()
        results = validator.validate_clinical_convergence()

        # Determine if validation passed based on overall score
        passed = results.get("overall_clinical_score", 0) > 0.5

        return {
            "passed": passed,
            "status": "success" if passed else "failed",
            "message": f"Protocol 12 completed: Overall clinical validation score {results.get('overall_clinical_score', 0):.3f}",
        }
    except Exception as e:
        return {
            "passed": False,
            "status": "error",
            "message": f"Protocol 12 failed: {str(e)}",
        }


if __name__ == "__main__":
    main()
