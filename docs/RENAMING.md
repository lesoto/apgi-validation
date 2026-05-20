# APGI File Renaming Plan

New Convention: [FP|VP]_[NN]_[CamelCaseConcept].py

Falsification Files (13 protocol files)
Current Name    New Name    Status
FP_01_ActiveInference.py    FP_01_ActiveInference.py    ✅ No change
FP_02_AgentComparison_ConvergenceBenchmark.py    FP_02_AgentComparisonConvergenceBenchmark.py    ⚠️ Rename
FP_03_FrameworkLevel_MultiProtocol.py    FP_03_FrameworkLevelMultiProtocol.py    ⚠️ Rename
FP_04_PhaseTransition_EpistemicArchitecture.py    FP_04_PhaseTransitionEpistemicArchitecture.py    ⚠️ Rename
FP_05_EvolutionaryPlausibility.py    FP_05_EvolutionaryPlausibility.py    ✅ No change
FP_06_LiquidNetwork_EnergyBenchmark.py    FP_06_LiquidNetworkEnergyBenchmark.py    ⚠️ Rename
FP_07_MathematicalConsistency.py    FP_07_MathematicalConsistency.py    ✅ No change
FP_08_ParameterSensitivity_Identifiability.py    FP_08_ParameterSensitivityIdentifiability.py    ⚠️ Rename
FP_09_NeuralSignatures_P3b_HEP.py    FP_09_NeuralSignaturesP3bHEP.py    ⚠️ Rename
FP_10_BayesianEstimation_MCMC.py    FP_10_BayesianEstimationMCMC.py    ⚠️ Rename
FP_11_LiquidNetworkDynamicsEchoState.py    FP_11_LiquidNetworkDynamicsEchoState.py    ✅ No change
FP_12_CrossSpeciesScaling.py    FP_12_CrossSpeciesScaling.py    ✅ No change
Special Files (not protocol files):

FP_ALL_Aggregator.py → Keep as-is (aggregator)
Master_Falsification.py → Keep as-is (master orchestrator)
Validation Files (21 protocol files)
Current Name    New Name    Status
VP_01_SyntheticEEG_MLClassification.py    VP_01_SyntheticEEGMLClassification.py    ⚠️ Rename
VP_02_Behavioral_BayesianComparison.py    VP_02_BehavioralBayesianComparison.py    ⚠️ Rename
VP_03_ActiveInference_AgentSimulations.py    VP_03_ActiveInferenceAgentSimulations.py    ⚠️ Rename
VP_04_PhaseTransition_EpistemicLevel2.py    VP_04_PhaseTransitionEpistemicLevel2.py    ⚠️ Rename
VP_05_EvolutionaryEmergence.py    VP_05_EvolutionaryEmergence.py    ✅ No change
VP_06_LiquidNetwork_InductiveBias.py    VP_06_LiquidNetworkInductiveBias.py    ⚠️ Rename
VP_07_TMS_CausalInterventions.py    VP_07_TMSCausalInterventions.py    ⚠️ Rename
VP_08_Psychophysical_ThresholdEstimation.py    VP_08_PsychophysicalThresholdEstimation.py    ⚠️ Rename
VP_09_NeuralSignatures_EmpiricalPriority1.py    VP_09_NeuralSignaturesEmpiricalPriority1.py    ⚠️ Rename
VP_10_CausalManipulations_Priority2.py    VP_10_CausalManipulationsPriority2.py    ⚠️ Rename
VP_11_MCMC_CulturalNeuroscience_Priority3.py    VP_11_MCMCCulturalNeurosciencePriority3.py    ⚠️ Rename
VP_12_Clinical_CrossSpecies_Convergence.py    VP_12_ClinicalCrossSpeciesConvergence.py    ⚠️ Rename
VP_13_Epistemic_Architecture.py    VP_13_EpistemicArchitecture.py    ⚠️ Rename
VP_14_fMRI_Anticipation_Experience.py    VP_14_FMRIAnticipationExperience.py    ⚠️ Rename
VP_15_fMRI_Anticipation_vmPFC.py    VP_15_FMRIAnticipationVmPFC.py    ⚠️ Rename
VP_16_Metabolic_ATP_GroundTruth.py    VP_16_MetabolicATPGroundTruth.py    ⚠️ Rename
VP_17_AllenVisualCoding_Fatigue.py    VP_17_AllenVisualCodingFatigue.py    ⚠️ Rename
VP_18_EEG_Microstate_GFP_P3b.py    VP_18_EEGMicrostateGFPP3b.py    ⚠️ Rename
VP_19_InformationErasure_MVPA.py    VP_19_InformationErasureMVPA.py    ⚠️ Rename
VP_20_Empirical_iEEG.py    VP_20_EmpiricalIEEG.py    ⚠️ Rename
VP_21_FreeEnergy_PredictionError.py    VP_21_FreeEnergyPredictionError.py    ⚠️ Rename
Special Files (not protocol files):

VP_ALL_Aggregator.py → Keep as-is (aggregator)
Master_Validation.py → Keep as-is (master orchestrator)
Summary
Total Files to Rename: 23 files

Falsification: 6 files need renaming
Validation: 17 files need renaming
Files Already Compliant: 12 files

Falsification: 6 files
Validation: 6 files
Special Files (excluded from renaming): 4 files

FP_ALL_Aggregator.py
Master_Falsification.py
VP_ALL_Aggregator.py
Master_Validation.p
