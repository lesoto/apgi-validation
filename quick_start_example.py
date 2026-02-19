#!/usr/bin/env python3
"""
APGI Framework - Quick Start Example
=====================================

This example demonstrates basic usage of the APGI framework components
using sample data. Run this script to see the framework in action.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("🧠 APGI Framework - Quick Start Example")
    print("=" * 50)

    try:
        # 1. Cross-Species Scaling Example
        print("\n1. Cross-Species Consciousness Scaling")
        print("-" * 40)

        # Import the cross-species scaling module using importlib due to filename with hyphens
        import importlib

        apgi_cross_species = importlib.import_module("APGI-Cross-Species-Scaling")
        predict_species_consciousness = apgi_cross_species.predict_species_consciousness
        SpeciesParameters = apgi_cross_species.SpeciesParameters

        # Example: Human brain parameters
        human_params = SpeciesParameters(
            name="human",
            cortical_volume_mm3=500000,  # ~500 cm³
            cortical_thickness_mm=3.0,
            neuron_density_per_mm3=25000,
            synaptic_density_per_mm3=500000,
            conduction_velocity_m_s=50.0,
            body_mass_kg=70.0,
            brain_mass_g=1400.0,
        )

        predict_species_consciousness(human_params)
        print(f"Species: {human_params.name}")
        print(f"Consciousness Level: {human_params.consciousness_level:.3f}")
        print(f"Confidence: {human_params.confidence:.1f}")
        print(f"Processing Time: {human_params.processing_time:.3f}")

        # 2. Multimodal Integration Example
        print("\n2. Multimodal Data Integration")
        print("-" * 35)

        # Load sample multimodal data
        import pandas as pd

        data_path = (
            PROJECT_ROOT / "data_repository" / "raw_data" / "sample_multimodal_data.csv"
        )
        if data_path.exists():
            data = pd.read_csv(data_path)
            print(f"Loaded sample data: {len(data)} time points")
            print(
                f"Modalities available: {', '.join([col for col in data.columns if col not in ['subject_id', 'trial', 'time_ms']])}"
            )

            # Show basic statistics
            print("\nData Statistics:")
            for col in ["eeg_fz", "pupil_diameter", "eda"]:
                if col in data.columns:
                    print(f"  {col}: {data[col].mean():.3f}")
        else:
            print("Sample data not found. Run the framework setup first.")

        # 3. Formal Model Simulation
        print("\n3. Formal Model Simulation")
        print("-" * 28)

        print("Formal model simulation requires specific parameters.")
        print("Use: python main.py formal-model --simulation-steps 1000 --plot")

        # 4. Validation Protocols
        print("\n4. Validation Protocols")
        print("-" * 22)

        print("Available validation protocols can be run with:")
        print("python main.py validate --all-protocols")

        print("\n✅ Example completed successfully!")
        print("\nFor more advanced usage, see the documentation or run:")
        print("python main.py --help")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(
            "Make sure all dependencies are installed: pip install -r requirements.txt"
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check the framework setup and data files.")


if __name__ == "__main__":
    main()
