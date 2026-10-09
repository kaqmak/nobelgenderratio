"""
Example usage of Nobel Gender Bias Analysis
This script demonstrates how to run the analysis with your data
"""

import sys

import numpy as np
import pytensor

# Avoid PyTensor's macOS linker bug before nobelgenderbias imports PyMC.
if sys.platform == "darwin":
    pytensor.config.cxx = ""

from nobelgenderbias import NobelGenderBiasAnalysis  # noqa: E402


def main() -> None:
    """Run the Nobel gender bias analysis"""

    print("Nobel Gender Bias Analysis - Python Version")
    print("=" * 50)
    print("Using uv package manager with PyMC for Bayesian inference")
    print()

    # Initialize the analysis
    analysis = NobelGenderBiasAnalysis()

    try:
        # Run the full analysis
        trace_posterior, trace_prior, prize_names = analysis.run_full_analysis()
        analysis.plot_nobel_laureates_by_gender()

        print("\nAnalysis completed successfully!")
        print(f"Analyzed prizes: {prize_names}")

        # Display some basic statistics
        alpha_posterior = trace_posterior.posterior["alpha"].values.reshape(-1, len(prize_names))

        print("\nPosterior statistics for alpha parameters:")
        print("-" * 40)
        for i, prize in enumerate(prize_names):
            mean_alpha = np.mean(alpha_posterior[:, i])
            std_alpha = np.std(alpha_posterior[:, i])
            print(f"{prize:12}: {mean_alpha:.3f} ± {std_alpha:.3f}")

        # Optional: Run delay sweep analysis (warning: computationally intensive!)
        run_delay_sweep = (
            input("\nRun delay sweep analysis? This may take a while (y/n): ").lower().strip()
        )

        if run_delay_sweep == "y":
            print("Running delay sweep: lags 0–20, 50,000 draws per chain, 2 chains...")
            sweep_results = analysis.sweep_delay_analysis(range(21), draws=50_000, chains=2)
            analysis.plot_delay_sweep(sweep_results)
            analysis.plot_delay_distributions(sweep_results)
            print("Delay sweep analysis completed!")

    except FileNotFoundError as e:
        print(f"\nError: Required data file not found: {e}")
        print("\nPlease ensure the following files are present:")
        print("- genderDistribution.csv")
        print("- nobelLaurates.csv")
        print("- femaleNobelsL.csv (optional - will use synthetic data if missing)")

    # except Exception as e:
    #     print(f"\nAn error occurred during analysis: {e}")
    #     print("Please check your data files and try again.")


if __name__ == "__main__":
    main()
