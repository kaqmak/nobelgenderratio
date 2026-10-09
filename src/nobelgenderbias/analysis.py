"""
Nobel Gender Bias Analysis - Python Version
Converted from R script using RStan to Python using PyMC
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
from matplotlib.patches import Patch
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# Set style for publication-quality plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set1")

# Publication colors (matching R script)
PUBLICATION_COLORS = [
    "#386cb0",
    "#fdb462",
    "#7fc97f",
    "#ef3b2c",
    "#662506",
    "#a6cee3",
    "#fb9a99",
    "#984ea3",
    "#ffff33",
]


def set_publication_style(dark_theme=False):
    """Set publication-quality matp -> Nonelotlib style"""
    if dark_theme:
        plt.style.use("dark_background")
        text_color = "white"
        line_color = "white"
        bg_color = "black"
    else:
        text_color = "black"
        line_color = "black"
        bg_color = "white"

    plt.rcParams.update(
        {
            "figure.facecolor": bg_color,
            "axes.facecolor": bg_color,
            "axes.edgecolor": line_color,
            "axes.labelcolor": text_color,
            "xtick.color": text_color,
            "ytick.color": text_color,
            "text.color": text_color,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "font.size": 10,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
        }
    )


def get_root_dir() -> Path:
    return Path(__file__).parent.parent.parent


def _save_figure(fig: plt.Figure, save_path: str | Path) -> Path:
    output_path = Path(save_path)
    if not output_path.is_absolute():
        output_path = get_root_dir() / "figs" / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return output_path


class NobelGenderBiasAnalysis:
    def __init__(self):
        self.ratio_df = None
        self.nobel_df = None
        self.all_nobel_df = None
        self.tot_df = None
        self.mydf = None
        self.fit_results = {}

    def load_data(self):
        """Load and preprocess the data"""
        # Load gender distribution data
        gender_raw = pd.read_csv(get_root_dir() / "data/genderDistribution_v2.csv")
        # First, let's store the raw data for processing
        self.gender_raw = gender_raw

        # Load Nobel laureates data
        nobel_raw = pd.read_csv(get_root_dir() / "data/nobelLaurates.csv", encoding="utf-8")

        # Process Nobel laureates data
        self.nobel_df = self._process_nobel_data(nobel_raw)

        # Load female Nobel laureate data
        try:
            nobel_w = pd.read_csv(get_root_dir() / "data/femaleNobelsL.csv")
            self.all_nobel_df = self._merge_nobel_data(self.nobel_df, nobel_w)
            print(f"Successfully loaded female Nobel data: {len(self.all_nobel_df)} records")
        except FileNotFoundError:
            print(
                "Warning: femaleNobelsL.csv not found. Creating synthetic data for demonstration."
            )
            self.all_nobel_df = self._create_synthetic_nobel_data()

        # Process ratio data
        self._process_ratio_data()

    def _process_nobel_data(self, nobel_raw):
        """Process Nobel laureates data"""
        # Convert to long format
        nobel_long = nobel_raw.melt(id_vars=["Year"], var_name="prize", value_name="laureates")

        # Count laureates (split by semicolon)
        def count_laureates(x):
            if pd.isna(x) or x == "???" or x.lower() == "none":
                return 0
            return len([name.strip() for name in str(x).split(";") if name.strip()])

        nobel_long["nbl"] = nobel_long["laureates"].apply(count_laureates)

        # Rename prizes to match analysis
        prize_mapping = {
            "Physiology or Medicine": "PhysMed",
            "Physics": "Physics",
            "Chemistry": "Chemistry",
            "Economics": "Economics",
        }

        nobel_long = nobel_long[nobel_long["prize"].isin(prize_mapping.keys())].copy()
        nobel_long["prize"] = nobel_long["prize"].map(prize_mapping)

        return nobel_long[["Year", "prize", "nbl"]]

    def _create_synthetic_nobel_data(self):
        """Create synthetic female Nobel data for demonstration"""
        # This is a simplified version - in practice, you'd need real data
        years = range(1901, 2019)
        prizes = ["Physics", "Chemistry", "PhysMed", "Economics"]

        data = []
        for year in years:
            for prize in prizes:
                # Create synthetic female counts (very low historically, increasing over time)
                if prize == "Economics" and year < 1969:
                    nbl_f = 0  # Economics prize started in 1969
                else:
                    # Simple synthetic model: very few female winners, slightly increasing over time
                    base_rate = 0.02 if year < 1950 else 0.05 if year < 1990 else 0.1
                    nbl_f = np.random.binomial(3, base_rate)  # Max 3 winners typically

                data.append({"Year": year, "prize": prize, "nbl_f": nbl_f})

        female_df = pd.DataFrame(data)

        # Merge with total Nobel data
        merged = self.nobel_df.merge(female_df, on=["Year", "prize"], how="outer")
        merged["nbl_f"] = merged["nbl_f"].fillna(0)
        return merged

    def _merge_nobel_data(self, nobel_df, nobel_w):
        """Merge Nobel data with female laureate data"""
        # Process the female Nobel data structure
        female_long = nobel_w.melt(id_vars=["Year"], var_name="column", value_name="nbl_f")

        # Extract prize and gender information from column names
        female_long["prize_gender"] = female_long["column"].str.split("_")
        female_long["prize"] = female_long["prize_gender"].str[0]
        female_long["data_type"] = female_long["prize_gender"].str[1]

        # Keep only female counts, not total counts
        female_counts = female_long[female_long["data_type"] == "female"].copy()

        # Map prize names to match our analysis
        prize_mapping = {
            "Physics": "Physics",
            "Chemistry": "Chemistry",
            "Economics": "Economics",
            "Physiology.or.Medicine": "PhysMed",
        }

        female_counts["prize"] = female_counts["prize"].map(prize_mapping)
        female_counts = female_counts.dropna(subset=["prize"])

        # Clean and merge
        female_counts = female_counts[["Year", "prize", "nbl_f"]].copy()
        female_counts["nbl_f"] = pd.to_numeric(female_counts["nbl_f"], errors="coerce").fillna(0)

        # Merge with total Nobel data
        merged = nobel_df.merge(female_counts, on=["Year", "prize"], how="left")
        merged["nbl_f"] = merged["nbl_f"].fillna(0)

        return merged

    def _process_ratio_data(self):
        """Process gender ratio data"""
        # Use the raw gender data we loaded
        # Convert to long format
        ratio_long = self.gender_raw.melt(
            id_vars=["year", "gender", "position"], var_name="subject", value_name="value"
        )

        # Pivot to get both genders
        ratio_pivot = ratio_long.pivot_table(
            index=["year", "position", "subject"], columns="gender", values="value"
        ).reset_index()

        # Map subjects to prizes
        subject_mapping = {
            "Physical sciences": "Physics",
            "Social sciences": "Economics",
            "Life sciences": "PhysMed",
        }

        ratio_pivot["prize"] = ratio_pivot["subject"].map(subject_mapping)
        ratio_pivot = ratio_pivot.dropna(subset=["prize"])

        # Calculate ratio (handle division by zero)
        ratio_pivot["both"] = ratio_pivot["female"] + ratio_pivot["male"]
        ratio_pivot["ratio"] = ratio_pivot["female"] / ratio_pivot["both"].replace(0, np.nan)

        # Copy Physics data to Chemistry (assumption from original script)
        chem_data = ratio_pivot[ratio_pivot["prize"] == "Physics"].copy()
        chem_data["prize"] = "Chemistry"

        self.ratio_df = pd.concat([ratio_pivot, chem_data], ignore_index=True)
        self.ratio_df.rename(columns={"year": "Year"}, inplace=True)

    def fit_sigmoid_model(self, df):
        """Fit sigmoid model to gender ratio data"""

        def sigmoid(x, A, g, y00, off):
            return A / (1 + np.exp(-(x - 2000 - y00) / g)) + off

        try:
            popt, _ = curve_fit(
                sigmoid,
                df["Year"],
                df["ratio"],
                p0=[0.5, 10, 10, 0.01],
                bounds=([0, 0.1, -30, 0.0], [1, 20, 30, 0.2]),
                maxfev=5000,
            )
            return popt
        except Exception:
            # Return default parameters if fitting fails
            return [0.5, 10, 10, 0.01]

    def extrapolate_ratio(self, delay=10):
        """Extrapolate gender ratios with delay"""
        results = []

        for (prize, position), group in self.ratio_df.groupby(["prize", "position"]):
            if len(group) > 5:  # Need enough data points
                params = self.fit_sigmoid_model(group)

                # Generate predictions
                years = np.arange(1901, 2026)
                delayed_years = years - delay

                def sigmoid(x, A, g, y00, off):
                    return A / (1 + np.exp(-(x - 2000 - y00) / g)) + off

                ratios = sigmoid(delayed_years, *params)

                for year, ratio in zip(years, ratios, strict=False):
                    results.append(
                        {"Year": year, "prize": prize, "position": position, "ratioEx": ratio}
                    )

        return pd.DataFrame(results)

    def prepare_stan_data(self, position="senior", delay=10):
        """Prepare model data using workforce ratios from Year - delay."""
        ratio_ex_df = self.extrapolate_ratio(delay=delay)
        delayed_ratios = self.ratio_df.copy()
        # Align observed workforce years with the corresponding award years.
        delayed_ratios["Year"] = delayed_ratios["Year"] + delay

        # Merge all dataframes
        self.mydf = self.all_nobel_df.merge(ratio_ex_df, on=["Year", "prize"], how="left").merge(
            delayed_ratios, on=["Year", "prize", "position"], how="left"
        )

        # Use extrapolated ratio where original is missing
        self.mydf["ratio"] = self.mydf["ratio"].fillna(self.mydf["ratioEx"])

        # Filter for position
        df_filtered = self.mydf[self.mydf["position"] == position].copy()
        if df_filtered.empty:
            raise ValueError(f"No data available for position {position!r}")

        # Keep all matrices aligned and preserve missing values for validation.
        model_data = df_filtered.pivot_table(
            index="Year", columns="prize", values=["ratio", "nbl_f", "nbl"], dropna=False
        )
        r_matrix = model_data["ratio"].to_numpy()
        y_matrix = model_data["nbl_f"].to_numpy()
        total_matrix = model_data["nbl"].to_numpy()
        prize_names = model_data["ratio"].columns.tolist()

        if not np.all(np.isfinite(r_matrix) & (r_matrix >= 0) & (r_matrix <= 1)):
            raise ValueError("Gender ratios must be finite and between 0 and 1")

        for name, counts in (("Female", y_matrix), ("Total", total_matrix)):
            if not np.all(np.isfinite(counts)):
                raise ValueError(f"{name} laureate counts must be finite, nonnegative integers")
            if not np.all((counts >= 0) & (counts == np.floor(counts))):
                raise ValueError(f"{name} laureate counts must be finite, nonnegative integers")

        if np.any(y_matrix > total_matrix):
            raise ValueError("Female laureate counts cannot exceed total laureate counts")

        # Validate before converting so fractional counts cannot be silently truncated.
        y_matrix = y_matrix.astype(int)
        total_matrix = total_matrix.astype(int)

        # Debug information
        print(f"Data shapes: r_matrix={r_matrix.shape}, y_matrix={y_matrix.shape}")
        print(f"Prize names: {prize_names}")
        print(f"r_matrix range: {r_matrix.min():.4f} to {r_matrix.max():.4f}")
        print(f"y_matrix range: {y_matrix.min()} to {y_matrix.max()}")

        # Show some sample data for each prize
        for i, prize in enumerate(prize_names):
            y_col = y_matrix[:, i]
            r_col = r_matrix[:, i]
            total_col = total_matrix[:, i]
            nonzero_y = y_col[y_col > 0]
            if len(nonzero_y) > 0:
                print(f"{prize}: {len(nonzero_y)} years with female winners, max={nonzero_y.max()}")
                print(f"  Sample r values: {r_col[y_col > 0][:5]}")
                print(f"  Sample total values: {total_col[y_col > 0][:5]}")

        return r_matrix, y_matrix, total_matrix, prize_names

    def build_pymc_model(
        self,
        r_matrix: np.ndarray,
        y_matrix: np.ndarray,
        total_matrix: np.ndarray,
        sample_priors: bool = False,
    ) -> pm.Model:
        """Build the paper's hierarchical model with alpha multiplying workforce odds."""
        r_matrix = np.asarray(r_matrix, dtype=float)
        if not np.all(np.isfinite(r_matrix) & (r_matrix >= 0) & (r_matrix <= 1)):
            raise ValueError("Gender ratios must be finite and between 0 and 1")
        _, J = r_matrix.shape

        if not sample_priors and np.any(
            ((r_matrix == 0) & (y_matrix > 0)) | ((r_matrix == 1) & (y_matrix < total_matrix))
        ):
            raise ValueError(
                "Observed female counts are incompatible with workforce ratios of exactly 0 or 1"
            )

        # Dummy interior ratios keep logits finite; exact endpoints are restored below.
        interior_ratios = np.where((r_matrix > 0) & (r_matrix < 1), r_matrix, 0.5)
        workforce_log_odds = np.log(interior_ratios) - np.log1p(-interior_ratios)

        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0, sigma=1)
            sigma = pm.TruncatedNormal("sigma", mu=1, sigma=0.5, lower=0)
            # Noncentering avoids a funnel when the fields' log-odds effects are similar.
            log_alpha_offset = pm.Normal("log_alpha_offset", mu=0, sigma=1, shape=J)
            log_alpha = pm.Deterministic("log_alpha", mu + sigma * log_alpha_offset)
            pm.Deterministic("alpha", pm.math.exp(log_alpha))

            interior_theta = pm.math.sigmoid(workforce_log_odds + log_alpha[None, :])
            theta = pm.Deterministic(
                "theta",
                pm.math.switch(
                    r_matrix == 0, 0.0, pm.math.switch(r_matrix == 1, 1.0, interior_theta)
                ),
            )

            if not sample_priors:
                for j in range(J):
                    pm.Binomial(
                        f"y_{j}", n=total_matrix[:, j], p=theta[:, j], observed=y_matrix[:, j]
                    )

        return model

    def run_inference(
        self,
        r_matrix,
        y_matrix,
        total_matrix,
        draws=10_000,
        chains=2,
        sample_priors=True,
        var_names=None,
    ):
        """Run PyMC inference, optionally skipping priors or limiting stored variables."""
        # Posterior sampling
        model_posterior = self.build_pymc_model(
            r_matrix, y_matrix, total_matrix, sample_priors=False
        )

        with model_posterior:
            trace_posterior = pm.sample(
                draws=draws,
                chains=chains,
                tune=1000,
                target_accept=0.99,
                return_inferencedata=True,
                var_names=var_names,
            )

        if not sample_priors:
            return trace_posterior, None

        # Prior sampling
        model_prior = self.build_pymc_model(r_matrix, y_matrix, total_matrix, sample_priors=True)

        with model_prior:
            trace_prior = pm.sample_prior_predictive(draws=draws * chains)

        return trace_posterior, trace_prior

    def plot_alpha_distributions(
        self, trace_posterior, trace_prior, prize_names, save_path="alphaDistribution.pdf"
    ) -> None:
        """Plot alpha parameter distributions"""
        set_publication_style()

        # Extract alpha samples
        posterior_alphas = trace_posterior.posterior["alpha"].values.reshape(-1, len(prize_names))
        prior_alphas = trace_prior.prior["alpha"].values.reshape(-1, len(prize_names))
        bins = np.linspace(0, 2.5, 151)
        bin_width = bins[1] - bins[0]

        fig, axes = plt.subplots(len(prize_names), 1, figsize=(6, 8), sharex=True)
        if len(prize_names) == 1:
            axes = [axes]

        for i, (ax, prize) in enumerate(zip(axes, prize_names, strict=False)):
            # Normalize by all draws so the long lognormal tail is not rescaled into this window.
            ax.hist(
                prior_alphas[:, i],
                bins=bins,
                weights=np.full(len(prior_alphas), 1 / (len(prior_alphas) * bin_width)),
                alpha=0.6,
                color="slategrey",
                label="Prior",
            )
            ax.hist(
                posterior_alphas[:, i],
                bins=bins,
                weights=np.full(len(posterior_alphas), 1 / (len(posterior_alphas) * bin_width)),
                alpha=0.8,
                color="orange",
                label="Posterior",
            )
            ax.axvline(1, color="black", linestyle="--", linewidth=1, label="Proportional")

            ax.set_ylabel(f"{prize}\nDensity")
            ax.set_xlim(0, 2.5)
            if i == 0:
                ax.legend()

        axes[-1].set_xlabel(r"$\alpha$")
        plt.tight_layout()
        _save_figure(fig, save_path)
        plt.show()

    def plot_ratio_vs_year(self, save_path="ratioVsYear.pdf") -> tuple[plt.Figure, np.ndarray]:
        """Plot gender ratio vs year"""
        set_publication_style()

        # Get extrapolated ratios for plotting
        ratio_ex_plot = self.extrapolate_ratio(delay=0)

        # Merge with original data
        plot_data = ratio_ex_plot.merge(self.ratio_df, on=["Year", "prize", "position"], how="left")

        # Filter for senior position and exclude Chemistry
        plot_data = plot_data[
            (plot_data["position"] == "senior")
            & (plot_data["prize"] != "Chemistry")
            & (plot_data["prize"].notna())
        ].copy()

        # Set Economics to NaN before 1969
        plot_data.loc[
            (plot_data["prize"] == "Economics") & (plot_data["Year"] < 1969), "ratioEx"
        ] = np.nan

        fig, ax = plt.subplots(figsize=(7, 4))

        for i, prize in enumerate(["Economics", "Physics", "PhysMed"]):
            prize_data = plot_data[plot_data["prize"] == prize]
            color = PUBLICATION_COLORS[i]

            # Plot extrapolated line
            ax.plot(
                prize_data["Year"], prize_data["ratioEx"], color=color, label=prize, linewidth=2
            )

            # Plot actual data points
            actual_data = prize_data.dropna(subset=["ratio"])
            ax.scatter(actual_data["Year"], actual_data["ratio"], color=color, s=30, alpha=0.8)

        ax.set_xlabel("Year")
        ax.set_ylabel("Gender ratio r")
        ax.legend(
            handles=ax.lines, labels=["Social Sciences", "Physical Sciences", "Life Sciences"]
        )
        plt.tight_layout()
        _save_figure(fig, save_path)
        # plt.show()
        return fig, ax

    def plot_nobel_laureates_by_gender(
        self, save_path: str | Path | None = "nobelLaureatesByGender.pdf"
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot annual male and female laureate counts by prize."""
        if self.all_nobel_df is None:
            raise ValueError("Load the Nobel laureate data before plotting")

        set_publication_style()
        plot_data = self.all_nobel_df.copy()
        plot_data["nbl_f"] = plot_data["nbl_f"].fillna(0)
        plot_data["nbl_m"] = (plot_data["nbl"].fillna(0) - plot_data["nbl_f"]).clip(lower=0)
        prizes = ["Chemistry", "Economics", "Physics", "PhysMed"]
        plot_data = plot_data[plot_data["prize"].isin(prizes)]

        fig, axes = plt.subplots(len(prizes), 1, figsize=(12, 8), sharex=True)
        female_color = "#e41a1c"
        male_color = "#377eb8"

        for ax, prize in zip(axes, prizes, strict=False):
            prize_data = plot_data[plot_data["prize"] == prize].sort_values("Year")
            ax.bar(prize_data["Year"], prize_data["nbl_m"], color=male_color, width=0.9)
            ax.bar(
                prize_data["Year"],
                prize_data["nbl_f"],
                bottom=prize_data["nbl_m"],
                color=female_color,
                width=0.9,
            )
            ax.text(
                1.02,
                0.5,
                prize,
                transform=ax.transAxes,
                rotation=-90,
                va="center",
                ha="center",
                fontweight="bold",
                bbox={"facecolor": "#f0f0f0", "edgecolor": "none", "boxstyle": "square,pad=0.3"},
            )

        first_year = int(plot_data["Year"].min())
        last_year = int(plot_data["Year"].max())
        first_tick = first_year // 25 * 25
        last_tick = (last_year + 24) // 25 * 25
        axes[-1].set_xticks(np.arange(first_tick, last_tick + 1, 25))
        axes[-1].set_xlim(first_tick - 5, last_tick)
        axes[-1].set_xlabel("Year", fontweight="bold")
        fig.supylabel("Total", fontweight="bold")
        fig.legend(
            handles=[
                Patch(color=female_color, label="Female"),
                Patch(color=male_color, label="Male"),
            ],
            title="Nobel laureates",
            loc="center left",
            bbox_to_anchor=(0.81, 0.5),
        )
        fig.subplots_adjust(left=0.08, right=0.78, hspace=0.18)

        if save_path is not None:
            _save_figure(fig, save_path)
        return fig, axes

    def sweep_delay_analysis(self, delay_range=range(21), draws=50_000, chains=2):
        """Sample each lag with enough draws to resolve small posterior tail probabilities."""
        results = []

        for delay in delay_range:
            print(f"Processing delay: {delay}")
            r_matrix, y_matrix, total_matrix, prize_names = self.prepare_stan_data(delay=delay)

            # Only alpha and its hyperparameters are needed; theta and prior draws
            # would consume substantial memory at this sampling size.
            trace_posterior, _ = self.run_inference(
                r_matrix,
                y_matrix,
                total_matrix,
                draws=draws,
                chains=chains,
                sample_priors=False,
                var_names=["mu", "sigma", "alpha"],
            )
            alpha_samples = trace_posterior.posterior["alpha"].values.reshape(-1, len(prize_names))
            results.append(
                pd.DataFrame(
                    {
                        "delay": delay,
                        "Parameter": np.tile(prize_names, len(alpha_samples)),
                        "value": alpha_samples.ravel(),
                    }
                )
            )

        if not results:
            return pd.DataFrame(columns=["delay", "Parameter", "value"])
        return pd.concat(results, ignore_index=True)

    def plot_delay_distributions(
        self, sweep_results: pd.DataFrame, save_path: str | Path = "meanVsDelay.pdf"
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot posterior alpha distributions at even lags in stacked field panels."""
        plot_data = sweep_results[sweep_results["delay"] % 2 == 0].copy()
        if plot_data.empty:
            raise ValueError("No samples available at even-numbered delays")
        set_publication_style()

        fields = sorted(plot_data["Parameter"].unique())
        delays = sorted(plot_data["delay"].unique())
        fig, axes = plt.subplots(
            len(fields), 1, figsize=(6, 2 * len(fields)), sharex=True, squeeze=False
        )
        axes = axes.ravel()

        for i, (ax, field) in enumerate(zip(axes, fields, strict=True)):
            field_data = plot_data[plot_data["Parameter"] == field]
            sns.violinplot(
                data=field_data,
                x="delay",
                y="value",
                order=delays,
                ax=ax,
                color=PUBLICATION_COLORS[i % len(PUBLICATION_COLORS)],
                inner="box",
            )
            ax.set_ylabel(field)
            ax.set_xlabel("")
            # Keep the original 0–1 scale unless it would hide posterior samples.
            ax.set_ylim(0, max(1.0, 1.05 * field_data["value"].max()))

        axes[-1].set_xlabel("Lag δ [years]")
        plt.tight_layout()
        _save_figure(fig, save_path)
        plt.show()
        return fig, axes

    def plot_delay_sweep(self, sweep_results, save_path="probabilityVsDelay.pdf"):
        """Plot P(alpha >= 1) = 1 - count(alpha < 1) / count(alpha) for each lag."""
        set_publication_style()

        # Counting samples avoids histogram bins straddling the threshold at 1.
        plot_data = (
            sweep_results.groupby(["delay", "Parameter"])["value"]
            .agg(lambda samples: 1.0 - (samples < 1).mean())
            .rename("probability")
            .reset_index()
        )
        fields = ["Economics", "Physics", "PhysMed", "Chemistry"]
        fields.extend(param for param in plot_data["Parameter"].unique() if param not in fields)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, param in enumerate(fields):
            param_data = plot_data[plot_data["Parameter"] == param].sort_values("delay")
            if param_data.empty:
                continue
            ax.plot(
                param_data["delay"],
                param_data["probability"],
                marker="o",
                color=PUBLICATION_COLORS[i % len(PUBLICATION_COLORS)],
                label=param,
            )

        ax.set_xlabel("Lag δ [years]")
        ax.set_ylabel(r"$P(\alpha \geq 1)$")
        ax.set_xticks(np.arange(0, 21, 5))
        ax.legend(title="Scientific field", loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.tight_layout()
        _save_figure(fig, save_path)
        plt.show()
        return fig, ax

    def run_full_analysis(self, delay=15):
        """Run the complete analysis pipeline with a default 15-year workforce lag."""
        print("Loading data...")
        self.load_data()

        print(f"Preparing data for inference with a {delay}-year lag...")
        r_matrix, y_matrix, total_matrix, prize_names = self.prepare_stan_data(delay=delay)

        print("Running Bayesian inference...")
        trace_posterior, trace_prior = self.run_inference(r_matrix, y_matrix, total_matrix)

        print("Creating plots...")
        self.plot_alpha_distributions(trace_posterior, trace_prior, prize_names)
        self.plot_ratio_vs_year()

        print("Analysis complete!")

        return trace_posterior, trace_prior, prize_names


# Example usage
if __name__ == "__main__":
    # Initialize analysis
    analysis = NobelGenderBiasAnalysis()

    # Run full analysis
    trace_posterior, trace_prior, prize_names = analysis.run_full_analysis()

    # Optional: Run delay sweep analysis (computationally intensive)
    # print("Running delay sweep analysis...")
    # sweep_results = analysis.sweep_delay_analysis(range(21))  # Every delay from 0 to 20
    # analysis.plot_delay_sweep(sweep_results)
