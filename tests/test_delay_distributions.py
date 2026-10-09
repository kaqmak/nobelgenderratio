"""Regression tests for the restored posterior distribution plot."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from nobelgenderbias import NobelGenderBiasAnalysis
from nobelgenderbias import analysis as analysis_module
from scripts import run_analysis


@pytest.mark.parametrize("filename", ["meanVsDelay.pdf", "nested/distributions.pdf"])
def test_delay_distribution_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    rng = np.random.default_rng(42)
    samples = pd.DataFrame(
        [
            {"delay": delay, "Parameter": field, "value": value}
            for field in ["Physics", "PhysMed", "Economics", "Chemistry"]
            for delay in [20, 1, 0, 2]
            for value in rng.uniform(0.1, 0.9, size=30)
        ]
    )
    samples.loc[samples["Parameter"] == "Physics", "value"] += 1
    original_samples = samples.copy(deep=True)
    violinplot = Mock(wraps=analysis_module.sns.violinplot)
    monkeypatch.setattr(analysis_module.sns, "violinplot", violinplot)
    monkeypatch.setattr(analysis_module, "get_root_dir", lambda: tmp_path)
    monkeypatch.setattr(plt, "show", Mock())

    fig, axes = NobelGenderBiasAnalysis().plot_delay_distributions(samples, save_path=filename)
    try:
        fields = ["Chemistry", "Economics", "PhysMed", "Physics"]
        assert len(axes) == 4
        assert list(fig.axes) == list(axes)
        assert [ax.get_ylabel() for ax in axes] == fields
        assert [ax.get_xlabel() for ax in axes] == ["", "", "", "Lag δ [years]"]
        assert [tick.get_text() for tick in axes[-1].get_xticklabels()] == ["0", "2", "20"]
        assert violinplot.call_count == 4
        for i, (ax, plotting_call) in enumerate(zip(axes, violinplot.call_args_list, strict=True)):
            kwargs = plotting_call.kwargs
            assert kwargs["ax"] is ax
            assert kwargs["data"]["Parameter"].unique().tolist() == [fields[i]]
            assert set(kwargs["data"]["delay"]) == {0, 2, 20}
            assert kwargs["order"] == [0, 2, 20]
            assert kwargs["color"] == analysis_module.PUBLICATION_COLORS[i]
            assert kwargs["inner"] == "box"
            assert len(ax.collections) == 3
            assert ax.get_ylim()[0] == 0
            assert ax.get_ylim()[1] >= kwargs["data"]["value"].max()
            assert axes[-1].get_shared_x_axes().joined(ax, axes[-1])
        assert axes[0].get_ylim()[1] == 1
        assert (tmp_path / "figs" / filename).is_file()
        pd.testing.assert_frame_equal(samples, original_samples)
    finally:
        plt.close(fig)


def test_delay_distribution_single_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    samples = pd.DataFrame(
        {"delay": [0, 0, 0, 2, 2, 2], "Parameter": "Physics", "value": [0.1, 0.3, 0.5] * 2}
    )
    monkeypatch.setattr(plt, "show", Mock())
    output = tmp_path / "distribution.pdf"

    fig, axes = NobelGenderBiasAnalysis().plot_delay_distributions(samples, save_path=output)
    try:
        assert axes.shape == (1,)
        assert axes[0].get_ylabel() == "Physics"
        assert axes[0].get_xlabel() == "Lag δ [years]"
        assert output.is_file()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("delays", [[], [1, 3]])
def test_delay_distribution_requires_even_lag_samples(delays: list[int]) -> None:
    samples = pd.DataFrame({"delay": delays, "Parameter": "Physics", "value": 0.5})
    with pytest.raises(ValueError, match="No samples available at even-numbered delays"):
        NobelGenderBiasAnalysis().plot_delay_distributions(samples)


def test_runner_creates_both_plots_from_one_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    analysis = Mock(spec=NobelGenderBiasAnalysis)
    posterior = SimpleNamespace(posterior={"alpha": SimpleNamespace(values=np.array([0.5]))})
    analysis.run_full_analysis.return_value = posterior, None, ["Physics"]
    sweep_results = object()
    analysis.sweep_delay_analysis.return_value = sweep_results
    monkeypatch.setattr(run_analysis, "NobelGenderBiasAnalysis", Mock(return_value=analysis))
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    run_analysis.main()

    analysis.sweep_delay_analysis.assert_called_once_with(range(21), draws=50_000, chains=2)
    analysis.plot_delay_sweep.assert_called_once_with(sweep_results)
    analysis.plot_delay_distributions.assert_called_once_with(sweep_results)
