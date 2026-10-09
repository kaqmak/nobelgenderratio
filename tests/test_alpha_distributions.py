"""Check that plotting the lognormal prior preserves mass outside the displayed range."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nobelgenderbias import NobelGenderBiasAnalysis
from nobelgenderbias import analysis as analysis_module


@pytest.fixture
def plot_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    existing_figures = set(plt.get_fignums())
    monkeypatch.setattr(analysis_module, "get_root_dir", lambda: tmp_path)
    monkeypatch.setattr(plt, "show", Mock())
    try:
        yield
    finally:
        for figure_number in set(plt.get_fignums()) - existing_figures:
            plt.close(figure_number)


@pytest.mark.parametrize("prize_names", [["Physics"], ["Physics", "Chemistry"]])
def test_alpha_histograms_preserve_tail_mass_and_share_bins(
    prize_names: list[str], plot_environment: None, tmp_path: Path
) -> None:
    n_fields = len(prize_names)
    prior_values = np.tile(np.array([0.25, 0.75, 3.0, 100.0])[:, None], (1, n_fields))
    posterior_values = np.tile(np.array([0.25, 0.75, 1.25, 3.0])[:, None], (1, n_fields))
    prior = SimpleNamespace(prior={"alpha": SimpleNamespace(values=prior_values[None, ...])})
    posterior = SimpleNamespace(
        posterior={"alpha": SimpleNamespace(values=posterior_values[None, ...])}
    )

    NobelGenderBiasAnalysis().plot_alpha_distributions(posterior, prior, prize_names)

    fig = plt.gcf()
    assert len(fig.axes) == n_fields
    for ax in fig.axes:
        assert ax.get_xlim() == (0, 2.5)
        assert len(ax.patches) == 400
        prior_bars, posterior_bars = ax.patches[:200], ax.patches[200:]
        np.testing.assert_allclose([bar.get_width() for bar in prior_bars], 0.0125)
        np.testing.assert_allclose([bar.get_width() for bar in posterior_bars], 0.0125)
        # The shown interval contains half the prior and three quarters of the posterior.
        assert sum(bar.get_height() * bar.get_width() for bar in prior_bars) == pytest.approx(0.5)
        assert sum(bar.get_height() * bar.get_width() for bar in posterior_bars) == pytest.approx(
            0.75
        )
        np.testing.assert_allclose(
            [bar.get_x() for bar in prior_bars], [bar.get_x() for bar in posterior_bars]
        )
        assert len(ax.lines) == 1
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 1])
    assert (tmp_path / "figs" / "alphaDistribution.pdf").is_file()
    np.testing.assert_array_equal(prior_values[:, 0], [0.25, 0.75, 3.0, 100.0])
    np.testing.assert_array_equal(posterior_values[:, 0], [0.25, 0.75, 1.25, 3.0])
