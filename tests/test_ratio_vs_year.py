"""Regression coverage for line-only workforce-ratio legends."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from nobelgenderbias import NobelGenderBiasAnalysis
from nobelgenderbias import analysis as analysis_module


def test_ratio_legend_uses_three_matching_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = NobelGenderBiasAnalysis()
    analysis.load_data()
    monkeypatch.setattr(analysis_module, "get_root_dir", lambda: tmp_path)

    fig, ax = analysis.plot_ratio_vs_year()
    try:
        legend = ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == [
            "Social Sciences",
            "Physical Sciences",
            "Life Sciences",
        ]
        assert len(legend.legend_handles) == 3
        assert len(ax.lines) == 3
        assert len(ax.collections) == 3
        for handle, line, points, color in zip(
            legend.legend_handles,
            ax.lines,
            ax.collections,
            analysis_module.PUBLICATION_COLORS[:3],
            strict=True,
        ):
            assert isinstance(handle, Line2D)
            assert handle.get_marker() == "None"
            assert to_rgba(handle.get_color()) == to_rgba(line.get_color()) == to_rgba(color)
            np.testing.assert_allclose(points.get_facecolor()[0, :3], to_rgba(color)[:3])
        assert (tmp_path / "figs" / "ratioVsYear.pdf").is_file()
    finally:
        plt.close(fig)
