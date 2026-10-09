"""Regression tests for delay sweeps without expensive Bayesian sampling."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

from nobelgenderbias import NobelGenderBiasAnalysis
from nobelgenderbias import analysis as analysis_module


@pytest.fixture(params=[["Physics", "Chemistry"], ["Physics"]], ids=["two-fields", "one-field"])
def mocked_sweep(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> tuple[NobelGenderBiasAnalysis, Mock, Mock, list[str]]:
    analysis = NobelGenderBiasAnalysis()
    prize_names = request.param
    n_fields = len(prize_names)
    alpha_values = np.arange(6 * n_fields, dtype=float).reshape(2, 3, n_fields) / 10

    def prepare_data(*, delay: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        return (
            np.full((2, n_fields), delay / 100),
            np.zeros((2, n_fields), dtype=int),
            np.ones((2, n_fields), dtype=int),
            prize_names,
        )

    def infer(
        r_matrix: np.ndarray, *_args: object, **_kwargs: object
    ) -> tuple[SimpleNamespace, None]:
        delay = round(r_matrix[0, 0] * 100)
        trace = SimpleNamespace(posterior={"alpha": SimpleNamespace(values=alpha_values + delay)})
        return trace, None

    prepare = Mock(side_effect=prepare_data)
    inference = Mock(side_effect=infer)
    monkeypatch.setattr(analysis, "prepare_stan_data", prepare)
    monkeypatch.setattr(analysis, "run_inference", inference)
    return analysis, prepare, inference, prize_names


@pytest.mark.parametrize(
    "kwargs, delays, draws, chains",
    [
        pytest.param({}, list(range(21)), 50_000, 2, id="defaults-zero-through-twenty"),
        pytest.param(
            {"delay_range": [9, 1, 4], "draws": 17, "chains": 3},
            [9, 1, 4],
            17,
            3,
            id="custom-delays-and-sampling",
        ),
    ],
)
def test_sweep_forwards_delays_and_sampling_and_aligns_raw_samples(
    mocked_sweep: tuple[NobelGenderBiasAnalysis, Mock, Mock, list[str]],
    kwargs: dict[str, object],
    delays: list[int],
    draws: int,
    chains: int,
) -> None:
    analysis, prepare, inference, prize_names = mocked_sweep

    results = analysis.sweep_delay_analysis(**kwargs)

    assert prepare.call_args_list == [call(delay=delay) for delay in delays]
    assert inference.call_count == len(delays)
    for delay, inference_call in zip(delays, inference.call_args_list, strict=True):
        r_matrix, y_matrix, total_matrix = inference_call.args
        np.testing.assert_array_equal(r_matrix, np.full((2, len(prize_names)), delay / 100))
        np.testing.assert_array_equal(y_matrix, np.zeros((2, len(prize_names)), dtype=int))
        np.testing.assert_array_equal(total_matrix, np.ones((2, len(prize_names)), dtype=int))
        assert inference_call.kwargs == {
            "draws": draws,
            "chains": chains,
            "sample_priors": False,
            "var_names": ["mu", "sigma", "alpha"],
        }

    assert isinstance(results, pd.DataFrame)
    assert results.columns.tolist() == ["delay", "Parameter", "value"]
    expected = pd.DataFrame(
        [
            {
                "delay": delay,
                "Parameter": prize,
                "value": delay + (sample * len(prize_names) + field) / 10,
            }
            for delay in delays
            for field, prize in enumerate(prize_names)
            for sample in range(6)
        ]
    )
    sort_columns = ["delay", "Parameter", "value"]
    pd.testing.assert_frame_equal(
        results.sort_values(sort_columns).reset_index(drop=True),
        expected.sort_values(sort_columns).reset_index(drop=True),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "kwargs, draws, chains, sample_priors, var_names",
    [
        pytest.param({}, 10_000, 2, True, None, id="default-prior-sampling"),
        pytest.param(
            {"draws": 7, "chains": 3, "sample_priors": False, "var_names": ["mu", "alpha"]},
            7,
            3,
            False,
            ["mu", "alpha"],
            id="skip-priors-and-select-variables",
        ),
        pytest.param(
            {"draws": 11, "chains": 1, "sample_priors": True, "var_names": ["alpha"]},
            11,
            1,
            True,
            ["alpha"],
            id="explicit-priors-and-selected-variables",
        ),
    ],
)
def test_run_inference_can_skip_priors_and_select_posterior_variables(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    draws: int,
    chains: int,
    sample_priors: bool,
    var_names: list[str] | None,
) -> None:
    analysis = NobelGenderBiasAnalysis()
    r_matrix = np.array([[0.25]])
    y_matrix = np.array([[1]])
    total_matrix = np.array([[3]])
    model = MagicMock()
    build_model = Mock(return_value=model)
    posterior = object()
    prior = object()
    sample = Mock(return_value=posterior)
    sample_prior = Mock(return_value=prior)
    monkeypatch.setattr(analysis, "build_pymc_model", build_model)
    monkeypatch.setattr(analysis_module.pm, "sample", sample)
    monkeypatch.setattr(analysis_module.pm, "sample_prior_predictive", sample_prior)

    actual_posterior, actual_prior = analysis.run_inference(
        r_matrix, y_matrix, total_matrix, **kwargs
    )

    assert actual_posterior is posterior
    sample.assert_called_once()
    assert sample.call_args.kwargs["draws"] == draws
    assert sample.call_args.kwargs["chains"] == chains
    assert sample.call_args.kwargs.get("var_names") == var_names
    expected_builds = [call(r_matrix, y_matrix, total_matrix, sample_priors=False)]
    if sample_priors:
        assert actual_prior is prior
        sample_prior.assert_called_once_with(draws=draws * chains)
        expected_builds.append(call(r_matrix, y_matrix, total_matrix, sample_priors=True))
    else:
        assert actual_prior is None
        sample_prior.assert_not_called()
    assert build_model.call_args_list == expected_builds
    assert model.__enter__.call_count == len(expected_builds)


@pytest.mark.parametrize(
    "kwargs, expected_delay", [({}, 15), ({"delay": 0}, 0), ({"delay": 20}, 20)]
)
def test_full_analysis_and_api_share_sampling_defaults(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, int], expected_delay: int
) -> None:
    analysis = NobelGenderBiasAnalysis()
    r_matrix = np.array([[0.25]])
    y_matrix = np.array([[1]])
    total_matrix = np.array([[3]])
    prize_names = ["Physics"]
    posterior, prior = object(), object()
    sample = Mock(return_value=posterior)
    sample_prior = Mock(return_value=prior)
    monkeypatch.setattr(analysis, "load_data", Mock())
    prepare_data = Mock(return_value=(r_matrix, y_matrix, total_matrix, prize_names))
    monkeypatch.setattr(analysis, "prepare_stan_data", prepare_data)
    monkeypatch.setattr(analysis, "build_pymc_model", Mock(return_value=MagicMock()))
    plot_alpha = Mock()
    monkeypatch.setattr(analysis, "plot_alpha_distributions", plot_alpha)
    monkeypatch.setattr(analysis, "plot_ratio_vs_year", Mock())
    monkeypatch.setattr(analysis_module.pm, "sample", sample)
    monkeypatch.setattr(analysis_module.pm, "sample_prior_predictive", sample_prior)

    assert analysis.run_inference(r_matrix, y_matrix, total_matrix) == (posterior, prior)
    assert analysis.run_full_analysis(**kwargs) == (posterior, prior, prize_names)
    prepare_data.assert_called_once_with(delay=expected_delay)

    assert sample.call_count == 2
    for sampling_call in sample.call_args_list:
        assert sampling_call.kwargs["draws"] == 10_000
        assert sampling_call.kwargs["chains"] == 2
    assert sample_prior.call_args_list == [call(draws=20_000), call(draws=20_000)]
    plot_alpha.assert_called_once_with(posterior, prior, prize_names)


@pytest.fixture
def plot_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Mock]:
    existing_figures = set(plt.get_fignums())
    monkeypatch.setattr(analysis_module, "get_root_dir", lambda: tmp_path)
    monkeypatch.setattr(plt, "show", Mock())
    save_figure = Mock(wraps=analysis_module._save_figure)
    monkeypatch.setattr(analysis_module, "_save_figure", save_figure)
    try:
        yield save_figure
    finally:
        for figure_number in set(plt.get_fignums()) - existing_figures:
            plt.close(figure_number)


@pytest.fixture
def sweep_samples() -> pd.DataFrame:
    samples = {
        "Chemistry": {5: [0.2, 0.3], 1: [1.0, 1.2, 0.5], 0: [1.0]},
        "PhysMed": {5: [0.5, 1.0], 1: [0.1, 0.2, 1.0], 0: [1.1]},
        "Physics": {5: [0.7, 0.8, 0.9, 1.1], 1: [1.0], 0: [0.2, 0.3]},
        "Economics": {5: [1.0, 1.2], 1: [0.8, 1.2], 0: [0.5, 1.0, 2.0]},
    }
    return pd.DataFrame(
        [
            {"delay": delay, "Parameter": field, "value": value}
            for field, delay_samples in samples.items()
            for delay, values in delay_samples.items()
            for value in values
        ]
    )


@pytest.mark.parametrize(
    "kwargs, filename",
    [
        pytest.param({}, "probabilityVsDelay.pdf", id="default-output"),
        pytest.param({"save_path": "nested/delay.pdf"}, "nested/delay.pdf", id="relative-output"),
    ],
)
def test_plot_sweep_probabilities_colors_order_and_all_lags(
    sweep_samples: pd.DataFrame,
    plot_environment: Mock,
    tmp_path: Path,
    kwargs: dict[str, str],
    filename: str,
) -> None:
    original_samples = sweep_samples.copy(deep=True)

    fig, ax = NobelGenderBiasAnalysis().plot_delay_sweep(sweep_samples, **kwargs)

    assert fig.axes == [ax]
    fields = ["Economics", "Physics", "PhysMed", "Chemistry"]
    assert [line.get_label() for line in ax.lines] == fields
    expected_probabilities = [
        [2 / 3, 0.5, 1.0],
        [0.0, 1.0, 0.25],
        [1.0, 1 / 3, 0.5],
        [1.0, 2 / 3, 0.0],
    ]
    for line, probabilities, color in zip(
        ax.lines, expected_probabilities, analysis_module.PUBLICATION_COLORS[:4], strict=True
    ):
        np.testing.assert_array_equal(line.get_xdata(), [0, 1, 5])
        np.testing.assert_allclose(line.get_ydata(), probabilities)
        assert to_rgba(line.get_color()) == to_rgba(color)
        assert line.get_marker() not in (None, "None", "none", "", " ")
    assert ax.get_legend() is not None
    assert [text.get_text() for text in ax.get_legend().get_texts()] == fields
    plot_environment.assert_called_once_with(fig, filename)
    assert (tmp_path / "figs" / filename).is_file()
    pd.testing.assert_frame_equal(sweep_samples, original_samples)


@pytest.mark.parametrize("one_field", [False, True], ids=["unknown-field-appended", "one-field"])
def test_plot_sweep_supports_unknown_fields_and_single_field(
    sweep_samples: pd.DataFrame, plot_environment: Mock, tmp_path: Path, one_field: bool
) -> None:
    unknown_samples = pd.DataFrame(
        {"delay": [5, 1, 1, 0], "Parameter": "Literature", "value": [1.2, 0.5, 1.0, 0.9]}
    )
    if one_field:
        results = unknown_samples
        expected_fields = ["Literature"]
    else:
        results = pd.concat([unknown_samples, sweep_samples], ignore_index=True)
        expected_fields = ["Economics", "Physics", "PhysMed", "Chemistry", "Literature"]

    fig, ax = NobelGenderBiasAnalysis().plot_delay_sweep(results, save_path="extra-fields.pdf")

    assert fig.axes == [ax]
    assert [line.get_label() for line in ax.lines] == expected_fields
    np.testing.assert_array_equal(ax.lines[-1].get_xdata(), [0, 1, 5])
    np.testing.assert_allclose(ax.lines[-1].get_ydata(), [0.0, 0.5, 1.0])
    assert ax.lines[-1].get_marker() not in (None, "None", "none", "", " ")
    if not one_field:
        for line, color in zip(ax.lines[:4], analysis_module.PUBLICATION_COLORS[:4], strict=True):
            assert to_rgba(line.get_color()) == to_rgba(color)
    plot_environment.assert_called_once_with(fig, "extra-fields.pdf")
    assert (tmp_path / "figs" / "extra-fields.pdf").is_file()
