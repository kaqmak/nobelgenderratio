"""Regression tests for the data passed to the Binomial model."""

from unittest.mock import Mock, call

import numpy as np
import pandas as pd
import pytest

from nobelgenderbias import NobelGenderBiasAnalysis


@pytest.fixture
def analysis(monkeypatch: pytest.MonkeyPatch) -> NobelGenderBiasAnalysis:
    instance = NobelGenderBiasAnalysis()
    instance.all_nobel_df = pd.DataFrame(
        {
            "Year": [1901, 1901, 1902, 1902],
            "prize": ["Physics", "Chemistry", "Physics", "Chemistry"],
            "nbl": [3.0, 1.0, 0.0, 2.0],
            "nbl_f": [2.0, 0.0, 0.0, 1.0],
        }
    )
    instance.ratio_df = instance.all_nobel_df[["Year", "prize"]].assign(
        position="senior", ratio=0.25
    )
    extrapolated = instance.ratio_df.rename(columns={"ratio": "ratioEx"}).copy()
    extrapolated["ratioEx"] = np.nan
    monkeypatch.setattr(instance, "extrapolate_ratio", lambda delay: extrapolated)
    return instance


def test_multiple_female_laureates_are_valid(
    analysis: NobelGenderBiasAnalysis, capsys: pytest.CaptureFixture[str]
) -> None:
    r_matrix, y_matrix, total_matrix, prize_names = analysis.prepare_stan_data(delay=0)

    assert prize_names == ["Chemistry", "Physics"]
    np.testing.assert_array_equal(r_matrix, np.full((2, 2), 0.25))
    np.testing.assert_array_equal(y_matrix, [[0, 2], [1, 0]])
    np.testing.assert_array_equal(total_matrix, [[1, 3], [2, 0]])
    assert np.issubdtype(y_matrix.dtype, np.integer)
    assert np.issubdtype(total_matrix.dtype, np.integer)
    output = capsys.readouterr().out
    assert "Bernoulli" not in output
    assert "WARNING" not in output


@pytest.mark.parametrize("ratio", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_invalid_gender_ratios_are_rejected(
    analysis: NobelGenderBiasAnalysis, ratio: float
) -> None:
    analysis.ratio_df.loc[0, "ratio"] = ratio

    with pytest.raises(ValueError, match="Gender ratios must be finite and between 0 and 1"):
        analysis.prepare_stan_data(delay=0)


def test_boundary_gender_ratios_are_valid(analysis: NobelGenderBiasAnalysis) -> None:
    analysis.ratio_df["ratio"] = [0.0, 1.0, 0.0, 1.0]

    r_matrix, _, _, _ = analysis.prepare_stan_data(delay=0)

    np.testing.assert_array_equal(r_matrix, [[1.0, 0.0], [1.0, 0.0]])


def test_missing_observed_ratio_uses_valid_extrapolation(
    analysis: NobelGenderBiasAnalysis, monkeypatch: pytest.MonkeyPatch
) -> None:
    extrapolated = analysis.ratio_df.rename(columns={"ratio": "ratioEx"}).copy()
    extrapolated["ratioEx"] = 0.5
    monkeypatch.setattr(analysis, "extrapolate_ratio", lambda delay: extrapolated)
    analysis.ratio_df.loc[0, "ratio"] = np.nan

    r_matrix, _, _, _ = analysis.prepare_stan_data(delay=0)

    assert r_matrix[0, 1] == 0.5


@pytest.mark.parametrize("column, name", [("nbl_f", "Female"), ("nbl", "Total")])
@pytest.mark.parametrize("count", [-1.0, 1.5, np.nan, np.inf, -np.inf])
def test_invalid_laureate_counts_are_rejected(
    analysis: NobelGenderBiasAnalysis, column: str, name: str, count: float
) -> None:
    analysis.all_nobel_df.loc[0, column] = count

    with pytest.raises(
        ValueError, match=f"{name} laureate counts must be finite, nonnegative integers"
    ):
        analysis.prepare_stan_data(delay=0)


def test_female_counts_cannot_exceed_total_counts(analysis: NobelGenderBiasAnalysis) -> None:
    analysis.all_nobel_df.loc[0, "nbl_f"] = 4.0

    with pytest.raises(ValueError, match="Female laureate counts cannot exceed total"):
        analysis.prepare_stan_data(delay=0)


def test_unknown_position_is_rejected(analysis: NobelGenderBiasAnalysis) -> None:
    with pytest.raises(ValueError, match="No data available for position 'unknown'"):
        analysis.prepare_stan_data(position="unknown", delay=0)


def test_loading_does_not_truncate_fractional_female_counts(
    analysis: NobelGenderBiasAnalysis,
) -> None:
    female_counts = pd.DataFrame({"Year": [1901, 1902], "Physics_female": [1.5, 0.0]})
    nobel_counts = analysis.all_nobel_df.drop(columns="nbl_f")
    analysis.all_nobel_df = analysis._merge_nobel_data(nobel_counts, female_counts)

    assert analysis.all_nobel_df.loc[0, "nbl_f"] == 1.5
    with pytest.raises(ValueError, match="Female laureate counts must be finite"):
        analysis.prepare_stan_data(delay=0)


@pytest.mark.parametrize(
    "delay, expected_ratios",
    [
        pytest.param(0, [[0.51, 0.61], [0.52, 0.62], [0.8, 0.9]], id="same-year"),
        pytest.param(1, [[0.5, 0.6], [0.51, 0.61], [0.52, 0.62]], id="odd-delay"),
        pytest.param(10, [[0.3, 0.4], [0.31, 0.41], [0.8, 0.9]], id="ten-years"),
        pytest.param(20, [[0.1, 0.2], [0.11, 0.21], [0.8, 0.9]], id="twenty-years"),
        pytest.param(None, [[0.3, 0.4], [0.31, 0.41], [0.8, 0.9]], id="default-ten-years"),
    ],
)
def test_delay_shifts_observed_ratios_without_changing_counts_or_source_data(
    monkeypatch: pytest.MonkeyPatch, delay: int | None, expected_ratios: list[list[float]]
) -> None:
    analysis = NobelGenderBiasAnalysis()
    analysis.all_nobel_df = pd.DataFrame(
        {
            "Year": [2000, 2000, 2001, 2001, 2002, 2002],
            "prize": ["Physics", "Chemistry"] * 3,
            "nbl": [3, 1, 0, 2, 1, 2],
            "nbl_f": [2, 0, 0, 1, 1, 1],
        }
    )
    analysis.ratio_df = pd.DataFrame(
        {
            "Year": np.repeat([1980, 1981, 1990, 1991, 1999, 2000, 2001], 2),
            "prize": ["Chemistry", "Physics"] * 7,
            "position": "senior",
            "ratio": [0.1, 0.2, 0.11, 0.21, 0.3, 0.4, 0.31, 0.41, 0.5, 0.6, 0.51, 0.61, 0.52, 0.62],
        }
    )
    original_ratios = analysis.ratio_df.copy(deep=True)
    original_counts = analysis.all_nobel_df.copy(deep=True)
    extrapolated = analysis.all_nobel_df[["Year", "prize"]].assign(
        position="senior", ratioEx=[0.9, 0.8] * 3
    )
    extrapolate = Mock(return_value=extrapolated)
    monkeypatch.setattr(analysis, "extrapolate_ratio", extrapolate)

    kwargs = {} if delay is None else {"delay": delay}
    # Repeating the call also catches shifts that accumulate on the source dataframe.
    for _ in range(2):
        r_matrix, y_matrix, total_matrix, prize_names = analysis.prepare_stan_data(**kwargs)

        assert prize_names == ["Chemistry", "Physics"]
        np.testing.assert_allclose(r_matrix, expected_ratios)
        np.testing.assert_array_equal(y_matrix, [[0, 2], [1, 0], [1, 1]])
        np.testing.assert_array_equal(total_matrix, [[1, 3], [2, 0], [2, 1]])
        pd.testing.assert_frame_equal(analysis.ratio_df, original_ratios)
        pd.testing.assert_frame_equal(analysis.all_nobel_df, original_counts)

    actual_delay = 10 if delay is None else delay
    assert extrapolate.call_args_list == [call(delay=actual_delay)] * 2


def test_repository_data_satisfies_binomial_constraints() -> None:
    analysis = NobelGenderBiasAnalysis()
    analysis.load_data()

    r_matrix, y_matrix, total_matrix, _ = analysis.prepare_stan_data()

    assert np.all(np.isfinite(r_matrix))
    assert np.all((r_matrix >= 0) & (r_matrix <= 1))
    assert np.all(y_matrix >= 0)
    assert np.all(total_matrix >= 0)
    assert np.all(y_matrix <= total_matrix)
