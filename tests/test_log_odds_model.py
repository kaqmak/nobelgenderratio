"""Regression tests for the noncentered hierarchical log-odds model."""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from scipy.special import expit, logit
from scipy.stats import norm, truncnorm

from nobelgenderbias import NobelGenderBiasAnalysis


def _prior_model(r_matrix: np.ndarray) -> pm.Model:
    return NobelGenderBiasAnalysis().build_pymc_model(
        r_matrix,
        np.zeros_like(r_matrix, dtype=int),
        np.ones_like(r_matrix, dtype=int),
        sample_priors=True,
    )


def _evaluate_hierarchy(
    model: pm.Model, mu: float, sigma: float, offsets: np.ndarray
) -> list[np.ndarray]:
    evaluate = pytensor.function(
        [model["mu"], model["sigma"], model["log_alpha_offset"]],
        [model["log_alpha"], model["alpha"], model["theta"]],
    )
    return evaluate(mu, sigma, offsets)


def _evaluate_theta_and_gradient(model: pm.Model, offsets: np.ndarray) -> list[np.ndarray]:
    evaluate = pytensor.function(
        [model["mu"], model["sigma"], model["log_alpha_offset"]],
        [model["theta"], pt.grad(model["theta"].sum(), model["log_alpha_offset"])],
    )
    return evaluate(0.0, 1.0, offsets)


@pytest.mark.parametrize(
    "r_matrix, odds_multipliers",
    [
        pytest.param(
            np.array([[0.2, 0.5, 0.8, 0.9], [0.1, 0.4, 0.7, 0.6], [0.3, 0.6, 0.2, 0.5]]),
            np.array([0.25, 0.5, 1.0, 3.0]),
            id="multiple-years-and-fields",
        ),
        pytest.param(np.array([[0.2, 0.5, 0.8]]), np.array([0.5, 1.0, 2.0]), id="one-year"),
        pytest.param(np.array([[0.1], [0.2], [0.5], [0.8]]), np.array([2.0]), id="one-field"),
    ],
)
def test_odds_multiplier_matches_analytic_mapping_and_broadcasts_by_field(
    r_matrix: np.ndarray, odds_multipliers: np.ndarray
) -> None:
    model = _prior_model(r_matrix)
    mu, sigma = -0.3, 0.7
    offsets = (np.log(odds_multipliers) - mu) / sigma

    log_alpha, alpha, theta = _evaluate_hierarchy(model, mu, sigma, offsets)

    expected = r_matrix * odds_multipliers / (1 - r_matrix + r_matrix * odds_multipliers)
    assert log_alpha.shape == alpha.shape == (r_matrix.shape[1],)
    assert theta.shape == r_matrix.shape
    np.testing.assert_allclose(log_alpha, np.log(odds_multipliers))
    np.testing.assert_allclose(alpha, odds_multipliers)
    np.testing.assert_allclose(theta, expected)


def test_half_odds_at_twenty_percent_gives_one_ninth() -> None:
    model = _prior_model(np.array([[0.2]]))

    _, alpha, theta = _evaluate_hierarchy(model, 0.0, 1.0, np.array([np.log(0.5)]))

    assert alpha[0] == pytest.approx(0.5)
    assert theta[0, 0] == pytest.approx(1 / 9)


def test_unit_odds_multiplier_is_neutral_including_near_and_exact_boundaries() -> None:
    r_matrix = np.array([[0.0, 1e-12, 0.2], [1.0 - 1e-12, 0.8, 1.0]])
    model = _prior_model(r_matrix)

    _, alpha, theta = _evaluate_hierarchy(model, 0.0, 0.7, np.zeros(3))

    np.testing.assert_array_equal(alpha, np.ones(3))
    np.testing.assert_allclose(theta, r_matrix, rtol=1e-12, atol=0)


@pytest.mark.parametrize("log_alpha", [-30.0, 30.0], ids=["small-odds", "large-odds"])
def test_large_log_odds_remain_smooth_without_probability_clipping(log_alpha: float) -> None:
    r_matrix = np.array([[0.2, 0.8], [0.4, 0.6]])
    model = _prior_model(r_matrix)
    offsets = np.full(2, log_alpha)

    theta, gradient = _evaluate_theta_and_gradient(model, offsets)

    expected = expit(logit(r_matrix) + log_alpha)
    np.testing.assert_allclose(theta, expected, rtol=1e-12, atol=0)
    np.testing.assert_allclose(gradient, (expected * (1 - expected)).sum(axis=0), rtol=1e-12)
    assert np.all(np.isfinite(gradient))
    assert np.all(gradient > 0)
    if log_alpha < 0:
        assert np.all((theta > 0) & (theta < 1e-8))
    else:
        assert np.all((theta > 1 - 1e-8) & (theta < 1))


@pytest.mark.parametrize(
    "offsets",
    [np.array([-1000.0, 1000.0]), np.array([1000.0, -1000.0])],
    ids=["opposite-extreme-odds", "reversed-extreme-odds"],
)
def test_endpoint_ratios_remain_exact_with_finite_zero_gradients(offsets: np.ndarray) -> None:
    r_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    model = _prior_model(r_matrix)

    theta, gradient = _evaluate_theta_and_gradient(model, offsets)

    np.testing.assert_array_equal(theta, r_matrix)
    np.testing.assert_array_equal(gradient, np.zeros(2))


@pytest.mark.parametrize("sample_priors", [False, True], ids=["posterior", "prior-only"])
@pytest.mark.parametrize(
    "ratio",
    [-0.1, 1.1, np.nan, np.inf, -np.inf],
    ids=["negative", "above-one", "nan", "inf", "-inf"],
)
def test_direct_builder_rejects_invalid_ratios(ratio: float, sample_priors: bool) -> None:
    r_matrix = np.array([[0.25, 0.75], [ratio, 0.5]])

    with pytest.raises(ValueError, match=r"(?i)ratios.*finite.*0.*1"):
        NobelGenderBiasAnalysis().build_pymc_model(
            r_matrix,
            np.zeros((2, 2), dtype=int),
            np.ones((2, 2), dtype=int),
            sample_priors=sample_priors,
        )


@pytest.mark.parametrize(
    "ratio, female_count, total_count",
    [
        pytest.param(0.0, 1, 3, id="positive-female-count-at-zero"),
        pytest.param(1.0, 0, 3, id="zero-female-count-at-one"),
        pytest.param(1.0, 2, 3, id="partial-female-count-at-one"),
    ],
)
def test_impossible_endpoint_observations_raise_clear_errors(
    ratio: float, female_count: int, total_count: int
) -> None:
    # Put the contradiction away from the first row and field to check the whole matrix.
    r_matrix = np.array([[0.25, 0.75], [0.5, ratio]])
    y_matrix = np.array([[0, 1], [1, female_count]])
    total_matrix = np.array([[2, 2], [2, total_count]])

    with pytest.raises(ValueError, match=r"(?i)(female|observ|count|impossible)") as error:
        NobelGenderBiasAnalysis().build_pymc_model(
            r_matrix, y_matrix, total_matrix, sample_priors=False
        )
    assert str(int(ratio)) in str(error.value)


@pytest.mark.parametrize(
    "r_matrix, y_matrix, total_matrix",
    [
        pytest.param(
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0, 3], [4, 0]]),
            np.array([[2, 3], [4, 5]]),
            id="consistent-endpoints",
        ),
        pytest.param(
            np.array([[0.0, 1.0, 0.2], [1.0, 0.0, 0.8]]),
            np.array([[0, 4, 1], [3, 0, 4]]),
            np.array([[4, 4, 5], [3, 3, 5]]),
            id="endpoints-and-interior",
        ),
        pytest.param(
            np.array([[0.0, 1.0, 0.25], [1.0, 0.0, 0.75]]),
            np.zeros((2, 3), dtype=int),
            np.zeros((2, 3), dtype=int),
            id="zero-trials-at-endpoints-and-interior",
        ),
    ],
)
def test_logp_and_gradients_are_finite_for_consistent_endpoints_and_zero_trials(
    r_matrix: np.ndarray, y_matrix: np.ndarray, total_matrix: np.ndarray
) -> None:
    model = NobelGenderBiasAnalysis().build_pymc_model(
        r_matrix, y_matrix, total_matrix, sample_priors=False
    )
    assert model.observed_RVs
    logp = model.compile_logp()
    gradient = model.compile_dlogp()
    likelihood_logp = model.compile_logp(vars=model.observed_RVs)

    for mu in (-2.0, 0.0, 2.0):
        point = model.initial_point()
        point[model.rvs_to_values[model["mu"]].name] = np.asarray(mu)
        point[model.rvs_to_values[model["log_alpha_offset"]].name] = np.linspace(
            -0.4, 0.6, r_matrix.shape[1]
        )
        assert np.isfinite(logp(point))
        assert np.all(np.isfinite(gradient(point)))
        if np.all((r_matrix == 0) | (r_matrix == 1)) or np.all(total_matrix == 0):
            assert likelihood_logp(point) == pytest.approx(0.0, abs=1e-12)


def test_hierarchy_has_normal_location_positive_scale_and_noncentered_offsets() -> None:
    model = _prior_model(np.full((2, 3), 0.25))

    assert {variable.name for variable in model.free_RVs} == {"mu", "sigma", "log_alpha_offset"}
    assert {variable.name for variable in model.deterministics} == {"log_alpha", "alpha", "theta"}
    assert not model.observed_RVs
    assert model.rvs_to_transforms[model["sigma"]] is not None
    assert model["log_alpha_offset"].eval().shape == (3,)

    locations = np.array([-2.0, 0.0, 2.0])
    np.testing.assert_allclose(pm.logp(model["mu"], locations).eval(), norm.logpdf(locations))
    np.testing.assert_allclose(
        pm.logp(model["log_alpha_offset"], locations).eval(), norm.logpdf(locations)
    )
    scales = np.array([-0.1, 0.25, 1.0, 1.75])
    scale_logp = pm.logp(model["sigma"], scales).eval()
    assert np.isneginf(scale_logp[0])
    np.testing.assert_allclose(
        scale_logp[1:], truncnorm.logpdf(scales[1:], -2, np.inf, loc=1, scale=0.5)
    )

    offsets = np.array([-2.0, 0.0, 1.5])
    for sigma in (0.25, 1.0, 2.0):
        log_alpha, alpha, _ = _evaluate_hierarchy(model, -0.4, sigma, offsets)
        np.testing.assert_allclose(log_alpha, -0.4 + sigma * offsets)
        np.testing.assert_allclose(alpha, np.exp(-0.4 + sigma * offsets))
        assert np.all(alpha > 0)


def test_actual_prior_sampling_keeps_alpha_theta_and_allows_dummy_endpoint_observations() -> None:
    r_matrix = np.array([[0.0, 0.2], [0.8, 1.0]])
    # Both endpoint observations are impossible, but are irrelevant without a likelihood.
    model = NobelGenderBiasAnalysis().build_pymc_model(
        r_matrix, np.array([[1, 0], [0, 2]]), np.full((2, 2), 3), sample_priors=True
    )
    assert not model.observed_RVs
    assert not any(name.startswith("y_") for name in model.named_vars)

    with model:
        trace = pm.sample_prior_predictive(draws=32, random_seed=42)

    prior = trace.prior
    assert {"mu", "sigma", "log_alpha_offset", "log_alpha", "alpha", "theta"} <= set(
        prior.data_vars
    )
    # PyMC 6 may include an empty predictive group even without a likelihood.
    assert "prior_predictive" not in trace or not trace["prior_predictive"].data_vars
    mu = prior["mu"].values
    sigma = prior["sigma"].values
    offsets = prior["log_alpha_offset"].values
    log_alpha = prior["log_alpha"].values
    alpha = prior["alpha"].values
    theta = prior["theta"].values
    assert mu.shape == sigma.shape == (1, 32)
    assert offsets.shape == log_alpha.shape == alpha.shape == (1, 32, 2)
    assert theta.shape == (1, 32, 2, 2)
    assert np.all(np.isfinite(sigma) & (sigma > 0))
    assert np.all(np.isfinite(alpha) & (alpha > 0))
    assert np.all(np.isfinite(theta) & (theta >= 0) & (theta <= 1))
    np.testing.assert_allclose(log_alpha, mu[..., None] + sigma[..., None] * offsets)
    np.testing.assert_allclose(alpha, np.exp(log_alpha))
    field_alpha = alpha[..., None, :]
    expected = r_matrix * field_alpha / (1 - r_matrix + r_matrix * field_alpha)
    np.testing.assert_allclose(theta, expected)
    np.testing.assert_array_equal(theta[..., 0, 0], np.zeros((1, 32)))
    np.testing.assert_array_equal(theta[..., 1, 1], np.ones((1, 32)))
