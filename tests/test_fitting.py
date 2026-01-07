"""Tests for grb_common.fitting module."""

import numpy as np
import pytest


class TestPriors:
    """Tests for prior distributions."""

    def test_uniform_prior_sample(self):
        """Test uniform prior sampling."""
        from grb_common.fitting import UniformPrior

        prior = UniformPrior(0, 10)
        samples = prior.sample(1000, rng=np.random.default_rng(42))

        assert len(samples) == 1000
        assert np.all(samples >= 0)
        assert np.all(samples <= 10)

    def test_uniform_prior_log_prob(self):
        """Test uniform prior log probability."""
        from grb_common.fitting import UniformPrior

        prior = UniformPrior(0, 10)

        # Inside bounds: log(1/10) = -log(10)
        assert np.isclose(prior.log_prob(5), -np.log(10))

        # Outside bounds: -inf
        assert prior.log_prob(-1) == -np.inf
        assert prior.log_prob(11) == -np.inf

    def test_uniform_prior_ppf(self):
        """Test uniform prior percent point function."""
        from grb_common.fitting import UniformPrior

        prior = UniformPrior(0, 10)

        assert np.isclose(prior.ppf(0), 0)
        assert np.isclose(prior.ppf(0.5), 5)
        assert np.isclose(prior.ppf(1), 10)

    def test_log_uniform_prior_sample(self):
        """Test log-uniform prior sampling."""
        from grb_common.fitting import LogUniformPrior

        prior = LogUniformPrior(1e50, 1e54)
        samples = prior.sample(1000, rng=np.random.default_rng(42))

        assert len(samples) == 1000
        assert np.all(samples >= 1e50)
        assert np.all(samples <= 1e54)

    def test_log_uniform_prior_log_prob(self):
        """Test log-uniform prior log probability."""
        from grb_common.fitting import LogUniformPrior

        prior = LogUniformPrior(1, 100)

        # At x=10: log_prob = -log(10) - log(log(100))
        expected = -np.log(10) - np.log(np.log(100))
        assert np.isclose(prior.log_prob(10), expected)

        # Outside bounds
        assert prior.log_prob(0.5) == -np.inf
        assert prior.log_prob(200) == -np.inf

    def test_log_uniform_prior_ppf(self):
        """Test log-uniform prior ppf."""
        from grb_common.fitting import LogUniformPrior

        prior = LogUniformPrior(1, 100)

        assert np.isclose(prior.ppf(0), 1)
        assert np.isclose(prior.ppf(0.5), 10)  # geometric mean
        assert np.isclose(prior.ppf(1), 100)

    def test_gaussian_prior(self):
        """Test Gaussian prior."""
        from grb_common.fitting import GaussianPrior

        prior = GaussianPrior(mu=2.5, sigma=0.1)
        samples = prior.sample(1000, rng=np.random.default_rng(42))

        # Mean should be close to 2.5
        assert np.isclose(np.mean(samples), 2.5, atol=0.01)

        # Std should be close to 0.1
        assert np.isclose(np.std(samples), 0.1, atol=0.01)

    def test_delta_prior(self):
        """Test delta (fixed) prior."""
        from grb_common.fitting import DeltaPrior

        prior = DeltaPrior(3.14)

        samples = prior.sample(100)
        assert np.all(samples == 3.14)

        assert prior.log_prob(3.14) == 0.0
        assert prior.log_prob(3.0) == -np.inf

    def test_composite_prior(self):
        """Test composite prior."""
        from grb_common.fitting import CompositePrior, LogUniformPrior, UniformPrior

        priors = CompositePrior({
            'a': UniformPrior(0, 1),
            'b': LogUniformPrior(1, 100),
        })

        assert priors.n_params == 2
        assert priors.param_names == ['a', 'b']

        # Sample
        samples = priors.sample(100, rng=np.random.default_rng(42))
        assert samples.shape == (100, 2)

        # Log prob
        theta = np.array([0.5, 10])
        lp = priors.log_prob(theta)
        assert np.isfinite(lp)

        # Out of bounds
        theta_bad = np.array([2.0, 10])  # a out of bounds
        assert priors.log_prob(theta_bad) == -np.inf

        # Prior transform
        u = np.array([0.5, 0.5])
        theta = priors.prior_transform(u)
        assert np.isclose(theta[0], 0.5)
        assert np.isclose(theta[1], 10)


class TestLikelihood:
    """Tests for likelihood functions."""

    def test_chi_squared(self):
        """Test chi-squared calculation."""
        from grb_common.fitting import chi_squared

        observed = np.array([1.0, 2.0, 3.0])
        model = np.array([1.0, 2.0, 3.0])
        error = np.array([0.1, 0.1, 0.1])

        # Perfect fit: chi2 = 0
        chi2 = chi_squared(observed, model, error)
        assert chi2 == 0.0

        # With residuals
        model_off = np.array([1.1, 2.1, 3.1])
        chi2 = chi_squared(observed, model_off, error)
        expected = 3 * (0.1 / 0.1)**2  # 3 points, 1-sigma each
        assert np.isclose(chi2, expected)

    def test_gaussian_likelihood(self):
        """Test Gaussian likelihood."""
        from grb_common.fitting import gaussian_likelihood

        observed = np.array([1.0, 2.0])
        model = np.array([1.0, 2.0])
        error = np.array([0.1, 0.1])

        log_like = gaussian_likelihood(observed, model, error)

        # Perfect fit: should be maximum (only normalization terms)
        expected = -np.sum(np.log(error)) - 0.5 * len(observed) * np.log(2 * np.pi)
        assert np.isclose(log_like, expected)

    def test_chi_squared_upper_limits(self):
        """Test chi-squared with upper limits."""
        from grb_common.fitting import chi_squared_upper_limits

        observed = np.array([1.0, 2.0, 0.5])  # Last is upper limit
        model = np.array([0.9, 2.1, 0.3])  # Below limit
        error = np.array([0.1, 0.1, 0.1])
        upper_limits = np.array([False, False, True])

        chi2 = chi_squared_upper_limits(observed, model, error, upper_limits)

        # Model below limit: no penalty
        assert chi2 < 3  # Less than if all were detections

        # Model above limit: penalty
        model_bad = np.array([0.9, 2.1, 0.8])  # Exceeds limit
        chi2_bad = chi_squared_upper_limits(observed, model_bad, error, upper_limits)
        assert chi2_bad > chi2

    def test_poisson_likelihood(self):
        """Test Poisson likelihood."""
        from grb_common.fitting import poisson_likelihood

        counts = np.array([10, 20, 15])
        model = np.array([10, 20, 15])

        log_like = poisson_likelihood(counts, model)
        assert np.isfinite(log_like)

        # Better fit should have higher likelihood
        model_good = np.array([10, 20, 15])
        model_bad = np.array([5, 10, 8])

        assert poisson_likelihood(counts, model_good) > poisson_likelihood(counts, model_bad)


class TestSamplerResult:
    """Tests for SamplerResult."""

    def test_result_creation(self):
        """Test creating a result object."""
        from grb_common.fitting import SamplerResult

        samples = np.random.randn(1000, 3)
        log_like = -0.5 * np.sum(samples**2, axis=1)

        result = SamplerResult(
            samples=samples,
            log_likelihood=log_like,
            param_names=['a', 'b', 'c'],
        )

        assert result.n_samples == 1000
        assert result.n_params == 3

    def test_result_statistics(self):
        """Test result statistics."""
        from grb_common.fitting import SamplerResult

        # Create samples from known distribution
        rng = np.random.default_rng(42)
        samples = rng.normal(loc=[1, 2, 3], scale=[0.1, 0.2, 0.3], size=(10000, 3))
        log_like = -0.5 * np.sum(samples**2, axis=1)

        result = SamplerResult(
            samples=samples,
            log_likelihood=log_like,
            param_names=['a', 'b', 'c'],
        )

        # Check means
        assert np.isclose(result.mean('a'), 1.0, atol=0.02)
        assert np.isclose(result.mean('b'), 2.0, atol=0.02)
        assert np.isclose(result.mean('c'), 3.0, atol=0.02)

        # Check stds
        assert np.isclose(result.std('a'), 0.1, atol=0.02)
        assert np.isclose(result.std('b'), 0.2, atol=0.02)
        assert np.isclose(result.std('c'), 0.3, atol=0.02)

        # Check percentiles
        p16, p50, p84 = result.percentile('a', [16, 50, 84])
        assert np.isclose(p50, 1.0, atol=0.02)
        assert p16 < p50 < p84

    def test_result_save_load(self):
        """Test saving and loading results."""
        pytest.importorskip("h5py")
        import tempfile
        from pathlib import Path

        from grb_common.fitting import SamplerResult

        samples = np.random.randn(100, 2)
        log_like = np.random.randn(100)

        result = SamplerResult(
            samples=samples,
            log_likelihood=log_like,
            param_names=['x', 'y'],
            metadata={'sampler': 'test'},
        )

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = Path(f.name)

        try:
            result.save(filepath)
            loaded = SamplerResult.load(filepath)

            np.testing.assert_array_almost_equal(loaded.samples, result.samples)
            np.testing.assert_array_almost_equal(loaded.log_likelihood, result.log_likelihood)
            assert loaded.param_names == result.param_names
        finally:
            filepath.unlink()


class TestBackends:
    """Tests for sampler backends."""

    def test_available_backends(self):
        """Test backend availability detection."""
        from grb_common.fitting.backends import available_backends

        backends = available_backends()
        assert isinstance(backends, list)

    def test_get_sampler_error(self):
        """Test error for unknown sampler."""
        from grb_common.fitting.backends import get_sampler

        with pytest.raises(ValueError, match="Unknown sampler"):
            get_sampler("nonexistent")
