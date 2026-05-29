import numpy as np

from compare_controls import metric_intrinsic_dimension


def test_intrinsic_dimension_caps_components_by_sample_count():
    rng = np.random.default_rng(123)
    obs = rng.normal(size=(30, 2102))

    dim = metric_intrinsic_dimension(obs, n_components=50)

    assert 1 <= dim <= 30


def test_intrinsic_dimension_handles_empty_or_bad_inputs():
    assert metric_intrinsic_dimension(np.empty((0, 12))) == 0.0
    assert metric_intrinsic_dimension(np.empty((12, 0))) == 0.0
    assert metric_intrinsic_dimension(np.array([1.0, 2.0, 3.0])) == 0.0
