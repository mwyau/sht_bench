from sht_bench.backends import SUPPORTED_GRIDS, nalm
from sht_bench.cli import measure
from sht_bench.grids import grid_shape


def test_gl_grid_shape():
    assert grid_shape(42, "gl") == (64, 128)
    assert grid_shape(85, "gl") == (128, 256)
    assert grid_shape(319, "gl") == (320, 640)
    assert grid_shape(1279, "gl") == (1280, 2560)
    assert grid_shape(31, "gl") == (32, 63)


def test_cc_grid_shape():
    assert grid_shape(31, "cc") == (63, 124)
    assert grid_shape(36, "cc") == (73, 144)
    assert grid_shape(90, "cc") == (181, 360)
    assert grid_shape(360, "cc") == (721, 1440)


def test_supported_grids():
    assert SUPPORTED_GRIDS["ducc"] == ("gl", "cc")
    assert SUPPORTED_GRIDS["pyshtools"] == ("gl",)


def test_nalm():
    assert nalm(31) == 528


def test_measure_returns_positive_samples():
    iterations, samples = measure(lambda: sum(range(100)), warmup=1, repeat=3, min_time=0.001)
    assert iterations >= 1
    assert len(samples) == 3
    assert all(sample > 0 for sample in samples)
