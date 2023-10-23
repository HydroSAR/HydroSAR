import numpy as np

from asf_tools.hydrosar.threshold import expectation_maximization_threshold


def test_determine_em_threshold(raster_tiles, thresholds):
    scaling = 8.732284197109262
    test_tiles = (np.around(raster_tiles * scaling)).astype(int)
    for tile, threshold in zip(test_tiles, thresholds):
        assert np.isclose(expectation_maximization_threshold(tile), threshold)
