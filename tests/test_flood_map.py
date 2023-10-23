import numpy as np
import pytest
from osgeo_utils.gdalcompare import find_diff

from osgeo import gdal

from asf_tools.hydrosar import flood_map


@pytest.mark.integration
def test_get_waterbody():
    water_raster = '/vsicurl/https://hyp3-testing.s3.us-west-2.amazonaws.com/asf-tools/' \
                   'S1A_IW_20230228T120437_DVR_RTC30/flood_map/watermap.tif'
    info = gdal.Info(water_raster, format='json')

    known_water_mask = flood_map.get_waterbody(info, threshold=30)

    test_mask = '/vsicurl/https://hyp3-testing.s3.us-west-2.amazonaws.com/asf-tools/S1A_IW_20230228T120437_DVR_RTC30/' \
                'flood_map/known_water_mask.tif'
    test_mask_array = gdal.Open(test_mask, gdal.GA_ReadOnly).ReadAsArray()

    assert np.all(known_water_mask == test_mask_array)


def test_logstat():
    arr = [10, 100, 1000, 10000, 100000]
    logstd = flood_map.logstat(arr)

    assert np.isclose(logstd, 25.95455351947008)


"""
@pytest.mark.integration
def test_estimate_flood_depths_iterative(flood_window, hand_window):
    water_height = flood_map.estimate_flood_depth(1, hand_window, flood_window, estimator='iterative',
                                                  water_level_sigma=3,
                                                  iterative_bounds=(0, 25))
    # FIXME: Basin-hopping appears to be non-deterministic. Return values vary *wildly*.
    # assert np.isclose(water_height, 7.394713346252969)
"""


@pytest.mark.integration
def test_estimate_flood_depths_logstat(flood_window, hand_window):
    water_height = flood_map.estimate_flood_depth(1, hand_window, flood_window, estimator='logstat',
                                                  water_level_sigma=3,
                                                  iterative_bounds=(0, 15))
    assert water_height == 21.02364492416382


@pytest.mark.integration
def test_estimate_flood_depths_nmad(flood_window, hand_window):
    water_height = flood_map.estimate_flood_depth(1, hand_window, flood_window, estimator='nmad', water_level_sigma=3,
                                                  iterative_bounds=(0, 15))

    assert water_height == 7.887911175434299


@pytest.mark.integration
def test_estimate_flood_depths_numpy(flood_window, hand_window):
    water_height = flood_map.estimate_flood_depth(1, hand_window, flood_window, estimator='numpy', water_level_sigma=3,
                                                  iterative_bounds=(0, 15))
    assert water_height == 16.353520154953003


@pytest.mark.integration
def test_make_flood_map(tmp_path):
    water_raster = '/vsicurl/https://hyp3-testing.s3.us-west-2.amazonaws.com/asf-tools/' \
                    'S1A_IW_20230228T120437_DVR_RTC30/flood_map/watermap.tif'
    vv_raster = '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/asf-tools/' \
                'S1A_IW_20230228T120437_DVR_RTC30/flood_map/RTC_VV.tif'
    hand_raster = '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/asf-tools/' \
                  'S1A_IW_20230228T120437_DVR_RTC30/flood_map/watermap_HAND.tif'

    out_flood_map = tmp_path / 'flood_map.tif'
    flood_map.make_flood_map(out_flood_map, vv_raster, water_raster, hand_raster, estimator='nmad')
    out_flood_map = out_flood_map.parent / f'{out_flood_map.stem}_nmad_FloodDepth.tif'

    assert out_flood_map.exists()

    golden_flood_map = '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/' \
                       'asf-tools/S1A_IW_20230228T120437_DVR_RTC30/' \
                       'flood_map/flood_map_nmad.tif'

    diffs = find_diff(golden_flood_map, str(out_flood_map))
    assert diffs == 0
