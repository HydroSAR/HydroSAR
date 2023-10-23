import numpy as np
import pytest
from osgeo_utils.gdalcompare import find_diff

from asf_tools.hydrosar import water_map
from asf_tools.raster import read_as_array
from asf_tools.tile import tile_array


def test_determine_em_threshold(raster_tiles):
    scaling = 8.732284197109262
    threshold = water_map.determine_em_threshold(raster_tiles, scaling)
    assert np.isclose(threshold, 27.482176801248677)


@pytest.mark.integration
def test_select_hand_tiles(hand_candidates):
    hand_geotif = '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/asf-tools/water-map/20200603_HAND.tif'
    hand_array = read_as_array(str(hand_geotif))
    hand_tiles = np.ma.masked_invalid(tile_array(hand_array, tile_shape=(100, 100), pad_value=np.nan))

    selected_tiles = water_map.select_hand_tiles(hand_tiles, 15., 0.8)
    assert np.all(selected_tiles == hand_candidates)

    with pytest.raises(ValueError):
        _ = water_map.select_hand_tiles(np.zeros(shape=(10, 10, 10), dtype=float), 15., 0.8)


@pytest.mark.integration
def test_select_backscatter_tiles(hand_candidates):
    backscatter_geotif = '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/asf-tools/water-map/20200603_VH.tif'
    backscatter_array = np.ma.masked_invalid(read_as_array(backscatter_geotif))
    backscatter_tiles = np.ma.masked_less_equal(tile_array(backscatter_array, tile_shape=(100, 100), pad_value=0.), 0.)

    tile_indexes = water_map.select_backscatter_tiles(backscatter_tiles, hand_candidates)
    assert np.all(tile_indexes == np.array([771, 1974, 2397, 1205, 2577]))


@pytest.mark.integration
def test_make_water_map(tmp_path):
    vv_geotif = '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/asf-tools/' \
                'S1A_IW_20230228T120437_DVR_RTC30/water_map/RTC_VV.tif'
    vh_geotif = '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/asf-tools/' \
                'S1A_IW_20230228T120437_DVR_RTC30/water_map/RTC_VH.tif'
    hand_geotif = '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/asf-tools/' \
                  'S1A_IW_20230228T120437_DVR_RTC30/water_map/HAND.tif'

    out_water_map = tmp_path / 'water_map.tif'
    water_map.make_water_map(out_water_map, vv_geotif, vh_geotif, hand_geotif)

    assert out_water_map.exists()

    golden_water_map = '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/asf-tools/' \
                       'S1A_IW_20230228T120437_DVR_RTC30/water_map/fuzzy_water_map.tif'
    diffs = find_diff(golden_water_map, str(out_water_map))
    assert diffs == 0
