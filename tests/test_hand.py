import json

import pytest
from osgeo import gdal, ogr
from osgeo_utils.gdalcompare import find_diff
import numpy as np

from asf_tools import vector
from asf_tools.hydrosar import hand

HAND_BASINS = '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/' \
              'asf-tools/S1A_IW_20230228T120437_DVR_RTC30/hand/hybas_af_lev12_v1c_firstpoly.geojson'
GOLDEN_HAND = '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/' \
              'asf-tools/S1A_IW_20230228T120437_DVR_RTC30/hand/hybas_af_lev12_v1c_firstpoly.tif'

gdal.UseExceptions()


def nodata_equal_nan(golden_hand, out_hand):
    ds_golden = gdal.Open(str(golden_hand))
    ds_out = gdal.Open(str(out_hand))
    nodata_golden = ds_golden.GetRasterBand(1).GetNoDataValue()
    nodata_out = ds_out.GetRasterBand(1).GetNoDataValue()
    if nodata_golden and nodata_out:
        return np.isnan(nodata_golden) and np.isnan(nodata_out)
    else:
        return False


@pytest.mark.integration
def test_make_copernicus_hand(tmp_path):

    out_hand = tmp_path / 'hand.tif'
    hand.make_copernicus_hand(out_hand, HAND_BASINS)

    assert out_hand.exists()

    diffs = find_diff(str(GOLDEN_HAND), str(out_hand))
    assert diffs == 0


def test_prepare_hand_vrt_no_coverage():
    geojson = {
        'type': 'Point',
        'coordinates': [0, 0],
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))
    with pytest.raises(ValueError):
        hand.prepare_hand_vrt('foo', geometry)


def test_prepare_hand_vrt(tmp_path):
    hand_vrt = tmp_path / 'hand.tif'
    geojson = {
        'type': 'Polygon',
        'coordinates': [[
            [0.4, 10.16],
            [0.4, 10.86],
            [0.6, 10.86],
            [0.6, 10.16],
            [0.4, 10.16],
        ]],
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))

    hand.prepare_hand_vrt(str(hand_vrt), geometry)
    assert hand_vrt.exists()

    info = gdal.Info(str(hand_vrt), format='json')
    assert info['geoTransform'] == \
           [-0.0001388888888889, 0.0002777777777778, 0.0, 11.00013888888889, 0.0, -0.0002777777777778]
    assert info['size'] == [3600, 3600]


def test_prepare_hand_vrt_antimeridian():
    geojson = {
        'type': 'MultiPolygon',
        'coordinates': [
            [[
                [179.5, 51.4],
                [179.5, 51.6],
                [180.0, 51.6],
                [180.0, 51.4],
                [179.5, 51.4],
            ]],
            [[
                [-180.0, 51.4],
                [-180.0, 51.6],
                [-179.5, 51.6],
                [-179.5, 51.4],
                [-180.0, 51.4],
            ]],
        ],
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))

    with pytest.raises(ValueError):
        hand.prepare_hand_vrt('foo', geometry)


def test_intersects_hand_feature():
    features = vector.get_features(hand.prepare.HAND_GEOJSON)

    geojson = {
        'type': 'Point',
        'coordinates': [169, -45],
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))
    assert vector.get_property_values_for_intersecting_features(geometry, features)

    geojson = {
        'type': 'Point',
        'coordinates': [0, 0],
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))
    assert not vector.get_property_values_for_intersecting_features(geometry, features)


def test_get_features():
    assert len(vector.get_features(hand.prepare.HAND_GEOJSON)) == 26450
