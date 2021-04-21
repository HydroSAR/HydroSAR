from pathlib import Path

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption('--integration', action='store_true', default=False, dest='integration',
                     help='enable integration tests')


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--integration'):
        integration_skip = pytest.mark.skip(reason='Integration tests not requested; skipping.')
        for item in items:
            if 'integration' in item.keywords:
                item.add_marker(integration_skip)


@pytest.fixture(scope='session')
def raster_tiles():
    tiles_file = Path(__file__).parent / 'data' / 'em_tiles.npz'
    tile_data = np.load(tiles_file)
    tiles = np.ma.MaskedArray(tile_data['tiles'], mask=tile_data['mask'])
    return np.log10(tiles) + 30


@pytest.fixture(scope='session')
def thresholds():
    thresholds_file = Path(__file__).parent / 'data' / 'em_thresholds.npz'
    thresholds_data = np.load(thresholds_file)
    return thresholds_data['thresholds']


@pytest.fixture(scope='session')
def rtc_raster_pair():
    primary = '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/' \
              'asf-tools/water-map/ki-threshold-pototype-scene.tif'
    secondary = '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/' \
                'asf-tools/water-map/ki-threshold-pototype-scene.tif'
    return primary, secondary


@pytest.fixture(scope='session')
def golden_water_map():
    return '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/' \
           'asf-tools/water-map/ki-threshold-initial-water-map.tif'


@pytest.fixture(scope='session')
def golden_hand():
    return '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/' \
           'asf-tools/hand/hybas_af_lev12_v1c_firstpoly.tif'


@pytest.fixture(scope='session')
def hand_basin():
    return '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/' \
           'asf-tools/hand/hybas_af_lev12_v1c_firstpoly.geojson'
