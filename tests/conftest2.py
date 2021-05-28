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
def hand_candidates():
    hand_file = Path(__file__).parent / 'data' / 'hand_candidates.npz'
    hand_data = np.load(hand_file)
    return hand_data['hand_candidates']
