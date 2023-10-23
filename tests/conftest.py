from pathlib import Path

import numpy as np
import pytest


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


@pytest.fixture(scope='session')
def hand_window():
    hand_file = Path(__file__).parent / 'data' / 'hand_window.npz'
    hand_data = np.load(hand_file)
    return hand_data['hand_window']


@pytest.fixture(scope='session')
def flood_window():
    flood_file = Path(__file__).parent / 'data' / 'flood_window.npz'
    flood_data = np.load(flood_file)
    return flood_data['flood_window']
