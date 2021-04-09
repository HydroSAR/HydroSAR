import pytest

def pytest_addoption(parser):
    parser.addoption('--integration', action='store_true', default=False, dest="integration",
                     help="enable integration tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--integration"):
        integration_skip = pytest.mark.skip(reason="Integration tests not requested; skipping.")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(integration_skip)


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
