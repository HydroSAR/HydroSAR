import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests as integration"
    )


def pytest_addoption(parser):
    parser.addoption('--integration', action='store_true', default=False, dest='integration',
                     help='enable integration tests')


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--integration'):
        integration_skip = pytest.mark.skip(reason='Integration tests not requested; skipping.')
        for item in items:
            if 'integration' in item.keywords:
                item.add_marker(integration_skip)
