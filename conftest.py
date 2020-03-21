import pytest

# Add command line option '--long' for pytest, to be used to enable long tests
def pytest_addoption(parser):
    parser.addoption("--long", action="store_true", default=False,
                     help="enable tests marked as 'long'")

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--long"):
        # --long not given in cli: skip long tests
        print("\n    skipping long tests, pass '--long' to enable")
        skip_long = pytest.mark.skip(reason="need --long option to run")
        for item in items:
            if "long" in item.keywords:
                item.add_marker(skip_long)
