import pkgutil
import importlib

avail = []


class skeleton:
    name = "not set"

    def does_match(self, ds):
        return False

    def update(self, ds):
        return ds


for module in pkgutil.iter_modules(__path__):
    importlib.import_module(f"xbout.modules.{module.name}")
