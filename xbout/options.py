from configparser import ConfigParser
from pathlib import Path
from pprint import PrettyPrinter


class BoutOptionsParser(ConfigParser):
    def __init__(self):
        super().__init__(self,
                         delimiters=('=',),
                         comment_prefixes=('#',),
                         inline_comment_prefixes=('#',),
                         empty_lines_in_values=False)


class BoutOptions(BoutOptionsParser):
    def __init__(self, filepath):
        super().__init__()

        # Initialise options by reading file
        # Insert section header on 1st line to avoid MissingSectionHeaderError
        with open(filepath, 'r') as original:
            modified = "[top]\n" + original.read()
            self.read_string(modified)
        self.filepath = Path(filepath)

    def to_flat_dict(self):
        return {section: self.items(section) for section in self.sections()}

    def __repr__(self):
        return f"BoutOptions('{self.filepath}')"

    def __str__(self):
        # TODO overload PrettyPrinter.isrecursive() to print nested dicts
        contents = PrettyPrinter().pformat(self.to_flat_dict())
        return f"Bout options file at '{self.filepath}':\n" \
               f"{contents}"
