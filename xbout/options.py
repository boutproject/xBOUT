from collections import UserDict
from pathlib import Path


# These are imported to be used by 'eval' in
# BoutOptions.evaluate_scalar() and BoutOptionsFile.evaluate().
# Change the names to match those used by C++/BOUT++
from numpy import (pi, sin, cos, tan, arccos as acos, arcsin as asin,
                   arctan as atan, arctan2 as atan2, sinh, cosh, tanh,
                   arcsinh as asinh, arccosh as acosh, arctanh as atanh,
                   exp, log, log10, power as pow, sqrt, ceil, floor,
                   round, abs)

# TODO ability to read from/write to nested dictionary

SECTION_DELIM = ':'
COMMENT_DELIM = '#'


class Section(UserDict):
    """
    Section of an options file.

    This section can have multiple sub-sections, and each section stores the
    options as multiple key-value pairs.

    Parameters
    ----------
    name : str
        Name of the section
    data : dict, optional
    parent : BoutOptionsParser or Section
        A parent Section or BoutOptionsParser object
    """

    # Inheriting from UserDict gives keys, items, values, iter, del, & len
    # methods already, and the data attribute for storing the contents

    def __init__(self, name, data=None, parent=None):
        if data is None: data = {}
        super().__init__(data)

        # TODO should name be optional?
        self.name = name
        # TODO check if parent has a section with the same name?
        self._parent = parent

    def __getitem__(self, key):
        return self.get(key, evaluate=True, keep_comments=False)

    def get(self, key, evaluate=True, keep_comments=False):
        """
        Fetch the value stored under a certain key.

        Parameters
        ----------
        key : str
        evaluate : bool, optional (default: True)
            If true, attempts to evaluate the value as an expression by
            substituting in other values from the options file. Other values
            are specified by key, or by section:key.
            If false, will return value as an unmodified string.
        keep_comments : bool, optional (default: False)
            If false, will strip off any inline comments, delimited by '#', and
            any whitespace before returning.
            If true, will return the whole line.
        """

        if evaluate and keep_comments:
            # TODO relax this?
            raise ValueError("Cannot keep comments and evaluate the contents.")

        line = self.data[key]
        if keep_comments or isinstance(line, Section):
            return line
        else:
            # any comment will be discarded
            value, comment, = line.split(COMMENT_DELIM)
            # TODO remove whitespace?

            if evaluate:
                raise NotImplementedError
            else:
                return value

    def __setitem__(self, key, value):
        self.set(key, value)

    def set(self, key, value):
        """
        Store a value under a certain key.

        Will cast the value to a string (unless value is a new section).
        If a dictionary is passed, will create a new Section.

        Parameters
        ----------
        key : str
        value : str
        """

        if isinstance(value, dict):
            value = Section(name=key, data=value)
        if not isinstance(value, Section):
            value = str(value)
        self.data[key] = value

    def __str__(self):
        # TODO this needs to know the overall depth

        namestr = f"[{self.name}]\n"
        indent = "|-- "
        datastr = indent.join(indent + f"{key} = {val}\n"
                          for key, val in self.items())
        return namestr + datastr

    def _write(self, file, evaluate=False, keep_comments=True):
        """
        Writes out a single section to an open file object as
        [...:parent:section]
        key = value     # comment
        ...

        If it encounters a nested section it will recursively call this method
        on that nested section.
        """

        section_header = f"[{self.path()}]\n"
        file.write(section_header)

        for key in self.keys():
            val = self.get(key, evaluate, keep_comments)

            if isinstance(val, Section):
                # Recursively write sections
                val._write(file, evaluate, keep_comments)
            else:
                line = f"{key} = {val}\n"
                file.write(line)

    # TODO .sections(), repr?



class OptionsTree(Section):
    """
    Tree of options, with the same structure as a complete BOUT++ input file.

    This class represents a tree structure. Each section (Section object) can
    have multiple sub-sections, and each section stores the options as multiple
    key-value pairs.

    Examples
    --------

    >>> opts = OptionsTree()  # Create a root

    Specify value of a key in a section "test"

    >>> opts["test"]["key"] = 4

    Get the value of a key in a section "test"
    If the section does not exist then a KeyError is raised

    >>> print(opts["test"]["key"])
    4

    To pretty print the options

    >>> print(opts)
    root
     |- test
     |   |- key = 4

    """

    def __init__(self, data=None):
        super().__init__(name='root', data=data)

    # TODO .sections(), .as_dict(), str,

    def write_to(self, file, evaluate=False, keep_comments=True):
        """
        Writes out contents to a file, following the format of a BOUT.inp file.

        Parameters
        ----------
        file
        evaluate : bool, optional (default: True)
            If true, attempts to evaluate the value as an expression by
            substituting in other values from the options file, before writing.
            Other values are specified by key, or by section:key.
            If false, will write value as an unmodified string.
        keep_comments : bool, optional (default: False)
            If false, will strip off any inline comments, delimited by '#', and
            any whitespace before writing.
            If true, will write out the whole line.
        """

        with open(Path(file), 'w') as f:
            self._write(file=f)
            # TODO strip first section header


class OptionsFile(OptionsTree):
    """
    The full contents of a particular BOUT++ input file.

    This class represents a tree structure of options, all loaded from `file`.
    Each section (Section object) can have multiple sub-sections, and each
    section stores the options as multiple key-value pairs.

    Parameters
    ----------
    file : str or path-like object
        Path to file from which to read input options.
    lower : bool, optional (default: False)
        If true, converts all strings to lowercase on reading.
    """

    # TODO gridfile stuff?

    def __init__(self, file, lower=False):
        contents, self.file = self._read(Path(file), lower)
        super().__init__(data=contents)

    def _read(self, filepath, lower):
        with open(filepath, 'r') as f:
            # TODO add in first section header?
            #for line in f:
            #    if is_section(line):
            #        name
            #        self.data[name] = Section(name, data)
            #    elif is_comment(line):
            #        continue
            #    else:
            data = None
        return data, str(filepath.resolve())

    def write(self, evaluate=False, keep_comments=True):
        """
        Writes out contents to the file, following the BOUT.inp format.

        Parameters
        ----------
        evaluate : bool, optional (default: True)
            If true, attempts to evaluate the value as an expression by
            substituting in other values from the options file, before writing.
            Other values are specified by key, or by section:key.
            If false, will write value as an unmodified string.
        keep_comments : bool, optional (default: False)
            If false, will strip off any inline comments, delimited by '#', and
            any whitespace before writing.
            If true, will write out the whole line.
        """

        self.write_to(self.file, evaluate, keep_comments)

    def __repr__(self):
        # TODO Add grid-related options
        return f"BoutOptionsFile('{self.file}')"
