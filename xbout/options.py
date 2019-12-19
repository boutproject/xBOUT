from collections import UserDict
from pathlib import Path


# These are imported to be used by 'evaluate=True' in Section.get().
# Change the names to match those used by C++/BOUT++
from numpy import (pi, sin, cos, tan, arccos as acos, arcsin as asin,
                   arctan as atan, arctan2 as atan2, sinh, cosh, tanh,
                   arcsinh as asinh, arccosh as acosh, arctanh as atanh,
                   exp, log, log10, power as pow, sqrt, ceil, floor,
                   round, abs)

# TODO file reader
# TODO substitution of keys within strings
# TODO Sanitise some of the weirder things BOUT's option parser does:
#  - Detect escaped arithmetic symbols (+-*/^), dot (.), or brackets ((){}[]) in option names
#  - Detect escaping through backquotes in option names (`2ndvalue`)
#  - Evaluate numletter as num*letter (and numbracket as num*bracket)
#  - Substitute unicode representation of pi
#  - Ensure option names don't contain ':' or '='
#  - Check if all expressions are rounded to nearest integer?!
# TODO ability to read from/write to nested dictionary


SECTION_DELIM = ':'
COMMENT_DELIM = ['#', ';']
INDENT_STR = '|-- '
BOOLEAN_STATES = {'true': True, 'on': True,
                  'false': False, 'off': False}


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
    parent : str, Section or None
        The parent Section
    """

    # Inheriting from UserDict gives keys, items, values, iter, del, & len
    # methods already, and the data attribute for storing the contents

    def __init__(self, name, parent=None, data=None):
        if data is None: data = {}
        super().__init__(data)

        self.name = name
        # TODO check if parent has a section with the same name?
        self._parent = parent
        if isinstance(parent, Section):
            self.parent = parent.name
        else:
            self.parent = parent

    def __getitem__(self, key):
        return self.get(key, evaluate=False, keep_comments=False)

    def get(self, key, evaluate=False, keep_comments=False):
        """
        Fetch the value stored under a certain key.

        Parameters
        ----------
        key : str
        evaluate : bool, optional (default: False)
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
            # any comment will be discarded, along with any trailing whitespace
            for delim in COMMENT_DELIM:
                line, *comments = line.split(delim)
            value = line.rstrip()

            if evaluate:
                return self.evaluate(value)
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
            value = Section(name=key, parent=self, data=value)
        if not isinstance(value, Section):
            value = str(value)
        self.data[key] = value

    def lineage(self):
        """
        Returns the full route to this section by working back up the tree.

        Returns
        -------
        str
        """
        if self._parent is None or self._parent == 'root':
            return self.name
        else:
            return self._parent.lineage() + SECTION_DELIM + self.name

    def _find_sections(self, sections):
        """
        Recursively find a list of all the section objects below this one, and
        append them to the list passed before returning them all.
        """
        for key in self.keys():
            val = self[key]
            if isinstance(val, Section):
                sections.append(val)
                sections = val._find_sections(sections)
        return sections

    def sections(self):
        """
        Returns a list of all sections contained, including nested ones.

        Returns
        -------
        list of Section objects
        """
        return self._find_sections([self])

    def __str__(self):
        depth = self.lineage().count(SECTION_DELIM)
        text = INDENT_STR * depth + f"[{self.name}]\n"
        for key, val in self.items():
            if isinstance(val, Section):
                text += str(val)
            else:
                text += INDENT_STR * (depth+1) + f"{key} = {val}\n"
        return text

    def _write(self, file, evaluate=False, keep_comments=True):
        """
        Writes out a single section to an open file object as
        [...:parent:section]
        key = value     # comment
        ...

        If it encounters a nested section it will recursively call this method
        on that nested section.
        """

        if self.name is not None and self.name != 'root':
            section_header = f"[{self.lineage()}]\n"
            file.write(section_header)

        for key in self.keys():
            entry = self.get(key, evaluate, keep_comments)

            if isinstance(entry, Section):
                # Recursively write sections
                file.write("\n")
                entry._write(file, evaluate, keep_comments)
            else:
                file.write(f"{key} = {entry}\n")

    def __repr__(self):
        return f"Section(name='{self.name}', parent='{self.parent}', " \
               f"data={self.data})"

    # TODO could this be a class method, or static method?
    def evaluate(self, value, substitute=True):
        """
        Evaluates the string using eval, as it would when read in BOUT++.

        Parameters
        ----------
        value : str
        substitute : bool (default: True)

        """

        # TODO substitution of other keys
        #if substitute:
        #    value = self._substitute_keys_within(value)

        if value.lower() in BOOLEAN_STATES:
            # Treat booleans separately to cover lowercase 'true' etc.
            return BOOLEAN_STATES[value.lower()]
        else:
            if '^' in value:
                value = value.replace('^', '**')
            return eval(value)


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
        super().__init__(name='root', data=data, parent=None)

    # TODO .as_dict() ?

    def write_to(self, file, evaluate=False, keep_comments=True, lower=False):
        """
        Writes out contents to a file, following the format of a BOUT.inp file.

        Parameters
        ----------
        file
        evaluate : bool, optional (default: False)
            If true, attempts to evaluate the value as an expression by
            substituting in other values from the options file, before writing.
            Other values are specified by key, or by section:key.
            If false, will write value as an unmodified string.
        keep_comments : bool, optional (default: False)
            If false, will strip off any inline comments, delimited by '#', and
            any whitespace before writing.
            If true, will write out the whole line.

        Returns
        -------
        filepath : str
            Full path of output file written to
        """

        with open(Path(file), 'w') as f:
            self._write(file=f)
        return str(Path(file).resolve())

    def _read_from(self, filepath, lower):
        with open(filepath, 'r') as f:
            # TODO add in first section header?
            #for l in f:
            #    line = Line(l)
            #    if line.is_section():
            #        name
            #        self.data[name] = Section(name, parent, data)
            #    elif line.is_comment() or line.is_empty()
            #        continue
            #    else:
            data = None
        return data, str(filepath.resolve())


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
        contents, self.file = self._read_from(Path(file), lower)
        super().__init__(data=contents)

    def write(self, evaluate=False, keep_comments=True):
        """
        Writes out contents to the file, following the BOUT.inp format.

        Parameters
        ----------
        evaluate : bool, optional (default: False)
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
        return f"OptionsFile(file='{self.file}')"
