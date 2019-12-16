from pathlib import Path
import os
import re

import numpy


# These are imported to be used by 'eval' in
# BoutOptions.evaluate_scalar() and BoutOptionsFile.evaluate().
# Change the names to match those used by C++/BOUT++
from numpy import (pi, sin, cos, tan, arccos as acos, arcsin as asin,
                   arctan as atan, arctan2 as atan2, sinh, cosh, tanh,
                   arcsinh as asinh, arccosh as acosh, arctanh as atanh,
                   exp, log, log10, power as pow, sqrt, ceil, floor,
                   round, abs)


# TODO rewrite using pathlib
# TODO make forced lowercase optional
# TODO ability to read from/write to nested dictionary

class BoutOptions:
    """This class represents a tree structure. Each node (BoutOptions
    object) can have several sub-nodes (sections), and several
    key-value pairs.

    Parameters
    ----------
    name : str, optional
        Name of the root section (default: "root")
    parent : BoutOptions, optional
        A parent BoutOptions object (default: None)

    Examples
    --------

    >>> optRoot = BoutOptions()  # Create a root

    Specify value of a key in a section "test"
    If the section does not exist then it is created

    >>> optRoot.getSection("test")["key"] = 4

    Get the value of a key in a section "test"
    If the section does not exist then a KeyError is raised

    >>> print(optRoot["test"]["key"])
    4

    To pretty print the options

    >>> print(optRoot)
    root
     |- test
     |   |- key = 4

    """

    # TODO instead of storing itself, make sections separate classes?
    def __init__(self, name="root", parent=None):
        self._sections = {}
        self._keys = {}
        self._name = name
        self._parent = parent

    # TODO get_section should not also be a setter!
    # separate set_section method?
    def get_section(self, name):
        """
        Return a section object. If the section does not exist then it is
        created.

        Parameters
        ----------
        name : str
            Name of the section to get/create

        Returns
        -------
        BoutOptions
            A new section with the original object as the parent
        """
        name = name.lower()

        if name in self._sections:
            return self._sections[name]
        else:
            newsection = BoutOptions(name, self)
            self._sections[name] = newsection
            return newsection

    def getSection(self, name):
        """
        Return a section object. If the section does not exist then it is
        created

        (non-PEP8 compliant alias for get_section, for backwards compatibility)

        Parameters
        ----------
        name : str
            Name of the section to get/create

        Returns
        -------
        BoutOptions
            A new section with the original object as the parent

        """
        return self.get_section(name)

    # TODO .get() method with optional evaluation

    # TODO comment stripping here
    def __getitem__(self, key):
        """
        First check if it's a section, then a value
        """
        key = key.lower()
        if key in self._sections:
            return self._sections[key]

        if key not in self._keys:
            raise KeyError(f"Key {key} not in section {self.path()}")
        return self._keys[key]

    def __setitem__(self, key, value):
        """
        Set a key
        """
        if len(key) == 0:
            return
        self._keys[key.lower()] = value

    # TODO give this a new name which can't be confused with filepath
    def path(self):
        """
        Returns the path of this section, joining together names of
        parents
        """

        if self._parent:
            return self._parent.path() + ":" + self._name
        return self._name

    # TODO this should return a dict of pairs: (section, list of keys)
    def keys(self):
        """Returns all keys, including sections and values

        """
        return list(self._sections) + list(self._keys)

    def sections(self):
        """Return a list of sub-sections

        """
        return self._sections.keys()

    def values(self):
        """Return a list of values

        """
        return self._keys.keys()

    # TODO an .items() method like configparser

    def as_dict(self):
        """Return a nested dictionary of all the options.

        """
        dicttree = {name:self[name] for name in self.values()}
        dicttree.update({name:self[name].as_dict() for name in self.sections()})
        return dicttree

    # TODO more intuitive definition of length
    def __len__(self):
        return len(self._sections) + len(self._keys)

    # TODO this should yield pairs: (section key, section)
    def __iter__(self):
        """Iterates over all keys. First values, then sections

        """
        for k in self._keys:
            yield k
        for s in self._sections:
            yield s

    def __str__(self, indent=""):
        """Print a pretty version of the options tree

        """
        text = self._name + "\n"

        for k in self._keys:
            text += indent + " |- " + k + " = " + str(self._keys[k]) + "\n"

        for s in self._sections:
            text += indent + " |- " + self._sections[s].__str__(indent+" |  ")
        return text

    # TODO import configparser's interpolation classes?
    def evaluate_scalar(self, name):
        """
        Evaluate (recursively) scalar expressions
        """
        expression = self._substitute_expressions(name)

        # replace ^ with ** so that Python evaluates exponentiation
        expression = expression.replace("^", "**")

        return eval(expression)

    def _substitute_expressions(self, name):
        expression = str(self[name]).lower()
        expression = self._evaluate_section(expression, "")
        parent = self._parent
        while parent is not None:
            sectionname = parent._name
            if sectionname is "root":
                sectionname = ""
            expression = parent._evaluate_section(expression, sectionname)
            parent = parent._parent

        return expression

    def _evaluate_section(self, expression, nested_sectionname):
        # pass a nested section name so that we can traverse the options tree
        # rooted at our own level and each level above us so that we can use
        # relatively qualified variable names, e.g. if we are in section
        # 'foo:bar:baz' then a variable 'x' from section 'bar' could be called
        # 'bar:x' (found traversing the tree starting from 'bar') or
        # 'foo:bar:x' (found when traversing tree starting from 'foo').
        for var in self.values():
            if nested_sectionname is not "":
                nested_name = nested_sectionname + ":" + var
            else:
                nested_name = var
            if re.search(r"(?<!:)\b"+re.escape(nested_name.lower())+r"\b", expression.lower()):
                # match nested_name only if not preceded by colon (which indicates more nesting)
                expression = re.sub(r"(?<!:)\b" + re.escape(nested_name.lower()) + r"\b",
                                    "(" + self._substitute_expressions(var) + ")",
                                    expression)

        for subsection in self.sections():
            if nested_sectionname is not "":
                nested_name = nested_sectionname + ":" + subsection
            else:
                nested_name = subsection
            expression = self.getSection(subsection)._evaluate_section(expression, nested_name)

        return expression

    # TODO option to write out comments - would allow for roundtripping
    def write(self, filename=None, overwrite=False):
        """ Write to BOUT++ options file

        This method will throw an error rather than overwriting an existing
        file unless the overwrite argument is set to true.
        Note, no comments from the original input file are transferred to the
        new one.

        Parameters
        ----------
        filename : str
            Path of the file to write
            (defaults to path of the file that was read in)
        overwrite : bool
            If False then throw an exception if 'filename' already exists.
            Otherwise, just overwrite without asking.
            (default False)
        """
        if filename is None:
            filename = self.filename

        if not overwrite and os.path.exists(filename):
            raise ValueError("Not overwriting existing file, cannot write "
                             f"output to {filename}")

        def write_section(basename, opts, f):
            if basename:
                f.write("["+basename+"]\n")
            for key, value in opts._keys.items():
                f.write(key+" = "+str(value)+"\n")
            for section in opts.sections():
                section_name = basename+":"+section if basename else section
                write_section(section_name, opts[section], f)

        with open(filename, "w") as f:
            write_section("", self, f)


class BoutOptionsFile(BoutOptions):
    """Parses a BOUT.inp configuration file, producing a tree of
    BoutOptions.

    Slight differences from ConfigParser, including allowing values
    before the first section header, and supporting nested section headers.
    Nesting uses the syntax [header:subheader].

    Parameters
    ----------
    filename : str, optional
        Path to file to read
    name : str, optional
        Name of root section (default: "root")
    gridfilename : str, optional
        If present, path to gridfile from which to read grid sizes (nx, ny, nz)
    nx, ny : int, optional
        - Specify sizes of grid, used when evaluating option strings
        - Cannot be given if gridfilename is specified
        - Must both be given if either is
        - If neither gridfilename nor nx, ny are given then will try to
          find nx, ny from (in order) the 'mesh' section of options,
          outputfiles in the same directory is the input file, or the grid
          file specified in the options file (used as a path relative to
          the current directory)
    nz : int, optional
        Use this value for nz when evaluating option expressions, if given.
        Overrides values found from input file, output files or grid files

    Examples
    --------

    >>> opts = BoutOptionsFile("BOUT.inp")
    >>> print(opts)   # Print all options in a tree
    root
    |- nout = 100
    |- timestep = 2
    ...

    >>> opts["All"]["scale"] # Value "scale" in section "All"
    1.0

    """

    # TODO options to read grid size from Dataset
    def __init__(self, filename="BOUT.inp", name="root",
                 gridfilename=None, nx=None, ny=None, nz=None):
        super().__init__(self, name)

        # Open the file
        self.filepath = Path(filename)
        with open(self.filepath, "r") as f:
            self._read_file(f)

        # TODO
        #self.nx, self.ny, self.nz = self._read_grid(gridfilename, nx, ny, nz)

    # TODO needs to be changed: DataFile -> Dataset
    def _read_grid(self, gridfilename=None, nx=None, ny=None, nz=None):
        try:
            # define arrays of x, y, z to be used for substitutions
            gridfile = None
            nzfromfile = None
            if gridfilename:
                if nx is not None or ny is not None:
                    raise ValueError("nx or ny given as inputs even though "
                                     "gridfilename was given explicitly, "
                                     "don't know which parameters to choose")
                with DataFile(gridfilename) as gridfile:
                    self.nx = float(gridfile["nx"])
                    self.ny = float(gridfile["ny"])
                    try:
                        nzfromfile = gridfile["MZ"]
                    except KeyError:
                        pass
            elif nx or ny:
                errmsg = "{} not specified. If either nx or ny are given, " \
                         "then both must be."
                if nx is None:
                    raise ValueError(errmsg.format(nx))
                if ny is None:
                    raise ValueError(errmsg.format(ny))
                self.nx = nx
                self.ny = ny
            else:
                try:
                    self.nx = self["mesh"].evaluate_scalar("nx")
                    self.ny = self["mesh"].evaluate_scalar("ny")
                except KeyError:
                    try:
                        # get nx, ny, nz from output files
                        from boutdata.collect import findFiles
                        file_list = findFiles(path=os.path.dirname(), prefix="BOUT.dmp")
                        with DataFile(file_list[0]) as f:
                            self.nx = f["nx"]
                            self.ny = f["ny"]
                            nzfromfile = f["MZ"]
                    except (IOError, KeyError):
                        try:
                            gridfilename = self["mesh"]["file"]
                        except KeyError:
                            gridfilename = self["grid"]
                        with DataFile(gridfilename) as gridfile:
                            self.nx = float(gridfile["nx"])
                            self.ny = float(gridfile["ny"])
                            try:
                                nzfromfile = float(gridfile["MZ"])
                            except KeyError:
                                pass
            if nz is not None:
                self.nz = nz
            else:
                try:
                    self.nz = self["mesh"].evaluate_scalar("nz")
                except KeyError:
                    try:
                        self.nz = self.evaluate_scalar("mz")
                    except KeyError:
                        if nzfromfile is not None:
                            self.nz = nzfromfile
            mxg = self._keys.get("MXG", 2)
            myg = self._keys.get("MYG", 2)

            # make self.x, self.y, self.z three dimensional now so
            # that expressions broadcast together properly.
            self.x = numpy.linspace((0.5 - mxg)/(self.nx - 2*mxg),
                                    1. - (0.5 - mxg)/(self.nx - 2*mxg),
                                    self.nx)[:, numpy.newaxis, numpy.newaxis]
            self.y = 2.*numpy.pi*numpy.linspace((0.5 - myg)/self.ny,
                                                1.-(0.5 - myg)/self.ny,
                                                self.ny + 2*myg)[numpy.newaxis, :, numpy.newaxis]
            self.z = 2.*numpy.pi*numpy.linspace(0.5/self.nz,
                                                1.-0.5/self.nz,
                                                self.nz)[numpy.newaxis, numpy.newaxis, :]
        except Exception as msg:
            raise BaseException("While building x, y, z coordinate arrays, an "
                                f"exception occured: {str(msg)}\n"
                                "Evaluating non-scalar options not available")

    # TODO better exception handling (import from configparser?)
    def _read_file(self, f):
        # Go through each line in the file
        section = self  # Start with root section
        for linenr, line in enumerate(f.readlines()):
            # First remove comments, either # or ;
            startpos = line.find("#")
            if startpos != -1:
                line = line[:startpos]
            startpos = line.find(";")
            if startpos != -1:
                line = line[:startpos]

            # Check section headers
            startpos = line.find("[")
            endpos = line.find("]")
            if startpos != -1:
                # A section heading
                if endpos == -1:
                    raise SyntaxError("Missing ']' on line %d" % (linenr,))
                line = line[(startpos + 1):endpos].strip()

                section = self
                while True:
                    scorepos = line.find(":")
                    if scorepos == -1:
                        break
                    sectionname = line[0:scorepos]
                    line = line[(scorepos + 1):]
                    section = section.getSection(sectionname)
                section = section.getSection(line)
            else:
                # A key=value pair

                eqpos = line.find("=")
                if eqpos == -1:
                    # No '=', so just set to true
                    section[line.strip()] = True
                else:
                    value = line[(eqpos + 1):].strip()
                    try:
                        # Try to convert to an integer
                        value = int(value)
                    except ValueError:
                        try:
                            # Try to convert to float
                            value = float(value)
                        except ValueError:
                            # Leave as a string
                            pass

                    section[line[:eqpos].strip()] = value

    def __repr__(self):
        # TODO Add grid-related options
        return f"BoutOptionsFile('{self.filepath}')"

    def evaluate(self, name):
        """Evaluate (recursively) expressions

        Sections and subsections must be given as part of 'name',
        separated by colons

        Parameters
        ----------
        name : str
            Name of variable to evaluate, including sections and
            subsections

        """
        section = self
        split_name = name.split(":")
        for subsection in split_name[:-1]:
            section = section.getSection(subsection)
        expression = section._substitute_expressions(split_name[-1])

        # replace ^ with ** so that Python evaluates exponentiation
        expression = expression.replace("^", "**")

        # substitute for x, y and z coordinates
        for coord in ["x", "y", "z"]:
            expression = re.sub(r"\b"+coord.lower()+r"\b", "self."+coord, expression)

        return eval(expression)
