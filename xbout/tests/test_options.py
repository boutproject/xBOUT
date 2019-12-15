from pathlib import Path

import pytest

from xbout.options import BoutOptionsParser, BoutOptions


@pytest.fixture
def example_options_file(tmpdir_factory):
    """
    Mocks up a temporary example BOUT.inp (or BOUT.settings) file, and returns
    the path to it.
    """

    # Create options file
    options = BoutOptionsParser()

    # Fill it with example data
    options['top'] = {'timestep': '80',
                      'nout': '2000    # Number of outputted timesteps'}
    options['mesh'] = {'Ly': '2000.0   # ~4m',
                       'Lx': '400.0',
                       'nx': '484      # including 4 guard cells',
                       'ny': '1        # excluding guard cells',
                       'dx': 'Lx/(nx-4)',
                       'dy': 'Ly/ny'}
    options['mesh:ddx'] = {'first': 'C2',
                           'second': 'C2',
                           'upwind': 'C2'}
    options['laplace'] = {'type': 'cyclic',
                          'global_flags': '0',
                          'inner_boundary_flags': '1',
                          'outer_boundary_flags': '16',
                          'include_yguards': 'false'}
    options['storm'] = {'B_0': '0.24   # Tesla',
                        'T_e0': '15    # eV'}

    # Create temporary directory
    save_dir = tmpdir_factory.mktemp("inputdata")

    # Save
    optionsfilepath = save_dir.join('BOUT.inp')
    with open(optionsfilepath, 'w') as optionsfile:
        options.write(optionsfile)

    return Path(optionsfilepath)


class TestReadFile:
    def test_no_file(self):
        with pytest.raises(FileNotFoundError):
            BoutOptions('./')

    def test_open_real_example(self):
        # TODO this absolute filepath is sensitive to the directory the tests are run from?
        example_inp_path = Path.cwd() / './xbout/tests/data/options/BOUT.inp'
        opts = BoutOptions(example_inp_path)
        assert opts.filepath.name == 'BOUT.inp'

    def test_open(self, example_options_file):
        opts = BoutOptions(example_options_file)
        assert opts.filepath.name == 'BOUT.inp'

    def test_repr(self, example_options_file):
        opts_repr = repr(BoutOptions(example_options_file))
        assert opts_repr == f"BoutOptions('{example_options_file}')"


@pytest.mark.skip
class TestAccess:
    def test_get_sections(self):
        ...

    def test_get_values(self):
        ...

    def test_get_nested_section(self):
        ...


@pytest.mark.skip
class TestWriting:
    ...


@pytest.mark.skip
class TestRoundTrip:
    ...


@pytest.mark.skip
class TestEval:
    ...


@pytest.mark.skip
class TestSubstitution:
    ...
