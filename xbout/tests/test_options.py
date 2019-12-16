from pathlib import Path

import pytest

from xbout.options import BoutOptions, BoutOptionsFile


@pytest.fixture
def example_options_file(tmpdir_factory):
    """
    Mocks up a temporary example BOUT.inp (or BOUT.settings) file, and returns
    the path to it.
    """

    # Create options file
    options = BoutOptions()

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
    options['storm'] = {'B_0': '0.24          # Tesla',
                        'T_e0': '15           # eV',
                        'isothermal': 'true   # switch'}

    # Create temporary directory
    save_dir = tmpdir_factory.mktemp("inputdata")

    # Save
    optionsfilepath = save_dir.join('BOUT.inp')
    options.write(optionsfilepath)

    # Remove the first (and default) section headers to emulate BOUT.inp file format
    # TODO do this without opening file 3 times
    with open(optionsfilepath, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(optionsfilepath, 'w') as fout:
        fout.writelines(data[4:])

    return Path(optionsfilepath)


class TestReadFile:
    def test_no_file(self):
        with pytest.raises(FileNotFoundError):
            BoutOptionsFile('./wrong.inp')

    def test_open_real_example(self):
        # TODO this absolute filepath is sensitive to the directory the tests are run from?
        example_inp_path = Path.cwd() / './xbout/tests/data/options/BOUT.inp'
        opts = BoutOptionsFile(example_inp_path)
        assert opts.filepath.name == 'BOUT.inp'

    def test_open(self, example_options_file):
        opts = BoutOptionsFile(example_options_file)
        assert opts.filepath.name == 'BOUT.inp'

    def test_repr(self, example_options_file):
        opts_repr = repr(BoutOptionsFile(example_options_file))
        assert opts_repr == f"BoutOptionsFile('{example_options_file}')"


class TestAccess:
    def test_get_sections(self, example_options_file):
        sections = BoutOptionsFile(example_options_file).sections()

        # TODO for now ignore problem of nesting
        sections.remove('mesh:ddx')
        assert sections == ['top', 'mesh', 'laplace', 'storm']

    def test_get_str_values(self, example_options_file):
        opts = BoutOptionsFile(example_options_file)
        assert opts['laplace']['type'] == 'cyclic'
        assert opts['laplace'].get('type') == 'cyclic'

    @pytest.mark.xfail(reason="Nesting not yet implemented")
    def test_get_nested_section_values(self, example_options_file):
        opts = BoutOptionsFile(example_options_file)
        assert opts['mesh']['ddx']['upwind'] == 'C2'
        assert opts['mesh']['ddx'].get('upwind') == 'C2'


@pytest.mark.xfail(reason='Type conversions not yet implemented')
class TestTypeConversion:
    def test_get_int_values(self, example_options_file):
        opts = BoutOptionsFile(example_options_file)
        assert opts['mesh']['nx'] == 484
        assert opts['mesh'].get('nx') == 484

    def test_get_float_values(self, example_options_file):
        opts = BoutOptionsFile(example_options_file)
        assert opts['mesh']['Lx'] == 400.0
        assert opts['mesh'].get('Lx') == 400.0

    def test_get_bool_values(self, example_options_file):
        opts = BoutOptionsFile(example_options_file)
        assert opts['storm']['isothermal'] == True
        assert opts['storm'].get('isothermal') == True


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
