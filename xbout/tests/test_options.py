from pathlib import Path
from textwrap import dedent

import pytest

from xbout.options import Section, OptionsTree, OptionsFile


from pprint import PrettyPrinter

pp = PrettyPrinter()


@pytest.fixture
def example_section():
    contents = {'type': 'cyclic',
                'global_flags': '0',
                'inner_boundary_flags': '1',
                'outer_boundary_flags': '16  # dirichlet',
                'include_yguards': 'false'}
    return Section(name='laplace', data=contents)


class TestSection:
    def test_create_section(self, example_section):
        sect = example_section
        assert sect.name == 'laplace'
        assert sect.data == {'type': 'cyclic',
                             'global_flags': '0',
                             'inner_boundary_flags': '1',
                             'outer_boundary_flags': '16  # dirichlet',
                             'include_yguards': 'false'}

    def test_get(self, example_section):
        sect = example_section
        assert sect.get('type', evaluate=False) == 'cyclic'

        assert sect.get('outer_boundary_flags', evaluate=False) == '16'
        with_comment = sect.get('outer_boundary_flags', evaluate=False,
                                keep_comments=True)
        assert with_comment == '16  # dirichlet'

    @pytest.mark.xfail(reason="Evaluate not yet implemented")
    def test_getitem(self, example_section):
        sect = example_section
        assert sect['type'] == 'cyclic'
        assert sect['outer_boundary_flags'] == '16'

    def test_set(self, example_section):
        sect = example_section
        sect.set('max', '10000')
        assert sect.get('max', evaluate=False) == '10000'

        sect['min'] = '100'
        assert sect.get('min', evaluate=False) == '100'

    def test_set_nested(self, example_section):
        sect = example_section
        nested = Section('naulin', data={'iterations': '1000'})
        sect['naulin'] = nested

        assert isinstance(sect.get('naulin'), Section)
        assert sect.get('naulin').get('iterations', evaluate=False) == '1000'
        assert sect['naulin']['iterations'] == '1000'

    def test_set_nested_as_dict(self, example_section):
        sect = example_section
        sect['naulin'] = {'iterations': '1000'}

        assert isinstance(sect.get('naulin'), Section)
        assert sect.get('naulin').get('iterations', evaluate=False) == '1000'
        assert sect['naulin']['iterations'] == '1000'
        assert sect['naulin'].parent == 'laplace'

    def test_find_parents(self, example_section):
        sect = example_section
        sect['naulin'] = {'iterations': '1000'}

        assert sect['naulin'].lineage() == 'laplace:naulin'

    def test_print(self, example_section):
        sect = example_section
        sect['naulin'] = {'iterations': '1000'}
        expected = dedent("""\
                          [laplace]
                          |-- type = cyclic
                          |-- global_flags = 0
                          |-- inner_boundary_flags = 1
                          |-- outer_boundary_flags = 16
                          |-- include_yguards = false
                          |-- [naulin]
                          |-- |-- iterations = 1000
                          """)
        assert str(sect) == expected

@pytest.fixture
def example_options_tree():
    """
    Mocks up a temporary example BOUT.inp (or BOUT.settings) file, and returns
    the path to it.
    """

    # Create options file
    options = OptionsTree()

    # Fill it with example data
    options.data = {'timestep': '80',
                    'nout': '2000    # Number of outputted timesteps'}
    options['mesh'] = {'Ly': '2000.0   # ~4m',
                       'Lx': '400.0',
                       'nx': '484      # including 4 guard cells',
                       'ny': '1        # excluding guard cells',
                       'dx': 'Lx/(nx-4)',
                       'dy': 'Ly/ny'}
    options['mesh']['ddx'] = {'first': 'C2',
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

    return options

"""
def example_options_file():
    # Create temporary directory
    save_dir = tmpdir_factory.mktemp("inputdata")

    # Save
    optionsfilepath = save_dir.join('BOUT.inp')
    options.write_to(optionsfilepath)

    # Remove the first (and default) section headers to emulate BOUT.inp file format
    # TODO do this without opening file 3 times
    with open(optionsfilepath, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(optionsfilepath, 'w') as fout:
        fout.writelines(data[4:])

    return Path(optionsfilepath)
"""


class TestAccess:
    def test_get_sections(self, example_options_tree):
        sections = OptionsTree(example_options_tree).sections()
        lineages = [section.lineage() for section in sections]
        assert lineages == ['root', 'root:mesh', 'root:mesh:ddx',
                            'root:laplace', 'root:storm']

    def test_get_str_values(self, example_options_tree):
        opts = OptionsTree(example_options_tree)
        assert opts['laplace']['type'] == 'cyclic'
        assert opts['laplace'].get('type') == 'cyclic'

    def test_get_nested_section_values(self, example_options_tree):
        opts = OptionsTree(example_options_tree)
        assert opts['mesh']['ddx']['upwind'] == 'C2'
        assert opts['mesh']['ddx'].get('upwind') == 'C2'


@pytest.mark.xfail
class TestReadFile:
    def test_no_file(self):
        with pytest.raises(FileNotFoundError):
            OptionsFile('./wrong.inp')

    def test_open_real_example(self):
        # TODO this absolute filepath is sensitive to the directory the tests are run from?
        example_inp_path = Path.cwd() / './xbout/tests/data/options/BOUT.inp'
        opts = OptionsFile(example_inp_path)
        assert opts.filepath.name == 'BOUT.inp'

    def test_open(self, example_options_file):
        opts = OptionsFile(example_options_file)
        assert opts.filepath.name == 'BOUT.inp'

    def test_repr(self, example_options_file):
        opts_repr = repr(OptionsFile(example_options_file))
        assert opts_repr == f"OptionsFile('{example_options_file}')"



@pytest.mark.xfail(reason='Type conversions not yet implemented')
class TestTypeConversion:
    def test_get_int_values(self, example_options_tree):
        opts = OptionsFile(example_options_tree)
        assert opts['mesh']['nx'] == 484
        assert opts['mesh'].get('nx') == 484

    def test_get_float_values(self, example_options_tree):
        opts = OptionsFile(example_options_tree)
        assert opts['mesh']['Lx'] == 400.0
        assert opts['mesh'].get('Lx') == 400.0

    def test_get_bool_values(self, example_options_tree):
        opts = OptionsFile(example_options_tree)
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
