from pathlib import Path
from textwrap import dedent

import numpy as np

import pytest

from xbout.options import Section, OptionsTree, OptionsFile


@pytest.fixture
def example_section():
    contents = {'type': 'cyclic',
                'global_flags': '0',
                'inner_boundary_flags': '1',
                'outer_boundary_flags': '16  # dirichlet',
                'include_yguards': 'false  ; another comment'}
    return Section(name='laplace', data=contents)


class TestSection:
    def test_create_section(self, example_section):
        sect = example_section
        assert sect.name == 'laplace'
        assert sect.data == {'type': 'cyclic',
                             'global_flags': '0',
                             'inner_boundary_flags': '1',
                             'outer_boundary_flags': '16  # dirichlet',
                             'include_yguards': 'false  ; another comment'}

    def test_get(self, example_section):
        sect = example_section
        assert sect.get('type') == 'cyclic'

    def test_get_comments(self, example_section):
        sect = example_section
        # Comments marked by '#'
        assert sect.get('outer_boundary_flags') == '16'
        with_comment = sect.get('outer_boundary_flags', evaluate=False,
                                keep_comments=True)
        assert with_comment == '16  # dirichlet'

        # Comments marked by ';'
        assert sect.get('include_yguards') == 'false'
        with_other_comment = sect.get('include_yguards', evaluate=False,
                                      keep_comments=True)
        assert with_other_comment == 'false  ; another comment'

    def test_getitem(self, example_section):
        sect = example_section
        assert sect['type'] == 'cyclic'
        assert sect['outer_boundary_flags'] == '16'

    def test_set(self, example_section):
        sect = example_section
        sect.set('max', '10000')
        assert sect.get('max') == '10000'

        sect['min'] = '100'
        assert sect.get('min') == '100'

    def test_set_nested(self, example_section):
        sect = example_section
        nested = Section('naulin', data={'iterations': '1000'})
        sect['naulin'] = nested

        assert isinstance(sect.get('naulin'), Section)
        assert sect.get('naulin').get('iterations') == '1000'
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

    def test_write(self, example_section, tmpdir_factory):
        sect = example_section
        sect['naulin'] = {'iterations': '1000'}
        file = tmpdir_factory.mktemp("write_data").join('write_test.inp')

        with open(file, 'w') as f:
            sect._write(f)

        with open(file, 'r') as f:
            result = file.read()
        expected = dedent("""\
                          [laplace]
                          type = cyclic
                          global_flags = 0
                          inner_boundary_flags = 1
                          outer_boundary_flags = 16  # dirichlet
                          include_yguards = false  ; another comment
                          
                          [laplace:naulin]
                          iterations = 1000
                          """)
        assert result == expected

    def test_write_root(self, tmpdir_factory):
        sect = Section(name='root', parent=None,
                       data={'timestep': '80  # Timestep length',
                             'nout': '2000  # Number of timesteps'})
        file = tmpdir_factory.mktemp("write_data").join('write_test.inp')

        with open(file, 'w') as f:
            sect._write(f)

        with open(file, 'r') as f:
            result = file.read()
        expected = dedent("""\
                          timestep = 80  # Timestep length
                          nout = 2000  # Number of timesteps
                          """)
        assert result == expected


@pytest.fixture
def example_options_tree():
    """
    Mocks up a temporary example BOUT.inp (or BOUT.settings) tree
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


class TestTypeConversion:
    def test_get_int_values(self, example_options_tree):
        opts = OptionsTree(example_options_tree)
        assert opts['mesh'].get('nx', evaluate=True) == 484

    def test_get_float_values(self, example_options_tree):
        opts = OptionsTree(example_options_tree)
        assert opts['mesh'].get('Lx', evaluate=True) == 400.0

    def test_get_bool_values(self, example_options_tree):
        opts = OptionsTree(example_options_tree)
        assert opts['storm'].get('isothermal', evaluate=True) == True


class TestEvaluation:
    def test_eval_arithmetic(self):
        params = Section(name='params', data={'T0': '5 * 10**18'})
        assert params.get('T0', evaluate=True) == 5 * 10**18

    def test_eval_powers(self):
        params = Section(name='params', data={'n0': '10^18'})
        assert params.get('n0', evaluate=True) == 10**18

    # TODO generalise this to test other numpy imports
    def test_eval_numpy(self):
        params = Section(name='params', data={'angle': 'sin(pi/2)'})
        assert params.get('angle', evaluate=True) == np.sin(np.pi/2)


@pytest.fixture
def example_options_tree_with_substitution():
    """
    Mocks up a temporary example BOUT.inp (or BOUT.settings) tree,
    containing variables which need to be substituted with values from
    other sections
    """

    # Create options file
    options = OptionsTree()

    # Fill it with example data
    options.data = {'timestep': '80',
                    'nout': '2000    # Number of outputted timesteps'}

    # Substitutions only within section
    options['mesh'] = {'Ly': '2000.0   # ~4m',
                       'Lx': '400.0',
                       'nx': '484      # including 4 guard cells',
                       'dx': 'Lx/(nx-4)',
                       'dy': 'Ly'}

    # Substitute from section above
    options['mesh']['ddx'] = {'first': 'C2',
                              'length': 'mesh:Lx*2',
                              'num': '3'}

    # TODO Substitute from section below
    options['model'] = {}
    #options['model']['energy'] = ''
    #options['model']['magnetic_energy'] = 'B_0^2'

    # Substitute from arbitrary section
    options['model']['storm'] = {'B_0': '0.24          # Tesla',
                                 'T_e0': '15           # eV',
                                 'L_parallel': 'mesh:ddx:num * 10',
                                 'isothermal': 'true   # switch'}

    # Self-referencing substution
    options['apples'] = {'green': 'lemons:yellow'}
    options['lemons'] = {'yellow': 'apples:green'}

    return options


@pytest.mark.xfail  
class TestSubstitution:
    def test_substitute_from_section(self, example_options_tree_with_substitution):
        opts = example_options_tree_with_substitution
        ly = opts['mesh']['Ly']
        assert opts['mesh'].get('dy', substitute=True) == ly

    def test_substitute_from_parent_section(self, example_options_tree_with_substitution):
        opts = example_options_tree_with_substitution
        lx = opts['mesh']['Lx']
        assert opts['mesh']['ddx'].get('length', substitute=True) == f"{lx}*2"

    @pytest.mark.skip
    def test_substitute_from_child_section(self, example_options_tree_with_substitution):
        opts = example_options_tree_with_substitution
        lx = opts['mesh']['Lx']
        assert opts['mesh']['ddx'].get('length', substitute=True) == f"{lx}*2"

    def test_substitute_from_arbitrary_section(self, example_options_tree_with_substitution):
        opts = example_options_tree_with_substitution
        ddx = opts['mesh']['ddx']['num']
        assert opts['model']['storm'].get('L_parallel', substitute=True) == f"{ddx} * 10"

    def test_two_substitutions_required(self, example_options_tree_with_substitution):
        opts = example_options_tree_with_substitution
        assert opts['mesh'].get('dx', substitute=True) == '400.0/(484-4)'

    def test_substitute_requires_substitution(self, example_options_tree_with_substitution):
        ...

    @pytest.mark.skip
    def test_contains_colon_but_no_substitute_found(self, example_options_tree_with_substitution):
        with pytest.raises(InterpolationError): #?
            ...

    def test_detect_infinite_substitution_cycle(self, example_options_tree_with_substitution):
        ...


@pytest.mark.skip
class TestWriting:
    ...


@pytest.mark.skip
class TestRoundTrip:
    ...
