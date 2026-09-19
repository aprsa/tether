"""Unit tests for interpreter discovery. No cluster needed.

`probe_interpreters()` takes its transport as an argument, so the cases that
matter -- six names for one file, a package's helper script sitting beside the
interpreter, a conda python on PATH -- are testable without arranging any of
them on a real machine.
"""

import pytest

import tether
from tether import venv


class Remote:
    """A transport answering from a script, remembering what it was asked."""

    def __init__(self, candidates='', identity=''):
        self.candidates, self.identity = candidates, identity
        self.commands = []

    def __call__(self, command):
        self.commands.append(command)
        return self.identity if '-c ' in command else self.candidates

    @property
    def trips(self):
        return len(self.commands)


# -- which paths are worth asking ----------------------------------------


def test_six_names_for_one_file_are_reported_once():
    """Terra reaches a single interpreter six ways; six answers would be noise."""
    got = venv._parse_candidates(
        '/usr/bin/python\t/usr/bin/python3.8\n'
        '/usr/bin/python3\t/usr/bin/python3.8\n'
        '/usr/bin/python3.8\t/usr/bin/python3.8\n'
        '/bin/python\t/usr/bin/python3.8\n'
        '/bin/python3\t/usr/bin/python3.8\n'
        '/bin/python3.8\t/usr/bin/python3.8\n'
    )
    assert got == ('/usr/bin/python3.8',)


def test_helpers_that_merely_start_with_python_are_dropped():
    """These would otherwise be *executed* by the identity probe."""
    got = venv._parse_candidates(
        '/usr/bin/python3-config\t/usr/bin/x86_64-linux-gnu-python3.8-config\n'
        '/usr/bin/python-dotenv\t/usr/bin/python-dotenv\n'
        '/usr/bin/python3-unidiff\t/usr/bin/python3-unidiff\n'
        '/usr/bin/python3.8\t/usr/bin/python3.8\n'
    )
    assert got == ('/usr/bin/python3.8',)


def test_the_typed_name_decides_not_the_resolved_one():
    """`readlink -f` can rename entirely, so filtering after resolution would
    drop a real interpreter that happens to link somewhere odd."""
    got = venv._parse_candidates('/usr/bin/python3\t/opt/vendor/bin/cpython-3.12\n')
    assert got == ('/opt/vendor/bin/cpython-3.12',)


@pytest.mark.parametrize('name', ['python', 'python3', 'python3.12', 'python3.13t'])
def test_interpreter_shaped_names(name):
    assert venv._parse_candidates(f'/usr/bin/{name}\t/usr/bin/{name}\n')


def test_order_follows_path():
    """First on PATH is first in the answer, so the shadowing order is visible."""
    got = venv._parse_candidates(
        '/opt/new/python3\t/opt/new/python3.12\n/usr/bin/python3\t/usr/bin/python3.8\n'
    )
    assert got == ('/opt/new/python3.12', '/usr/bin/python3.8')


def test_malformed_lines_are_skipped():
    assert venv._parse_candidates('nonsense\n\n/usr/bin/python3\n') == ()


# -- what each interpreter says about itself ------------------------------


def test_a_bare_metal_interpreter_is_kept():
    got = venv._parse_identity('/usr/bin/python3.8\t3.8.10\tFalse\tFalse\n')
    assert [(i.path, i.version) for i in got] == [('/usr/bin/python3.8', '3.8.10')]


def test_a_conda_interpreter_is_excluded():
    """Conda environments are conda's job; see probe_conda()."""
    assert venv._parse_identity('/opt/conda/bin/python\t3.14.7\tTrue\tFalse\n') == []


def test_an_interpreter_inside_a_venv_is_excluded():
    """A venv built from a venv inherits a base nobody asked for."""
    assert venv._parse_identity('/home/u/env/bin/python\t3.12.1\tFalse\tTrue\n') == []


def test_a_candidate_that_is_not_python_prints_nothing_and_is_skipped():
    assert venv._parse_identity('') == []


def test_a_login_banner_does_not_break_discovery():
    got = venv._parse_identity(
        'Welcome to the cluster!\n/usr/bin/python3.8\t3.8.10\tFalse\tFalse\n'
    )
    assert len(got) == 1


def test_a_path_with_a_space_survives():
    """Space-separated fields would split this into five and drop it."""
    got = venv._parse_identity('/opt/my python/bin/python3\t3.12.1\tFalse\tFalse\n')
    assert got[0].path == '/opt/my python/bin/python3'


def test_a_four_word_banner_is_not_an_interpreter():
    """`Welcome to the cluster!` is exactly four fields."""
    assert venv._parse_identity('Welcome to the cluster!\n') == []


def test_the_interpreter_is_asked_rather_than_the_filename_trusted():
    """A file called python3.8 may be anything; sys.version is authoritative."""
    got = venv._parse_identity('/usr/bin/python3.8\t3.11.9\tFalse\tFalse\n')
    assert got[0].version == '3.11.9'


# -- the commands that get sent -------------------------------------------


def test_setup_runs_before_the_search():
    """`module load python/3.12` is how a modern python appears at all."""
    remote = Remote()
    venv.probe_interpreters(remote, setup='module load python/3.12')
    assert remote.commands[0].startswith('module load python/3.12\n')


def test_a_path_entry_without_any_python_is_not_a_failure():
    """A `for` loop reports its last iteration's status."""
    assert venv._CANDIDATES.rstrip().endswith('|| true')


def test_interpreter_paths_are_quoted():
    got = venv._identity_query(['/opt/my python/bin/python3'])
    assert got.startswith("'/opt/my python/bin/python3' -c ")


def test_nothing_found_costs_one_trip():
    remote = Remote(candidates='\n')
    assert venv.probe_interpreters(remote) == []
    assert remote.trips == 1


def test_finding_something_costs_two():
    remote = Remote(
        candidates='/usr/bin/python3.8\t/usr/bin/python3.8\n',
        identity='/usr/bin/python3.8\t3.8.10\tFalse\tFalse\n',
    )
    assert len(venv.probe_interpreters(remote)) == 1
    assert remote.trips == 2


def test_an_interpreter_describes_itself_for_a_human():
    assert str(tether.PythonInstallation('/usr/bin/python3.8', '3.8.10')) == (
        '/usr/bin/python3.8 (python 3.8.10)'
    )


# -- finding virtual environments -----------------------------------------
#
# venv keeps no registry, so discovery is told where to look rather than
# hunting. These pin what "told where to look" accepts, and what happens to a
# venv whose interpreter has gone away -- a state no test machine happens to
# be in, and the one that matters most.


class VenvRemote:
    """A transport answering each of the three venv queries from a script."""

    def __init__(self, candidates='', inspect='', config=''):
        self.candidates, self.inspect, self.config = candidates, inspect, config
        self.commands = []

    def __call__(self, command):
        self.commands.append(command)
        if command.startswith('printf') and 'cat ' in command:
            return self.config
        if '-c ' in command:
            return self.inspect
        return self.candidates

    @property
    def trips(self):
        return len(self.commands)


def test_both_candidate_shapes_reduce_to_a_directory():
    """$VIRTUAL_ENV gives the venv; the glob gives the pyvenv.cfg inside it."""
    got = venv._parse_venvs(
        '/home/u/.venvs/active\n/home/u/.venvs/other/pyvenv.cfg\n'
    )
    assert got == ('/home/u/.venvs/active', '/home/u/.venvs/other')


def test_the_active_venv_appearing_twice_is_reported_once():
    got = venv._parse_venvs('/home/u/.venvs/a\n/home/u/.venvs/a/pyvenv.cfg\n')
    assert got == ('/home/u/.venvs/a',)


def test_nothing_active_and_no_base_finds_nothing():
    assert venv._parse_venvs('\n\n') == ()


def test_the_base_may_be_a_venv_or_a_shelf_of_them():
    """So `~/.venvs` and `~/.venvs/phoebe` both work."""
    got = venv._venvs_query('~/.venvs')
    assert '"$HOME/.venvs"/pyvenv.cfg' in got
    assert '"$HOME/.venvs"/*/pyvenv.cfg' in got


def test_no_base_asks_only_about_the_active_venv():
    got = venv._venvs_query()
    assert 'VIRTUAL_ENV' in got and 'ls -d' not in got


def test_a_venv_that_runs_is_reported():
    got = venv._parse_inspect('/home/u/.venvs/a\t3.12.1\t/usr\n')
    assert (got[0].path, got[0].version, got[0].base) == ('/home/u/.venvs/a', '3.12.1', '/usr')
    assert not got[0].broken


def test_a_plain_interpreter_is_not_a_venv():
    """Someone pointed venvs_base at a Python installation."""
    assert venv._parse_inspect('/usr\t3.12.1\t/usr\n') == []


def test_a_venv_that_will_not_run_says_nothing_and_is_absent():
    assert venv._parse_inspect('') == []


def test_a_dangling_venv_is_excluded_by_default():
    """Its bin/python exits 127, so it never answers."""
    remote = VenvRemote(candidates='/home/u/.venvs/dead/pyvenv.cfg\n', inspect='')
    assert venv.probe_venvs(remote, '~/.venvs') == []
    assert remote.trips == 2          # no third trip when broken are not wanted


def test_include_broken_reports_it_with_what_pyvenv_cfg_remembers():
    remote = VenvRemote(
        candidates='/home/u/.venvs/dead/pyvenv.cfg\n',
        inspect='',
        config='/home/u/.venvs/dead\nhome = /nonexistent\nversion = 3.9.2\n',
    )
    got = venv.probe_venvs(remote, '~/.venvs', include_broken=True)
    assert remote.trips == 3
    assert (got[0].path, got[0].version, got[0].base, got[0].broken) == (
        '/home/u/.venvs/dead', '3.9.2', '/nonexistent', True,
    )
    assert str(got[0]).endswith('(python 3.9.2) (broken)')


def test_a_working_venv_costs_no_third_trip_even_when_broken_are_wanted():
    remote = VenvRemote(
        candidates='/home/u/.venvs/a/pyvenv.cfg\n',
        inspect='/home/u/.venvs/a\t3.12.1\t/usr\n',
    )
    assert len(venv.probe_venvs(remote, '~/.venvs', include_broken=True)) == 1
    assert remote.trips == 2


def test_the_old_pyvenv_format_is_enough():
    """Python 3.8 records only home, version and site-packages -- no
    `executable`, which arrived in 3.11. Terra's venvs are all like this."""
    got = venv._parse_config(
        '/home/u/.venvs/a\n'
        'home = /usr/bin\n'
        'include-system-site-packages = false\n'
        'version = 3.8.10\n'
    )
    assert got['/home/u/.venvs/a']['version'] == '3.8.10'
    assert got['/home/u/.venvs/a']['home'] == '/usr/bin'


def test_config_blocks_do_not_bleed_into_each_other():
    got = venv._parse_config(
        '/home/u/a\nversion = 3.8.10\n/home/u/b\nversion = 3.12.1\n'
    )
    assert got['/home/u/a']['version'] == '3.8.10'
    assert got['/home/u/b']['version'] == '3.12.1'


def test_a_config_that_could_not_be_read_leaves_the_version_unknown():
    remote = VenvRemote(
        candidates='/home/u/.venvs/dead/pyvenv.cfg\n',
        inspect='',
        config='/home/u/.venvs/dead\n',
    )
    got = venv.probe_venvs(remote, '~/.venvs', include_broken=True)
    assert got[0].version == venv.UNKNOWN and got[0].base == venv.UNKNOWN


# -- the exit-status trap, in both probes ---------------------------------


def test_a_dud_interpreter_does_not_fail_the_whole_probe():
    """Commands are newline-joined, so the last one's status is the script's."""
    assert venv._identity_query(['/usr/bin/python3']).rstrip().endswith('|| true')


def test_a_dangling_venv_does_not_fail_the_whole_probe():
    assert venv._inspect_query(['/home/u/.venvs/dead']).rstrip().endswith('|| true')
