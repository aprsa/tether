"""Unit tests for conda discovery. No cluster needed.

`conda.probe()` takes its transport as an argument, so the whole algorithm --
the commands it builds, the output it parses, and the round trips it decides
to make or skip -- is exercisable against a dictionary. The awkward cases are
the point: a half-installed conda, an environment carrying its own
`conda-meta`, a base that will not name its version. Building those on a live
machine is possible but slow, and they are exactly the states a real cluster
never happens to be in when you need them.
"""

import pytest

import tether
from tether import conda


class Remote:
    """A transport that answers from a script and remembers what it was asked."""

    def __init__(self, search='', listing='', version='', fails=False):
        self.search, self.listing, self.version = search, listing, version
        self.fails = fails
        self.commands = []

    def __call__(self, command):
        self.commands.append(command)
        if command.startswith('ls -d'):
            return self.listing
        if command.endswith('--version'):
            if self.fails:
                raise tether.RemoteCommandError(
                    tether.Result(command, 127, '', 'no such file')
                )
            return self.version
        return self.search

    @property
    def trips(self):
        return len(self.commands)


# -- what gets asked ------------------------------------------------------


def test_every_candidate_source_emits_identically():
    """The workdir candidate once rendered `printf '%s\\\\n'` -- a literal
    backslash-n in shell -- so tether could not discover its own installation.
    Every source now goes through one emitter, which is what makes that
    unrepeatable."""
    remote = Remote()
    conda.probe(remote, '~/.tether/conda')
    printfs = [l for l in remote.commands[0].splitlines() if l.startswith('printf')]
    assert len(printfs) == 2
    assert {l.split(' ', 2)[1] for l in printfs} == {"'%s\\n'"}


def test_setup_runs_before_anything_is_looked_up():
    """A conda that only appears once `module load anaconda` has run is found
    only if the setup precedes the lookups."""
    remote = Remote()
    conda.probe(remote, setup='module load anaconda')
    assert remote.commands[0].startswith('module load anaconda\n')


def test_without_a_workdir_only_the_standard_sources_are_asked():
    remote = Remote()
    conda.probe(remote)
    assert remote.commands[0].count('printf') == 1


def test_paths_are_quoted_but_globs_are_not():
    """A quoted glob would be taken literally and match nothing."""
    remote = Remote(search='/opt/my conda\n')
    conda.probe(remote)
    assert "'/opt/my conda'/envs/*/" in remote.commands[1]


def test_a_glob_that_matches_nothing_is_not_a_failure():
    """`ls` exits non-zero when a glob misses, which is the ordinary case."""
    remote = Remote(search='/opt/conda\n')
    conda.probe(remote)
    assert remote.commands[1].endswith('|| true')


def test_the_per_user_fallback_is_always_asked_about():
    """Named environments land there whenever the base is not writable."""
    remote = Remote(search='/opt/conda\n')
    conda.probe(remote)
    assert '"$HOME"/.conda/envs/*/' in remote.commands[1]


def test_an_executable_is_asked_about_by_its_base():
    """A lookup lands on the executable; conda is addressed by its base."""
    remote = Remote(search='/opt/conda/bin/conda\n/home/u/mini/condabin/conda\n')
    conda.probe(remote)
    assert '/opt/conda/etc/profile.d/conda.sh' in remote.commands[1]
    assert '/home/u/mini/etc/profile.d/conda.sh' in remote.commands[1]


def test_a_repeated_candidate_is_asked_about_once():
    """Several sources routinely name the same installation."""
    remote = Remote(search='/opt/conda/bin/conda\n/opt/conda\n/opt/conda/\n\n  \n')
    conda.probe(remote)
    assert remote.commands[1].count('/opt/conda/envs/*/') == 1


# -- how many round trips -------------------------------------------------


def test_nothing_found_costs_one_trip_and_is_not_an_error():
    """A machine with no conda is a normal answer, not a failure."""
    remote = Remote(search='\n')
    assert conda.probe(remote) == []
    assert remote.trips == 1


def test_a_version_read_from_conda_meta_costs_no_extra_trip():
    """Reading it off a filename is the whole reason the listing asks for it."""
    remote = Remote(
        search='/opt/conda\n',
        listing='/opt/conda/etc/profile.d/conda.sh\n'
                '/opt/conda/conda-meta/conda-26.3.2-py312_0.json\n',
    )
    assert conda.probe(remote)[0].version == '26.3.2'
    assert remote.trips == 2


def test_only_a_base_that_withholds_its_version_pays_for_a_third_trip():
    remote = Remote(
        search='/opt/conda\n',
        listing='/opt/conda/etc/profile.d/conda.sh\n',
        version='conda 26.3.2\n',
    )
    assert conda.probe(remote)[0].version == '26.3.2'
    assert remote.trips == 3
    assert remote.commands[2] == '/opt/conda/bin/conda --version'


def test_a_base_whose_conda_will_not_run_is_still_reported():
    """We already know it is odd; unknown is the answer, not an exception."""
    remote = Remote(
        search='/opt/conda\n',
        listing='/opt/conda/etc/profile.d/conda.sh\n',
        fails=True,
    )
    found = conda.probe(remote)
    assert (found[0].base, found[0].version) == ('/opt/conda', 'unknown')


# -- what the output means ------------------------------------------------


def probe_listing(listing):
    """Drive a probe whose listing is the only thing under test."""
    return conda.probe(Remote(search='/candidate\n', listing=listing))


def test_unparseable_lines_are_skipped():
    """Same contract as the Slurm parsers: a malformed line is skipped, not
    fatal -- a login banner or a stray warning must not break discovery."""
    assert probe_listing('Welcome to the cluster!\n\n/opt/somewhere/else\n') == []


def test_only_the_activation_hook_makes_a_base():
    """An environment looks like a base in every way but this one."""
    got = probe_listing(
        '/opt/conda/etc/profile.d/conda.sh\n'
        '/opt/conda/conda-meta/conda-26.3.2-py312_0.json\n'
        '/home/u/mini/envs/crimpl/conda-meta/conda-25.5.1-py312_0.json\n'
    )
    assert [(i.base, i.version) for i in got] == [('/opt/conda', '26.3.2')]


def test_an_environment_carrying_conda_is_not_read_as_an_environment():
    """`<base>/envs/<name>/conda-meta/conda-*.json` contains `/envs/`, and a
    looser rule would invent an environment named after the whole tail."""
    got = probe_listing(
        '/home/u/mini/etc/profile.d/conda.sh\n'
        '/home/u/mini/envs/crimpl/\n'
        '/home/u/mini/envs/crimpl/conda-meta/conda-25.5.1-py312_0.json\n'
    )
    assert got[0].environments == ('crimpl',)


def test_environments_are_grouped_under_their_base():
    got = probe_listing(
        '/opt/conda/etc/profile.d/conda.sh\n'
        '/opt/conda/conda-meta/conda-26.3.2-py312_0.json\n'
        '/opt/conda/envs/phoebe/\n'
        '/opt/conda/envs/dev/\n'
        '/home/u/miniconda3/etc/profile.d/conda.sh\n'
        '/home/u/miniconda3/conda-meta/conda-25.5.1-py312_0.json\n'
        '/home/u/miniconda3/envs/crimpl/\n'
    )
    assert [(i.base, i.version, i.environments) for i in got] == [
        ('/opt/conda', '26.3.2', ('dev', 'phoebe')),
        ('/home/u/miniconda3', '25.5.1', ('crimpl',)),
    ]


def test_the_per_user_fallback_belongs_to_every_base():
    """Conda activates a name from `~/.conda/envs` whichever base is active."""
    got = probe_listing(
        '/opt/conda/etc/profile.d/conda.sh\n'
        '/home/u/mini/etc/profile.d/conda.sh\n'
        '/home/u/mini/envs/own/\n'
        '/home/u/.conda/envs/shared/\n'
    )
    assert [i.environments for i in got] == [('shared',), ('own', 'shared')]


def test_environments_under_a_failed_candidate_are_dropped():
    """A directory with `envs/` but no hook is not an installation, and its
    contents must not be credited to installations that are."""
    got = probe_listing('/opt/conda/etc/profile.d/conda.sh\n/opt/broken/envs/orphan/\n')
    assert got[0].environments == ()


def test_the_first_conda_meta_entry_settles_the_version():
    """An upgraded base keeps records of what it used to be."""
    got = probe_listing(
        '/opt/conda/etc/profile.d/conda.sh\n'
        '/opt/conda/conda-meta/conda-24.1.0-py311_0.json\n'
        '/opt/conda/conda-meta/conda-26.3.2-py312_0.json\n'
    )
    assert got[0].version == '24.1.0'


@pytest.mark.parametrize('noise', ['', 'conda\n', 'bash: conda: not found\n'])
def test_an_unrecognisable_version_is_unknown_not_a_crash(noise):
    remote = Remote(
        search='/opt/conda\n',
        listing='/opt/conda/etc/profile.d/conda.sh\n',
        version=noise,
    )
    assert conda.probe(remote)[0].version == 'unknown'


def test_an_installation_describes_itself_for_a_human():
    shown = str(tether.CondaInstallation('/opt/conda', '26.3.2', ('dev',)))
    assert shown == '/opt/conda (conda 26.3.2): dev'
    assert 'no environments' in str(tether.CondaInstallation('/opt/conda', '26.3.2'))
