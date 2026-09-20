"""Unit tests for conda discovery. No cluster needed.

`conda.probe()` takes its transport as an argument, so the whole algorithm --
the commands it builds, the output it parses, and the round trips it decides
to make or skip -- is exercisable against a dictionary. The awkward cases are
the point: a half-installed conda, an environment carrying its own
`conda-meta`, a base that will not name its version. Building those on a live
machine is possible but slow, and they are exactly the states a real cluster
never happens to be in when you need them.
"""

import json
import shlex

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

# -- installing: choosing what to download --------------------------------
#
# Discovery, not construction: the release names its own assets. These pin the
# selection rules against real Miniforge shapes without fetching 124MB.


def release_json(tag='26.7.2-0', names=None):
    """A GitHub release listing, in the shape the API actually returns."""
    if names is None:
        names = [
            f'Miniforge3-{tag}-Linux-x86_64.sh',
            f'Miniforge3-{tag}-Linux-x86_64.sh.sha256',
            f'Miniforge3-{tag}-Linux-aarch64.sh',
            f'Miniforge3-{tag}-Linux-aarch64.sh.sha256',
            'Miniforge3-Linux-x86_64.sh',          # the unchecksummed alias
        ]
    return json.dumps({
        'tag_name': tag,
        'assets': [{'name': n, 'browser_download_url': f'https://x/{n}'} for n in names],
    })


def test_the_installer_and_its_checksum_are_found_together():
    url, sha = conda._parse_release(release_json(), 'x86_64')
    assert url.endswith('Miniforge3-26.7.2-0-Linux-x86_64.sh')
    assert sha == url + '.sha256'


def test_the_unchecksummed_alias_is_never_chosen():
    """`Miniforge3-Linux-x86_64.sh` is the same bytes but has no `.sha256` at
    any URL, so choosing it would mean installing unverified."""
    url, _ = conda._parse_release(release_json(), 'x86_64')
    assert '26.7.2-0' in url


def test_the_architecture_decides():
    url, _ = conda._parse_release(release_json(), 'aarch64')
    assert 'aarch64' in url and 'x86_64' not in url


def test_an_unsupported_architecture_says_what_is_offered():
    with pytest.raises(tether.CondaError, match='no checksummed installer for Linux-s390x'):
        conda._parse_release(release_json(), 's390x')


def test_an_installer_without_a_checksum_is_refused():
    """Better to fail than to install something we cannot verify."""
    body = release_json(names=['Miniforge3-26.7.2-0-Linux-x86_64.sh'])
    with pytest.raises(tether.CondaError, match='no checksummed installer'):
        conda._parse_release(body, 'x86_64')


@pytest.mark.parametrize('body', ['', 'not json', '{}', '{"tag_name": "x"}'])
def test_unreadable_release_metadata_points_at_the_escape_hatch(body):
    with pytest.raises(tether.CondaError, match='installer='):
        conda._parse_release(body, 'x86_64')


def test_latest_is_asked_for_by_default_and_a_tag_when_pinned():
    assert conda._release_query(None).endswith('releases/latest')
    assert conda._release_query('26.7.2-0').endswith('releases/tags/26.7.2-0')


def test_a_pinned_version_cannot_smuggle_shell():
    """The tag is interpolated into a URL, so a hostile one must survive as a
    single argument rather than becoming a second command."""
    hostile = "x'; rm -rf /"
    assert shlex.split(conda._release_query(hostile)) == [
        'curl', '-fsSLm', '60', f'{conda._MINIFORGE}/releases/tags/{hostile}',
    ]


# -- installing: checksums ------------------------------------------------


def test_both_checksum_shapes_parse_the_same_way():
    """`sha256sum` output and a published `.sha256` file share one format."""
    published = '281b0ac7d5  ./Miniforge3-26.7.2-0-Linux-x86_64.sh'
    computed = '281b0ac7d5  /home/u/.tether/Miniforge3-26.7.2-0-Linux-x86_64.sh'
    assert conda._parse_sha256(published) == conda._parse_sha256(computed) == '281b0ac7d5'


def test_no_checksum_output_is_empty_not_a_crash():
    assert conda._parse_sha256('') == ''


# -- installing: what gets run --------------------------------------------


def test_the_installer_runs_non_interactively_into_the_prefix():
    got = conda._install_command('/tmp/Miniforge3.sh', '~/.tether/conda')
    assert 'bash /tmp/Miniforge3.sh -b -p "$HOME/.tether/conda"' in got


def test_only_the_parent_is_created_never_the_prefix():
    """The installer refuses an existing prefix, which is what we want."""
    got = conda._install_command('/tmp/i.sh', '/opt/a/b/conda')
    assert 'mkdir -p /opt/a/b' in got
    assert 'mkdir -p /opt/a/b/conda' not in got


def test_a_download_creates_the_directory_it_needs():
    got = conda._download('https://x/i.sh', '/opt/a/i.sh')
    assert 'mkdir -p /opt/a' in got and '-o /opt/a/i.sh' in got


def test_paths_and_urls_are_quoted():
    got = conda._download('https://x/i.sh', '/opt/my dir/i.sh')
    assert "'/opt/my dir/i.sh'" in got


# -- installing: what is already there ------------------------------------


def test_state_reports_absent_present_and_active():
    assert conda._parse_state('absent\n') == (False, False)
    assert conda._parse_state('present\n') == (True, False)
    assert conda._parse_state('present\nactive\n') == (True, True)


def test_state_asks_by_activating_not_by_looking():
    """A tree can look exactly like conda and still not run."""
    got = conda._state_query('~/.tether/conda')
    assert 'conda activate base' in got
    assert 'etc/profile.d/conda.sh' in got


class Installer(Remote):
    """A transport that also remembers what state the prefix is in."""

    def __init__(self, state='absent\n', **kw):
        super().__init__(**kw)
        self.state = state

    def __call__(self, command):
        if 'conda activate base' in command:
            self.commands.append(command)
            return self.state
        if command.startswith('uname'):
            self.commands.append(command)
            return 'x86_64\n'
        if command.startswith('curl') and 'api.github.com' in command:
            self.commands.append(command)
            return release_json()
        if command.startswith('curl'):
            self.commands.append(command)
            return 'abc123  ./Miniforge3.sh\n'
        if command.startswith('sha256sum'):
            self.commands.append(command)
            return 'abc123  /tmp/Miniforge3.sh\n'
        return super().__call__(command)


def test_an_existing_working_conda_is_adopted_without_installing():
    """This is what makes calling install() twice harmless."""
    remote = Installer(
        state='present\nactive\n',
        listing='/opt/conda/etc/profile.d/conda.sh\n'
                '/opt/conda/conda-meta/conda-26.7.2-py312_0.json\n',
    )
    got = conda.install(remote, '/opt/conda')
    assert got.version == '26.7.2'
    assert not any('bash' in c for c in remote.commands)


def test_adoption_can_be_refused_rather_than_silently_reused():
    remote = Installer(state='present\nactive\n')
    with pytest.raises(tether.CondaError, match='already installed'):
        conda.install(remote, '/opt/conda', adopt_if_exists=False)


def test_a_prefix_that_exists_but_will_not_activate_is_never_overwritten():
    """A half-extracted tree is exactly what must not be clobbered."""
    remote = Installer(state='present\n')
    with pytest.raises(tether.CondaError, match='does not activate'):
        conda.install(remote, '/opt/conda')
    assert not any('bash' in c or 'rm -f' in c for c in remote.commands)


def test_a_supplied_installer_skips_the_download_entirely():
    remote = Installer(
        listing='/opt/conda/etc/profile.d/conda.sh\n'
                '/opt/conda/conda-meta/conda-26.7.2-py312_0.json\n',
    )
    conda.install(remote, '/opt/conda', installer='/tmp/mine.sh')
    assert not any('api.github.com' in c for c in remote.commands)
    assert any('bash /tmp/mine.sh -b -p /opt/conda' in c for c in remote.commands)


def test_a_supplied_installer_is_trusted_unless_a_hash_is_given():
    remote = Installer(
        listing='/opt/conda/etc/profile.d/conda.sh\n'
                '/opt/conda/conda-meta/conda-26.7.2-py312_0.json\n',
    )
    conda.install(remote, '/opt/conda', installer='/tmp/mine.sh')
    assert not any(c.startswith('sha256sum') for c in remote.commands)


def test_a_mismatched_checksum_stops_before_installing_and_keeps_the_file():
    remote = Installer()
    with pytest.raises(tether.CondaError, match='checksum mismatch'):
        conda.install(remote, '/opt/conda', installer='/tmp/mine.sh', sha256='deadbeef')
    assert not any('bash' in c or 'rm -f' in c for c in remote.commands)


def test_a_downloaded_installer_is_removed_once_it_has_run():
    remote = Installer(
        listing='/opt/conda/etc/profile.d/conda.sh\n'
                '/opt/conda/conda-meta/conda-26.7.2-py312_0.json\n',
    )
    conda.install(remote, '/opt/conda')
    assert any(c.startswith('rm -f') for c in remote.commands)


def test_an_install_that_leaves_nothing_behind_is_an_error():
    """The installer said it worked; the prefix disagrees."""
    remote = Installer(listing='')
    with pytest.raises(tether.CondaError, match='nothing that looks like'):
        conda.install(remote, '/opt/conda', installer='/tmp/mine.sh')


# -- creating conda environments ------------------------------------------
#
# Where a named environment lands is not where you would guess: conda puts it
# under the base when the base is writable and in `~/.conda/envs` when it is
# not. These pin that the answer is asked for rather than assembled.


class EnvRemote:
    """A transport that answers the before-and-after state of an environment."""

    def __init__(self, before='', after=''):
        self.before, self.after = before, after
        self.commands = []
        self.created = False

    def __call__(self, command):
        self.commands.append(command)
        if 'conda create' in command:
            self.created = True
            return ''
        return self.after if self.created else self.before

    @property
    def trips(self):
        return len(self.commands)


def test_an_environment_is_created_and_its_prefix_reported():
    remote = EnvRemote(after='/opt/conda/envs/demo\n')
    assert conda.create_env(remote, 'demo', '/opt/conda') == '/opt/conda/envs/demo'
    assert remote.created


def test_the_prefix_is_asked_for_not_assembled():
    """A named environment created against a base that is not writable lands
    in the per-user fallback instead -- the usual outcome on a site install."""
    remote = EnvRemote(after='/home/u/.conda/envs/demo\n')
    got = conda.create_env(remote, 'demo', '/opt/conda')
    assert got == '/home/u/.conda/envs/demo'
    assert not got.startswith('/opt/conda')


def test_an_existing_environment_is_adopted_without_creating():
    remote = EnvRemote(before='/opt/conda/envs/demo\n')
    assert conda.create_env(remote, 'demo', '/opt/conda') == '/opt/conda/envs/demo'
    assert not remote.created
    assert remote.trips == 1


def test_adoption_can_be_refused():
    remote = EnvRemote(before='/opt/conda/envs/demo\n')
    with pytest.raises(tether.CondaError, match='already exists'):
        conda.create_env(remote, 'demo', '/opt/conda', adopt_if_exists=False)
    assert not remote.created


def test_a_creation_that_does_not_activate_afterwards_is_an_error():
    """conda can report success and leave something that will not enter."""
    remote = EnvRemote(after='')
    with pytest.raises(tether.CondaError, match='does not activate'):
        conda.create_env(remote, 'demo', '/opt/conda')


# -- the commands that get sent -------------------------------------------


def test_the_hook_is_sourced_before_conda_is_used():
    """`conda create` is a shell function; without the hook it does not exist."""
    got = conda._create_env_command('/opt/conda', 'demo')
    assert got.index('etc/profile.d/conda.sh') < got.index('conda create')


def test_creation_does_not_wait_for_a_prompt():
    """There is nobody at the other end to answer it."""
    assert 'conda create -y' in conda._create_env_command('/opt/conda', 'demo')


def test_a_python_version_is_solved_for_not_pointed_at():
    """conda fetches an interpreter; that is the whole reason to ask it."""
    got = conda._create_env_command('/opt/conda', 'demo', python='3.12')
    assert 'python=3.12' in got


def test_no_version_asked_for_leaves_conda_to_choose():
    assert 'python=' not in conda._create_env_command('/opt/conda', 'demo')


def test_package_specifiers_are_quoted():
    """Unquoted, `numpy>=1.20` is a redirection."""
    got = conda._create_env_command('/opt/conda', 'demo', packages=['numpy>=1.20'])
    assert shlex.quote('numpy>=1.20') in got


def test_failures_abort_rather_than_carrying_on():
    """The message names the environment, and survives being quoted into an
    `echo` -- which is why this asserts the quoted form rather than the plain
    one."""
    got = conda._create_env_command('/opt/conda', 'demo')
    assert 'could not source the conda hook' in got
    assert shlex.quote("tether: could not create conda environment 'demo'") in got


def test_setup_runs_before_any_of_it():
    """The conda being used may only be on PATH once a module is loaded."""
    got = conda._create_env_command('/opt/conda', 'demo', setup='module load anaconda')
    assert got.startswith('module load anaconda\n')


def test_the_state_query_activates_rather_than_looking():
    """A directory under envs/ can exist and be half-written."""
    got = conda._env_state_query('/opt/conda', 'demo')
    assert 'conda activate demo' in got and 'CONDA_PREFIX' in got


@pytest.mark.parametrize('answer', ['', '\n', 'EnvironmentNameNotFound: x\n'])
def test_an_environment_that_never_activated_has_no_prefix(answer):
    assert conda._parse_env_prefix(answer) == ''


def test_only_an_absolute_path_counts_as_a_prefix():
    assert conda._parse_env_prefix('warning: ignored\n/opt/conda/envs/x\n') == (
        '/opt/conda/envs/x'
    )
