"""Unit tests for environment activation. No cluster needed -- the module is pure.

These lean on the fact that the generated text *is* the contract: it ends up in
a batch script someone will read by hand while debugging, so the tests assert on
the emitted shell rather than on internal structure.
"""

import json
import pathlib
import shlex
import subprocess

import pytest

import tether
from tether.environment import EnvironmentKind, preamble, with_preamble
from tether.environment import env as make_env
from tether.shell import remote_path


def write_server(config_dir, name='x', **body):
    """Write one server config the way tether will read it back."""
    path = pathlib.Path(config_dir) / 'servers' / f'{name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))
    return path


def env(**kwargs):
    """Build an environment of whatever kind the keywords name."""
    name = kwargs.pop('name', 'test')
    kind = kwargs.pop('kind', EnvironmentKind.NONE)
    return make_env(name, kind, **kwargs)


def preamble_lines(cfg) -> list[str]:
    """The preamble split for tests that assert on order or position.

    Note this splits *physical* lines. A quoted value containing a newline --
    which `shlex.quote` preserves literally -- is one shell statement spread
    over two of them. No config below does that, and the ones that do
    (`HOSTILE`) assert on the whole string instead.
    """
    return preamble(cfg).splitlines()


# -- ordering -------------------------------------------------------------


def test_slots_appear_in_documented_order():
    cfg = env(
        kind=EnvironmentKind.VENV,
        path='/opt/venv',
        modules=('openmpi/4.1.5',),
        pre_activation_cmds=('source /etc/profile.d/modules.sh',),
        post_activation_cmds=('export PYTHONPATH=/late',),
        env={'OMP_NUM_THREADS': '1'},
    )
    got = preamble_lines(cfg)

    # pre_activation_cmds first, post_activation_cmds last, activation between the two.
    assert got[0] == 'source /etc/profile.d/modules.sh'
    assert got[-1] == 'export PYTHONPATH=/late'

    positions = {
        'module': next(i for i, ln in enumerate(got) if ln.startswith('module load')),
        'export': next(i for i, ln in enumerate(got) if ln.startswith('export OMP')),
        'source': next(i for i, ln in enumerate(got) if 'bin/activate' in ln),
    }
    assert positions['module'] < positions['export'] < positions['source']


def test_env_exports_precede_activation():
    """Variables that configure activation must be set before it runs."""
    cfg = env(
        kind=EnvironmentKind.CONDA,
        name='phoebe',
        env={'CONDA_ENVS_PATH': '/scratch/envs'},
    )
    got = preamble_lines(cfg)
    assert got.index('export CONDA_ENVS_PATH=/scratch/envs') < next(
        i for i, ln in enumerate(got) if 'conda activate' in ln
    )


# -- the three kinds ------------------------------------------------------


def test_bare_metal_has_no_activation():
    cfg = env(kind=EnvironmentKind.NONE, modules=('gcc',), env={'X': '1'})
    got = preamble(cfg)
    assert 'activate' not in got
    assert 'module load gcc' in got
    assert 'export X=1' in got


def test_no_environment_at_all_is_empty():
    assert preamble(None) == ''
    assert with_preamble(None, 'python run.py') == 'python run.py'


def test_venv_sources_the_activate_script():
    cfg = env(kind=EnvironmentKind.VENV, path='/opt/venv')
    assert 'source /opt/venv/bin/activate' in preamble(cfg)


def test_venv_tolerates_a_trailing_slash():
    cfg = env(kind=EnvironmentKind.VENV, path='/opt/venv/')
    assert '/opt/venv/bin/activate' in preamble(cfg)
    assert '//bin' not in preamble(cfg)


def test_conda_checks_for_conda_then_sources_the_hook_then_activates():
    """The presence check must be its own line: `eval "$(conda ...)"` reports
    the status of the evaluated string, so a missing conda would slip past a
    guard on the eval itself."""
    cfg = env(kind=EnvironmentKind.CONDA, conda_env='phoebe-dev')
    got = preamble_lines(cfg)
    assert 'command -v conda' in got[0]
    assert 'conda is not on PATH' in got[0]
    assert 'eval "$(conda shell.bash hook)"' in got[1]
    assert 'conda activate phoebe-dev' in got[2]


def test_conda_base_sources_the_hook_directly():
    cfg = env(kind=EnvironmentKind.CONDA, conda_env='phoebe', conda_base='/opt/conda')
    got = preamble_lines(cfg)
    assert 'source /opt/conda/etc/profile.d/conda.sh' in got[0]
    assert 'conda shell.bash hook' not in got[0]


def test_conda_env_defaults_to_the_environment_name():
    """For conda the two are usually the same word, so saying it twice is
    noise. A venv `path` is a path, so it is never guessed -- see below."""
    assert env(kind=EnvironmentKind.CONDA).conda_env == 'test'
    assert 'conda activate test' in preamble(env(kind=EnvironmentKind.CONDA))


def test_venv_without_a_path_is_refused_at_construction():
    """Not at use: an invalid environment must never reach a config file."""
    with pytest.raises(tether.ConfigError, match="a venv needs 'path'"):
        env(kind=EnvironmentKind.VENV)


def test_unknown_kind_is_an_error():
    with pytest.raises(tether.ConfigError, match='unsupported kind'):
        env(kind='poetry', conda_env='x')


def test_fields_of_another_kind_are_refused():
    """`conda_base` is not a field on a venv, so the class rejects it without
    anyone writing a check for it."""
    with pytest.raises(tether.ConfigError, match='not valid for kind'):
        env(kind=EnvironmentKind.VENV, path='/opt/v', conda_base='/opt/conda')


# -- failure is loud ------------------------------------------------------


@pytest.mark.parametrize(
    'cfg',
    [
        env(kind=EnvironmentKind.VENV, path='/opt/venv'),
        env(kind=EnvironmentKind.CONDA, name='phoebe'),
        env(kind=EnvironmentKind.NONE, modules=('gcc',)),
    ],
    ids=['venv', 'conda', 'module'],
)
def test_every_fallible_step_is_guarded(cfg):
    """A silent activation failure would run the payload against the wrong
    interpreter, so each step must abort instead of continuing."""
    for line in preamble_lines(cfg):
        assert 'exit 1' in line, f'unguarded step: {line}'
        assert 'tether:' in line


# -- quoting and injection ------------------------------------------------


HOSTILE = ['x; rm -rf /', 'x$(whoami)', 'x`id`', "x'y", 'x&&y', 'x|y', 'x\nrm -rf /']


@pytest.mark.parametrize('hostile', HOSTILE)
def test_venv_path_stays_one_quoted_word(hostile):
    got = preamble(env(kind=EnvironmentKind.VENV, path=hostile))
    assert f'source {remote_path(hostile.rstrip("/") + "/bin/activate")}' in got


@pytest.mark.parametrize('hostile', HOSTILE)
def test_env_values_are_quoted(hostile):
    got = preamble(env(env={'VAR': hostile}))
    assert got == f'export VAR={shlex.quote(hostile)}'


# Asserting on the generated text only proves it looks right. These two run it.


@pytest.mark.parametrize('template', ['/nope; rm -f {canary}', '/nope$(rm -f {canary})'])
def test_hostile_venv_path_cannot_execute(tmp_path, template):
    canary = tmp_path / 'canary'
    canary.write_text('alive')
    script = preamble(env(kind=EnvironmentKind.VENV, path=template.format(canary=canary)))

    # check=False: a nonzero exit is the expected outcome here.
    done = subprocess.run(
        ['bash', '-c', script], capture_output=True, text=True, check=False
    )

    assert done.returncode == 1                 # the guard fired
    assert 'tether:' in done.stderr             # ...and said so
    assert canary.read_text() == 'alive'        # the injection did not run


def test_hostile_env_value_round_trips_without_executing(tmp_path):
    canary = tmp_path / 'canary'
    canary.write_text('alive')
    hostile = f'; rm -f {canary}'
    script = preamble(env(env={'VAR': hostile}))

    done = subprocess.run(
        ['bash', '-c', f'{script}\nprintf %s "$VAR"'],
        capture_output=True,
        text=True,
        check=True,
    )

    assert done.stdout == hostile               # exact value, nothing eaten
    assert canary.read_text() == 'alive'


@pytest.mark.parametrize(
    'bad', ['FOO BAR', '1FOO', 'FOO;rm -rf /', 'FOO=BAR', '', 'FOO-BAR']
)
def test_env_names_must_be_shell_identifiers(bad):
    with pytest.raises(tether.ConfigError, match='not a usable shell variable'):
        preamble(env(env={bad: '1'}))


def test_module_names_are_quoted():
    got = preamble(env(modules=('openmpi/4.1.5', 'a b')))
    assert 'module load openmpi/4.1.5' in got
    assert "module load 'a b'" in got


def test_verbatim_slots_are_not_quoted():
    """The two verbatim slots are raw shell on purpose -- that is their whole job."""
    cfg = env(
        pre_activation_cmds=('export PATH="$PATH:/opt/bin"',),
        post_activation_cmds=('cd "$SLURM_SUBMIT_DIR"',),
    )
    got = preamble(cfg)
    assert 'export PATH="$PATH:/opt/bin"' in got
    assert 'cd "$SLURM_SUBMIT_DIR"' in got


# -- with_preamble -----------------------------------------------------------------


def test_wrap_puts_the_command_last():
    cfg = env(kind=EnvironmentKind.VENV, path='/opt/venv')
    got = with_preamble(cfg, 'python run.py')
    assert got.endswith('\npython run.py')
    assert got.startswith('source /opt/venv/bin/activate')


def test_wrap_without_a_preamble_is_the_bare_command():
    assert with_preamble(env(), 'python run.py') == 'python run.py'


# -- config plumbing -----------------------------------------------------


def test_config_reads_the_new_keys(tmp_path):
    write_server(tmp_path, 'x', environments={
        'phoebe': {
            'kind': 'conda',
            'conda_env': 'phoebe-dev',
            'conda_base': '/opt/conda',
            'pre_activation_cmds': ['source /etc/profile.d/modules.sh'],
            'post_activation_cmds': ['echo late'],
        }
    })
    cfg = tether.server('x', config_dir=tmp_path).environments['phoebe']
    assert cfg.conda_base == '/opt/conda'
    assert cfg.pre_activation_cmds == ('source /etc/profile.d/modules.sh',)
    assert cfg.post_activation_cmds == ('echo late',)


def test_superseded_key_names_are_rejected_not_ignored(tmp_path):
    """`bootstrap` and `prelude` were the earlier names for the two verbatim
    slots. `_reject_unknown` derives the allowed keys from
    `__dataclass_fields__`, so an old config fails loudly instead of silently
    dropping the lines it asked for -- and JSON keys can never drift from the
    field names."""
    for superseded in ('bootstrap', 'prelude'):
        write_server(tmp_path, 'x',
                     environments={'e': {superseded: ['echo hi']}})
        with pytest.raises(tether.ConfigError, match='not valid for kind'):
            tether.server('x', config_dir=tmp_path)


def test_conda_base_on_a_non_conda_environment_is_loud(tmp_path):
    write_server(tmp_path, 'x', environments={
        'e': {'kind': 'venv', 'path': '/opt/v', 'conda_base': '/opt/conda'}
    })
    with pytest.raises(tether.ConfigError, match='not valid for kind'):
        tether.server('x', config_dir=tmp_path)


def test_server_exposes_the_preamble(tmp_path):
    write_server(tmp_path, 'x',
                 default_environment='e',
                 environments={'e': {'kind': 'venv', 'path': '~/venvs/phoebe'}})
    srv = tether.server('x', config_dir=tmp_path)
    assert 'source "$HOME/venvs/phoebe/bin/activate"' in srv.preamble


def test_server_without_environment_has_an_empty_preamble(tmp_path):
    srv = tether.server('nonexistent.invalid', config_dir=tmp_path)
    assert srv.preamble == ''


# -- proof that activation took effect ------------------------------------
#
# Exit status cannot see an `activate` that ran and did nothing, so each kind
# names its own evidence. These need no machine: the probe's reported values
# are just a dict.


def test_bare_metal_has_nothing_to_prove():
    """No activation was attempted, so none can have failed."""
    assert env().activation_failure({}) is None


def test_a_venv_must_show_virtual_env():
    cfg = env(kind=EnvironmentKind.VENV, path='/opt/venv')
    assert cfg.activation_failure({'VIRTUAL_ENV': '/opt/venv'}) is None
    assert cfg.activation_failure({'VIRTUAL_ENV': ''}) == '$VIRTUAL_ENV is unset'
    assert cfg.activation_failure({}) == '$VIRTUAL_ENV is unset'


def test_a_conda_env_must_show_conda_prefix():
    cfg = env(kind=EnvironmentKind.CONDA)
    assert cfg.activation_failure({'CONDA_PREFIX': '/opt/conda/envs/x'}) is None
    assert cfg.activation_failure({'CONDA_PREFIX': ''}) == '$CONDA_PREFIX is unset'


def test_each_kind_ignores_the_other_kind_variable():
    """A venv that set CONDA_PREFIX, or vice versa, proves nothing."""
    venv = env(kind=EnvironmentKind.VENV, path='/opt/venv')
    conda = env(kind=EnvironmentKind.CONDA)
    assert venv.activation_failure({'CONDA_PREFIX': '/opt/conda'}) is not None
    assert conda.activation_failure({'VIRTUAL_ENV': '/opt/venv'}) is not None
