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
from tether.config import EnvironmentConfig, EnvironmentKind
from tether.environment import create_preamble, remote_path, wrap


def write_server(config_dir, name='x', **body):
    """Write one server config the way tether will read it back."""
    path = pathlib.Path(config_dir) / 'servers' / f'{name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))
    return path


def env(**kwargs) -> EnvironmentConfig:
    kwargs.setdefault('label', 'test')
    return EnvironmentConfig(**kwargs)


def preamble_lines(cfg) -> list[str]:
    """The preamble split for tests that assert on order or position.

    Note this splits *physical* lines. A quoted value containing a newline --
    which `shlex.quote` preserves literally -- is one shell statement spread
    over two of them. No config below does that, and the ones that do
    (`HOSTILE`) assert on the whole string instead.
    """
    return create_preamble(cfg).splitlines()


# -- ordering -------------------------------------------------------------


def test_slots_appear_in_documented_order():
    cfg = env(
        kind=EnvironmentKind.VENV,
        name='/opt/venv',
        modules=('openmpi/4.1.5',),
        pre_activation=('source /etc/profile.d/modules.sh',),
        post_activation=('export PYTHONPATH=/late',),
        env={'OMP_NUM_THREADS': '1'},
    )
    got = preamble_lines(cfg)

    # pre_activation first, post_activation last, activation between the two.
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
    got = create_preamble(cfg)
    assert 'activate' not in got
    assert 'module load gcc' in got
    assert 'export X=1' in got


def test_no_environment_at_all_is_empty():
    assert create_preamble(None) == ''
    assert wrap(None, 'python run.py') == 'python run.py'


def test_venv_sources_the_activate_script():
    cfg = env(kind=EnvironmentKind.VENV, name='/opt/venv')
    assert 'source /opt/venv/bin/activate' in create_preamble(cfg)


def test_venv_tolerates_a_trailing_slash():
    cfg = env(kind=EnvironmentKind.VENV, name='/opt/venv/')
    assert '/opt/venv/bin/activate' in create_preamble(cfg)
    assert '//bin' not in create_preamble(cfg)


def test_conda_checks_for_conda_then_sources_the_hook_then_activates():
    """The presence check must be its own line: `eval "$(conda ...)"` reports
    the status of the evaluated string, so a missing conda would slip past a
    guard on the eval itself."""
    cfg = env(kind=EnvironmentKind.CONDA, name='phoebe-dev')
    got = preamble_lines(cfg)
    assert 'command -v conda' in got[0]
    assert 'conda is not on PATH' in got[0]
    assert 'eval "$(conda shell.bash hook)"' in got[1]
    assert 'conda activate phoebe-dev' in got[2]


def test_conda_base_sources_the_hook_directly():
    cfg = env(kind=EnvironmentKind.CONDA, name='phoebe', conda_base='/opt/conda')
    got = preamble_lines(cfg)
    assert 'source /opt/conda/etc/profile.d/conda.sh' in got[0]
    assert 'conda shell.bash hook' not in got[0]


def test_conda_without_name_is_an_error():
    cfg = env(kind=EnvironmentKind.CONDA, name=None)
    with pytest.raises(tether.ConfigError, match="requires 'name'"):
        create_preamble(cfg)


def test_unknown_kind_is_an_error():
    cfg = env(kind='poetry', name='x')
    with pytest.raises(tether.ConfigError, match='unsupported kind'):
        create_preamble(cfg)


def test_unknown_kind_is_reported_as_such_even_without_a_name():
    """The kind is the actual problem; do not blame the missing name."""
    with pytest.raises(tether.ConfigError, match='unsupported kind'):
        create_preamble(env(kind='poetry'))


# -- failure is loud ------------------------------------------------------


@pytest.mark.parametrize(
    'cfg',
    [
        env(kind=EnvironmentKind.VENV, name='/opt/venv'),
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


def test_tilde_becomes_home_because_quoting_defeats_expansion():
    # shlex.quote('~/x') -> "'~/x'", and the quotes stop the shell expanding ~.
    assert remote_path('~/envs/phoebe') == '"$HOME/envs/phoebe"'
    assert remote_path('~') == '"$HOME"'


def test_absolute_and_relative_paths_are_quoted_normally():
    assert remote_path('/opt/venv') == '/opt/venv'
    assert remote_path('/opt/my venv') == "'/opt/my venv'"


def test_tilde_user_is_quoted_rather_than_mangled():
    """`~kelly` has no $HOME equivalent; quote it and fail loudly."""
    assert remote_path('~kelly/env') == "'~kelly/env'"


def test_double_quote_context_is_escaped():
    """Inside "$HOME/...", these four characters keep their meaning."""
    got = remote_path('~/a"b$c`d\\e')
    assert got == '"$HOME/a\\"b\\$c\\`d\\\\e"'


HOSTILE = ['x; rm -rf /', 'x$(whoami)', 'x`id`', "x'y", 'x&&y', 'x|y', 'x\nrm -rf /']


@pytest.mark.parametrize('hostile', HOSTILE)
def test_venv_path_stays_one_quoted_word(hostile):
    got = create_preamble(env(kind=EnvironmentKind.VENV, name=hostile))
    assert f'source {remote_path(hostile.rstrip("/") + "/bin/activate")}' in got


@pytest.mark.parametrize('hostile', HOSTILE)
def test_env_values_are_quoted(hostile):
    got = create_preamble(env(env={'VAR': hostile}))
    assert got == f'export VAR={shlex.quote(hostile)}'


# Asserting on the generated text only proves it looks right. These two run it.


@pytest.mark.parametrize('template', ['/nope; rm -f {canary}', '/nope$(rm -f {canary})'])
def test_hostile_venv_path_cannot_execute(tmp_path, template):
    canary = tmp_path / 'canary'
    canary.write_text('alive')
    script = create_preamble(env(kind=EnvironmentKind.VENV, name=template.format(canary=canary)))

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
    script = create_preamble(env(env={'VAR': hostile}))

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
        create_preamble(env(env={bad: '1'}))


def test_module_names_are_quoted():
    got = create_preamble(env(modules=('openmpi/4.1.5', 'a b')))
    assert 'module load openmpi/4.1.5' in got
    assert "module load 'a b'" in got


def test_verbatim_slots_are_not_quoted():
    """The two verbatim slots are raw shell on purpose -- that is their whole job."""
    cfg = env(
        pre_activation=('export PATH="$PATH:/opt/bin"',),
        post_activation=('cd "$SLURM_SUBMIT_DIR"',),
    )
    got = create_preamble(cfg)
    assert 'export PATH="$PATH:/opt/bin"' in got
    assert 'cd "$SLURM_SUBMIT_DIR"' in got


# -- wrap -----------------------------------------------------------------


def test_wrap_puts_the_command_last():
    cfg = env(kind=EnvironmentKind.VENV, name='/opt/venv')
    got = wrap(cfg, 'python run.py')
    assert got.endswith('\npython run.py')
    assert got.startswith('source /opt/venv/bin/activate')


def test_wrap_without_a_preamble_is_the_bare_command():
    assert wrap(env(), 'python run.py') == 'python run.py'


# -- config plumbing -----------------------------------------------------


def test_config_reads_the_new_keys(tmp_path):
    write_server(tmp_path, 'x', environments={
        'phoebe': {
            'kind': 'conda',
            'name': 'phoebe-dev',
            'conda_base': '/opt/conda',
            'pre_activation': ['source /etc/profile.d/modules.sh'],
            'post_activation': ['echo late'],
        }
    })
    cfg = tether.load_server('x', tmp_path).environments['phoebe']
    assert cfg.conda_base == '/opt/conda'
    assert cfg.pre_activation == ('source /etc/profile.d/modules.sh',)
    assert cfg.post_activation == ('echo late',)


def test_superseded_key_names_are_rejected_not_ignored(tmp_path):
    """`bootstrap` and `prelude` were the earlier names for the two verbatim
    slots. `_reject_unknown` derives the allowed keys from
    `__dataclass_fields__`, so an old config fails loudly instead of silently
    dropping the lines it asked for -- and JSON keys can never drift from the
    field names."""
    for superseded in ('bootstrap', 'prelude'):
        write_server(tmp_path, 'x',
                     environments={'e': {superseded: ['echo hi']}})
        with pytest.raises(tether.ConfigError, match='unknown key'):
            tether.load_server('x', tmp_path)


def test_conda_base_on_a_non_conda_environment_is_loud(tmp_path):
    write_server(tmp_path, 'x', environments={
        'e': {'kind': 'venv', 'name': '/opt/v', 'conda_base': '/opt/conda'}
    })
    with pytest.raises(tether.ConfigError, match='only meaningful for'):
        tether.load_server('x', tmp_path)


def test_server_exposes_the_preamble(tmp_path):
    write_server(tmp_path, 'x',
                 default_environment='e',
                 environments={'e': {'kind': 'venv', 'name': '~/venvs/phoebe'}})
    srv = tether.server('x', config_dir=tmp_path)
    assert 'source "$HOME/venvs/phoebe/bin/activate"' in srv.preamble


def test_server_without_environment_has_an_empty_preamble(tmp_path):
    srv = tether.server('nonexistent.invalid', config_dir=tmp_path)
    assert srv.preamble == ''
