"""Unit tests for the parts of tether that need no cluster."""

import json
import pathlib
from datetime import timedelta

import pytest

import tether
from tether.slurm import parse_duration, parse_sinfo, parse_squeue


def write_server(config_dir, name='a', **body):
    """Write one server config the way tether will read it back."""
    path = pathlib.Path(config_dir) / 'servers' / f'{name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))
    return path


@pytest.mark.parametrize(
    'raw,expected',
    [
        ('30', timedelta(seconds=30)),
        ('5:30', timedelta(minutes=5, seconds=30)),
        ('1:05:30', timedelta(hours=1, minutes=5, seconds=30)),
        ('2-03', timedelta(days=2, hours=3)),
        ('2-03:04', timedelta(days=2, hours=3, minutes=4)),
        ('2-03:04:05', timedelta(days=2, hours=3, minutes=4, seconds=5)),
        ('  10:00  ', timedelta(minutes=10)),
        ('UNLIMITED', None),
        ('INVALID', None),
        ('N/A', None),
        ('', None),
        ('nonsense', None),
    ],
)
def test_parse_duration(raw, expected):
    assert parse_duration(raw) == expected


def test_squeue_parses_realistic_output():
    out = (
        '12345|RUNNING|main|andrej|2|48|1-02:03:04|3-00:00:00|node[01-02]|'
        '/home/andrej/run|phoebe-fit\n'
        '12346|PENDING|main|kelly|1|24|0:00|1:00:00|Resources|'
        '/home/kelly/x|glaze-sim\n'
        '12347_3|COMPLETING|gpu|andrej|1|8|10:00|UNLIMITED|node05|'
        '/tmp|array task\n'
    )
    jobs = parse_squeue(out)
    assert len(jobs) == 3

    a, b, c = jobs
    assert (a.jobid, a.name, a.user) == ('12345', 'phoebe-fit', 'andrej')
    assert a.nodes == 2 and a.cpus == 48
    assert a.elapsed == timedelta(days=1, hours=2, minutes=3, seconds=4)
    assert a.is_running and not a.is_pending and not a.is_finished

    assert b.is_pending and b.reason == 'Resources'
    assert b.elapsed == timedelta(0)

    assert c.jobid == '12347_3'        # array task id survives
    assert c.timelimit is None          # UNLIMITED is not zero
    assert c.is_running                 # COMPLETING counts as active


def test_squeue_absorbs_pipe_in_job_name():
    """Name is the last field, so a `|` inside it must not shift the others."""
    out = '99|RUNNING|main|andrej|1|4|1:00|2:00|node01|/home/andrej|a|b\n'
    (job,) = parse_squeue(out)
    assert job.jobid == '99'
    assert job.workdir == '/home/andrej'
    assert job.name == 'a|b'


def test_squeue_skips_malformed_lines():
    out = '12345|RUNNING|main\n' + '\n' + '1|RUNNING|m|u|1|1|0:01|1:00|n|/w|ok\n'
    (job,) = parse_squeue(out)
    assert job.name == 'ok'


def test_sinfo_parses_and_computes_load():
    out = (
        'main*|up|7-00:00:00|48|12/4/0/16|node[01-16]\n'
        'gpu|down|1-00:00:00|64|0/2/1/3|gpu[01-03]\n'
    )
    main, gpu = parse_sinfo(out)

    assert main.name == 'main' and main.is_default and main.is_up
    assert (main.nodes_allocated, main.nodes_idle, main.nodes_total) == (12, 4, 16)
    assert main.load == pytest.approx(0.75)
    assert main.timelimit == timedelta(days=7)
    assert main.cpus_per_node == 48

    assert not gpu.is_default and not gpu.is_up
    assert gpu.nodes_other == 1


def test_no_config_is_not_an_error(tmp_path):
    """An unconfigured name is a hostname, not a mistake."""
    assert tether.list_servers(tmp_path) == []
    assert tether.server('anything', config_dir=tmp_path).host == 'anything'


def test_config_roundtrip(tmp_path):
    write_server(
        tmp_path,
        'terra',
        host='terra.villanova.edu',
        user='andrej',
        workdir='~/.tether',
        default_environment='phoebe',
        environments={
            'phoebe': {
                'kind': 'conda',
                'conda_env': 'phoebe-dev',
                'modules': ['openmpi/4.1.5'],
                'env': {'OMP_NUM_THREADS': '1'},
            }
        },
    )
    write_server(tmp_path, 'laptop', kind='plain')

    terra = tether.server('terra', config_dir=tmp_path)
    assert terra.host == 'terra.villanova.edu'
    assert terra.kind == 'slurm'                        # default
    assert terra.environments['phoebe'].modules == ('openmpi/4.1.5',)
    assert terra.environments['phoebe'].env == {'OMP_NUM_THREADS': '1'}

    # The filename is the identity, and the host falls back to it.
    assert tether.server('laptop', config_dir=tmp_path).host == 'laptop'
    assert tether.list_servers(tmp_path) == ['laptop', 'terra']


@pytest.mark.parametrize(
    'body,fragment',
    [
        ({'hostt': 'x'}, 'unknown key'),
        ({'kind': 'pbs'}, 'kind must be one of'),
        ({'default_environment': 'nope'}, 'not among the environments'),
        ({'environments': {'e': {'kind': 'venv'}}}, "a venv needs 'path'"),
        ({'environments': {'e': {'kind': 'venv', 'path': 'v',
                                 'conda_base': '/c'}}}, 'not valid for kind'),
        ({'environments': {'e': {'kind': 'poetry'}}}, 'unsupported kind'),
        ({'environments': []}, 'must be an object'),
    ],
)
def test_config_errors_are_loud(tmp_path, body, fragment):
    write_server(tmp_path, 'a', **body)
    with pytest.raises(tether.ConfigError) as exc:
        tether.server('a', config_dir=tmp_path)
    assert fragment in str(exc.value)


def test_malformed_json_names_the_file(tmp_path):
    path = write_server(tmp_path, 'a')
    path.write_text('{not json')
    with pytest.raises(tether.ConfigError, match='a.json'):
        tether.server('a', config_dir=tmp_path)


def test_server_needs_a_host(tmp_path):
    with pytest.raises(tether.ConfigError):
        tether.Server(config_dir=str(tmp_path))


def test_server_construction_does_no_io(tmp_path):
    s = tether.server('nonexistent.invalid', config_dir=str(tmp_path))
    assert isinstance(s, tether.SlurmServer)   # slurm is the default kind
    assert s.host == 'nonexistent.invalid'
    assert not s.connected                     # lazy: nothing opened


def test_server_kind_is_normalized_to_strenum(tmp_path):
    plain = tether.server('localhost', kind='plain', config_dir=str(tmp_path))
    assert isinstance(plain, tether.Server)
    assert plain.host == 'localhost'

    slurm = tether.server('localhost', kind=tether.ServerKind.SLURM, config_dir=str(tmp_path))
    assert isinstance(slurm, tether.SlurmServer)


def test_spec_formats_are_stable():
    from tether.slurm import SINFO_SPEC, SQUEUE_SPEC, spec_format
    assert spec_format(SQUEUE_SPEC) == '%i|%T|%P|%u|%D|%C|%M|%l|%R|%Z|%j'
    assert spec_format(SINFO_SPEC) == '%P|%a|%l|%c|%F|%N'
    assert list(SQUEUE_SPEC)[-1] == 'name'   # name last: absorbs an embedded |


# -- phase 0: timeouts are configurable -----------------------------------


def test_timeout_defaults_to_a_generous_value(tmp_path):
    srv = tether.server('nowhere.invalid', config_dir=tmp_path)
    assert srv.timeout == tether.link.DEFAULT_TIMEOUT
    assert srv.timeout >= 600      # long enough for a conda install


def test_timeout_comes_from_the_config_file(tmp_path):
    write_server(tmp_path, 'a', timeout=120)
    assert tether.server('a', config_dir=tmp_path).timeout == 120.0
    assert tether.server('a', config_dir=tmp_path).timeout == 120.0


def test_explicit_timeout_beats_the_config_file(tmp_path):
    write_server(tmp_path, 'a', timeout=120)
    assert tether.server('a', config_dir=tmp_path, timeout=5).timeout == 5.0


@pytest.mark.parametrize('bad', ['soon', 0, -1, True])
def test_nonsense_timeouts_are_loud(tmp_path, bad):
    write_server(tmp_path, 'a', timeout=bad)
    with pytest.raises(tether.ConfigError, match='timeout must be'):
        tether.server('a', config_dir=tmp_path)


# -- link injection --------------------------------------------------------


def test_an_injected_link_is_used_as_is(tmp_path):
    """The seam a local (non-SSH) transport will plug into."""
    mine = tether.Link('elsewhere.invalid', 'someone')
    srv = tether.server('nowhere.invalid', config_dir=tmp_path, link=mine)
    assert srv.link is mine
    assert not srv.connected          # still lazy; nothing opened


def test_without_injection_a_link_is_built_from_the_config(tmp_path):
    write_server(tmp_path, 'a', host='a.invalid', user='kelly')
    srv = tether.server('a', config_dir=tmp_path)
    assert srv.link.host == 'a.invalid'
    assert srv.link.user == 'kelly'


# -- writing config --------------------------------------------------------


def test_save_then_load_round_trips(tmp_path):
    srv = tether.server('terra', host='terra.villanova.edu', user='andrej',
                        timeout=120.0, config_dir=tmp_path)
    srv.add_environment(tether.CondaEnvironment(
        'phoebe',
        conda_base='/opt/conda',
        pre_activation=('source "$HOME/hook.sh"',),   # quotes must survive
        env={'OMP_NUM_THREADS': '1'},
    ))
    path = srv.save(config_dir=tmp_path)
    assert path == tmp_path / 'servers' / 'terra.json'

    back = tether.server('terra', config_dir=tmp_path)
    assert back.to_dict() == srv.to_dict()
    assert back.environments == srv.environments
    assert back.default_environment == 'phoebe'
    assert back.timeout == 120.0


def test_saved_files_omit_defaults(tmp_path):
    """A saved file should show what was chosen, not a transcript of every
    default in force the day it was written."""
    tether.server('a', host='a.invalid', config_dir=tmp_path).save(config_dir=tmp_path)
    body = json.loads((tmp_path / 'servers' / 'a.json').read_text())
    assert body['host'] == 'a.invalid'
    assert body['tether'] == tether.__version__
    for defaulted in ('workdir', 'timeout', 'user', 'environments'):
        assert defaulted not in body


def test_save_refuses_to_clobber(tmp_path):
    srv = tether.server('a', host='one.invalid', config_dir=tmp_path)
    srv.save(config_dir=tmp_path)
    with pytest.raises(tether.ConfigError, match='already exists'):
        srv.save(config_dir=tmp_path)

    tether.server('a', host='two.invalid', config_dir=tmp_path).save(
        config_dir=tmp_path, overwrite=True)
    assert tether.server('a', config_dir=tmp_path).host == 'two.invalid'


def test_delete_server(tmp_path):
    tether.server('a', config_dir=tmp_path).save(config_dir=tmp_path)
    assert tether.list_servers(tmp_path) == ['a']
    assert tether.delete_server('a', config_dir=tmp_path) is True
    assert tether.list_servers(tmp_path) == []
    assert tether.delete_server('a', config_dir=tmp_path) is False


@pytest.mark.parametrize('bad', ['../escape', 'a/b', '.', '..', '', 'has space'])
def test_names_that_would_escape_the_directory_are_refused(bad, tmp_path):
    """The name becomes a filename, so traversal has to be impossible."""
    with pytest.raises(tether.ConfigError, match='not a usable server name'):
        tether.server_path(bad, tmp_path)


def test_an_unloadable_config_cannot_be_built_let_alone_saved(tmp_path):
    """Validation moved into the classes, so the round trip cannot be broken.

    Before, `EnvironmentConfig(label='e', kind='conda')` constructed happily,
    `save_server` wrote it, and `load_server` then refused it -- a file tether
    had produced and could not read.
    """
    with pytest.raises(tether.ConfigError):
        tether.VenvEnvironment('e')            # no name: refused up front

    # And whatever does construct, survives the trip.
    for env in (
        tether.SystemEnvironment('bare', modules=['gcc']),
        tether.VenvEnvironment('dev', path='~/venvs/dev'),
        tether.CondaEnvironment('phoebe', conda_base='/opt/conda'),
    ):
        # A fresh server per kind: reusing one would reload what the previous
        # iteration saved and accumulate environments.
        srv = tether.server(f'srv-{env.kind}', host='h', config_dir=tmp_path)
        srv.add_environment(env)
        srv.save(config_dir=tmp_path, overwrite=True)
        loaded = tether.server(f'srv-{env.kind}', config_dir=tmp_path)
        assert loaded.environments == {env.name: env}


def test_lists_and_tuples_compare_equal(tmp_path):
    """JSON gives lists, the fields are tuples; a constructed environment must
    still equal the same one loaded back."""
    assert (tether.SystemEnvironment('a', modules=['gcc'])
            == tether.SystemEnvironment('a', modules=('gcc',)))
