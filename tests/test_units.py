"""Unit tests for the parts of tether that need no cluster."""

import json
import pathlib
import shlex
from datetime import UTC, datetime, timedelta

import pytest

import tether
from tether.slurm import (
    SACCT_SPEC,
    batch_script,
    check_name,
    claim_command,
    distill_state,
    job_directory,
    parse_claim,
    parse_duration,
    parse_exit_code,
    parse_sacct,
    parse_scontrol,
    parse_sinfo,
    parse_squeue,
    parse_submit,
    sacct_command,
    scontrol_command,
    submit_command,
)


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
        pre_activation_cmds=('source "$HOME/hook.sh"',),   # quotes must survive
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


# -- batch scripts ---------------------------------------------------------
#
# The script is the contract: someone will read it by hand while debugging a
# job that failed at 3am, so these assert on the emitted text rather than on
# internal structure.


def test_a_job_owns_a_directory_named_before_it_has_an_id():
    """`--chdir` needs a directory, and the jobid does not exist until sbatch
    has already been told where to run."""
    at = job_directory('~/.tether/jobs', 'fit', datetime(2026, 9, 19, 14, 30, 52, tzinfo=UTC))
    assert at == '~/.tether/jobs/fit.20260919-143052'


def test_a_trailing_slash_on_the_base_does_not_double():
    assert job_directory('/jobs/', 'fit', datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)) == (
        '/jobs/fit.20260102-030405'
    )


@pytest.mark.parametrize('name', ['fit', 'run-2', 'a.b', 'a_b', 'Fit9'])
def test_usable_job_names(name):
    check_name(name)


@pytest.mark.parametrize('name', ['', 'a/b', '../escape', 'a b', 'a;b', 'a$b', '*'])
def test_a_job_name_that_would_not_survive_being_a_directory(name):
    with pytest.raises(tether.SlurmError, match='usable job name'):
        check_name(name)


def test_the_script_is_sbatch_then_environment_then_payload():
    """The order is the whole point: directives are read before the job runs,
    and the payload must not start before the environment is in place."""
    got = batch_script('python run.py', name='fit', preamble='source /opt/v/bin/activate')
    lines = [ln for ln in got.splitlines() if ln.strip()]
    assert lines[0] == '#!/bin/bash'
    assert lines.index('source /opt/v/bin/activate') > lines.index('#SBATCH --job-name=fit')
    assert lines[-1] == 'python run.py'


def test_options_not_asked_for_are_not_emitted():
    """Absent means "Slurm's default", which is not the same as any value we
    could invent for it."""
    got = batch_script('true', name='fit')
    for absent in ('--partition', '--nodes', '--cpus-per-task', '--time', '--mem'):
        assert absent not in got


def test_options_use_slurm_names_not_tether_ones():
    got = batch_script('true', name='fit', cpus=8, memory='4G', nodes=2,
                       partition='intel', time='01:00:00')
    assert '#SBATCH --cpus-per-task=8' in got
    assert '#SBATCH --mem=4G' in got
    assert '#SBATCH --nodes=2' in got
    assert '#SBATCH --partition=intel' in got
    assert '#SBATCH --time=01:00:00' in got


def test_output_lands_beside_the_script():
    """Relative, because --chdir puts the job in its own directory."""
    got = batch_script('true', name='fit')
    assert '#SBATCH --output=stdout' in got and '#SBATCH --error=stderr' in got


def test_unanticipated_directives_pass_straight_through():
    got = batch_script('true', name='fit', directives={'gres': 'gpu:1', 'account': 'phoebe'})
    assert '#SBATCH --gres=gpu:1' in got and '#SBATCH --account=phoebe' in got


def test_a_valueless_directive_becomes_a_bare_flag():
    got = batch_script('true', name='fit', directives={'exclusive': ''})
    assert '#SBATCH --exclusive\n' in got
    assert '--exclusive=' not in got


def test_directive_values_are_not_quoted():
    """`#SBATCH` lines are read by Slurm, not by a shell. Quoting them would
    make the quotes part of the value."""
    got = batch_script('true', name='fit', directives={'comment': 'a b'})
    assert '#SBATCH --comment=a b' in got


def test_no_environment_means_no_blank_preamble_section():
    got = batch_script('true', name='fit')
    assert '\n\n\n' not in got


def test_a_multi_line_payload_is_kept_whole():
    got = batch_script('one\ntwo\nthree', name='fit')
    assert got.endswith('one\ntwo\nthree\n')


def test_the_script_refuses_a_name_that_is_not_a_directory():
    with pytest.raises(tether.SlurmError, match='usable job name'):
        batch_script('true', name='../escape')


# -- submitting ------------------------------------------------------------


def test_submission_asks_for_a_parsable_answer_in_the_right_directory():
    got = submit_command('/jobs/fit.20260919-143052')
    assert '--parsable' in got
    assert '--chdir=/jobs/fit.20260919-143052' in got
    assert got.endswith('/job.sh')


def test_a_directory_with_a_space_is_quoted():
    got = submit_command('/my jobs/fit.1')
    assert "'/my jobs/fit.1'" in got


def test_the_job_id_comes_back_from_parsable():
    assert parse_submit('12345\n') == '12345'


def test_a_federated_cluster_suffix_is_dropped():
    """`--parsable` prints `jobid;cluster` on a federation, and the cluster
    name would poison every later lookup."""
    assert parse_submit('12345;cluster\n') == '12345'


@pytest.mark.parametrize('noise', ['', '\n', 'Submitted batch job 5\n', 'error: nope\n'])
def test_anything_that_is_not_a_job_id_is_loud(noise):
    with pytest.raises(tether.SlurmError, match='did not return a job id'):
        parse_submit(noise)


def test_a_submission_describes_itself_for_a_human():
    s = tether.Submission(jobid='626', directory='/jobs/fit.1', name='fit', cpus=8)
    assert str(s) == "626 'fit' in /jobs/fit.1 (cpus=8)"
    bare = tether.Submission(jobid='1', directory='/j/a', name='a')
    assert str(bare) == "1 'a' in /j/a"


def test_a_directory_is_claimed_by_creating_it_not_by_checking_first():
    """`mkdir` either creates or fails, atomically. Asking and then creating
    would leave a window another submitter could step into."""
    got = claim_command('/jobs/fit.1')
    assert 'mkdir "$_tether_dir"' in got
    assert '[ -e' not in got and '[ -d' not in got


def test_the_first_job_of_a_name_gets_an_unsuffixed_directory():
    got = claim_command('/jobs/fit.1')
    assert '_tether_dir="$_tether_base"' in got


def test_later_ones_are_indexed_rather_than_refused():
    got = claim_command('/jobs/fit.1')
    assert '_tether_dir="$_tether_base.$_tether_n"' in got


def test_claiming_gives_up_rather_than_looping_forever():
    """A parent that cannot be written to would otherwise spin."""
    assert '-le 7 ]' in claim_command('/jobs/fit.1', limit=7)


def test_the_claimed_directory_is_the_one_reported():
    """It may carry an index nobody asked for, and everything after -- script,
    --chdir, the Submission -- has to use that one."""
    assert parse_claim('/jobs/fit.1.3\n') == '/jobs/fit.1.3'


@pytest.mark.parametrize('noise', ['', '\n', '   \n'])
def test_claiming_nothing_is_an_error_not_a_silent_reuse(noise):
    with pytest.raises(tether.SlurmError, match='could not create a job directory'):
        parse_claim(noise)


# -- asking Slurm what happened -------------------------------------------
#
# Two sources with different strengths: `scontrol` reads slurmctld and carries
# the pending reason, `sacct` reads the accounting database and is the only
# one that still knows about a job that finished long ago. A `Job` looks the
# same either way, which is what lets a caller not care.

SCONTROL_LINE = (
    'JobId=682 JobName=fit UserId=andrej(1000) GroupId=users Priority=1 '
    'JobState=FAILED Reason=NonZeroExitCode Dependency=(null) ExitCode=7:0 '
    'RunTime=00:01:05 TimeLimit=01:00:00 Partition=intel NumNodes=2 '
    'NumCPUs=8 WorkDir=/home/andrej/jobs/fit.1'
)


def test_a_job_read_from_slurmctld():
    job = parse_scontrol(SCONTROL_LINE)
    assert (job.jobid, job.name, job.state) == ('682', 'fit', 'FAILED')
    assert (job.exit_code, job.signal) == (7, 0)
    assert (job.nodes, job.cpus) == (2, 8)
    assert job.reason == 'NonZeroExitCode'
    assert job.elapsed == timedelta(minutes=1, seconds=5)
    assert job.is_finished and job.is_failed


def test_the_uid_slurm_appends_to_a_username_is_dropped():
    """`UserId=andrej(1000)` -- the number is not part of the name."""
    assert parse_scontrol(SCONTROL_LINE).user == 'andrej'


@pytest.mark.parametrize('answer', [
    '', '\n', 'slurm_load_jobs error: Invalid job id specified\n',
])
def test_a_job_slurmctld_never_heard_of(answer):
    assert parse_scontrol(answer) is None


def test_a_value_with_a_space_does_not_swallow_the_next_field():
    """Slurm does not escape values, so this is read key by key rather than by
    splitting the line. Every key tether reads has a single-token value."""
    line = SCONTROL_LINE + ' Comment=some words here Account=phoebe'
    job = parse_scontrol(line)
    assert job.state == 'FAILED' and job.exit_code == 7


# -- the accounting database ----------------------------------------------


def sacct_row(**over):
    fields = {
        'jobid': '682', 'state': 'COMPLETED', 'partition': 'intel',
        'user': 'andrej', 'nodes': '2', 'cpus': '8', 'elapsed': '00:01:05',
        'timelimit': '01:00:00', 'workdir': '/home/andrej', 'exit': '0:0',
        'name': 'fit',
    }
    fields.update(over)
    return '|'.join(fields[k] for k in SACCT_SPEC) + '\n'


def test_a_job_read_from_accounting():
    job = parse_sacct(sacct_row())[0]
    assert (job.jobid, job.name, job.state) == ('682', 'fit', 'COMPLETED')
    assert (job.exit_code, job.signal) == (0, 0)
    assert job.is_finished and not job.is_failed


def test_accounting_has_no_reason_to_give():
    """sacct offers the field and never fills it, so asking would produce a
    convincing blank. A pending job's reason comes from scontrol."""
    assert parse_sacct(sacct_row())[0].reason == ''


def test_a_cancelled_job_is_finished_despite_how_slurm_spells_it():
    """sacct writes `CANCELLED by 1000` -- the uid of whoever asked. Left whole
    it matches nothing in FINISHED_STATES, so the job would read as neither
    running nor finished."""
    job = parse_sacct(sacct_row(state='CANCELLED by 1000'))[0]
    assert job.state == 'CANCELLED'
    assert job.is_finished and job.is_failed


def test_a_cancelled_job_exits_zero_and_is_still_a_failure():
    """The allocation row reports 0:0 for a killed job -- only the .batch step
    records the signal. `state` is the authority, not `exit_code`."""
    job = parse_sacct(sacct_row(state='CANCELLED by 1000', exit='0:0'))[0]
    assert job.exit_code == 0
    assert job.is_failed


def test_a_pipe_in_the_job_name_is_absorbed():
    """Which is why `name` is requested last."""
    assert parse_sacct(sacct_row(name='a|b'))[0].name == 'a|b'


def test_unparseable_accounting_lines_are_skipped():
    assert parse_sacct('\nnonsense\n682|COMPLETED\n') == []


# -- the pieces both share ------------------------------------------------


@pytest.mark.parametrize(('raw', 'want'), [
    ('CANCELLED by 1000', 'CANCELLED'), ('completed', 'COMPLETED'),
    ('  RUNNING  ', 'RUNNING'), ('', ''),
])
def test_state_normalisation(raw, want):
    assert distill_state(raw) == want


@pytest.mark.parametrize(('raw', 'want'), [
    ('7:0', (7, 0)),        # exited 7
    ('0:9', (0, 9)),        # killed by SIGKILL, and so exited 0
    ('0:0', (0, 0)),
    ('', (None, None)),
    ('N/A', (None, None)),
])
def test_exit_codes_keep_the_signal_separate(raw, want):
    """A signalled job exits 0, so collapsing the two would make `scancel`
    indistinguishable from success."""
    assert parse_exit_code(raw) == want


def test_a_job_from_squeue_has_no_exit_code():
    """squeue does not carry one; None means "not known", not "exited zero"."""
    line = '682|RUNNING|intel|andrej|2|8|00:01:05|01:00:00|None|/home/andrej|fit'
    job = parse_squeue(line)[0]
    assert job.exit_code is None and job.signal is None


def test_the_commands_quote_the_job_id():
    assert shlex.split(scontrol_command('1; rm -rf /'))[-2] == '1; rm -rf /'
    assert shlex.split(sacct_command('1; rm -rf /'))[-1] == '1; rm -rf /'


def test_accounting_is_asked_for_the_job_not_its_steps():
    """Without -X every job comes back as three rows: the allocation, .batch
    and .extern."""
    assert ' -X ' in sacct_command('682')
    assert '--parsable2' in sacct_command('682')
