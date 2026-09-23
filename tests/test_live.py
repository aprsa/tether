"""End-to-end tests against the containerised cluster in `tests/cluster/`.

Everything here runs against real software: a real slurmctld/slurmd pair, real
conda, real environment modules, over a real sshd. Nothing is shimmed, so job
states, exit codes and activation failures are the ones tether will meet on an
HPC rather than ones a test fixture invented.

`tests/conftest.py` brings the cluster up on demand; see it for how to drive the
container by hand.
"""

import json
import shlex
import time

import pytest
from conftest import (
    ALIAS,
    CONDA_BASE,
    CONDA_ENV,
    HOST,
    MODULE,
    MODULES_INIT,
    VENV,
)

import tether


@pytest.fixture
def srv(rig):
    s = tether.SlurmServer(ALIAS, ssh_config=rig)
    yield s
    s.close()


@pytest.fixture
def plain(rig):
    s = tether.Server(ALIAS, ssh_config=rig)
    yield s
    s.close()


def env_server(rig, tmp_path, **environment):
    """A plain Server whose environment `e` is built from the keywords given."""
    path = tmp_path / 'servers' / f'{ALIAS}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'kind': 'plain', 'environments': {'e': environment}}))
    return tether.Server(
        ALIAS, ssh_config=rig, config_dir=str(tmp_path), environment='e'
    )


# -- helpers for driving real jobs ---------------------------------------


def submit(srv, script, name='probe', sbatch_args=''):
    srv.run('mkdir -p ~/jobs', check=True)
    srv.run(
        f'printf %s {shlex.quote(script)} > ~/jobs/{name}.sh', check=True
    )
    return srv.run(
        f'cd ~/jobs && sbatch --parsable {sbatch_args} {name}.sh', check=True
    ).stdout.strip()


def wait_for(srv, jobid, timeout=90):
    """Block until the job reports a finished state.

    Not "until `job()` returns None": since `job()` consults `scontrol` and
    then `sacct`, a finished job keeps answering -- with its state and exit
    code -- rather than vanishing. Waiting for silence would wait forever.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = srv.job(jobid)
        if found is None or found.is_finished:
            return found
        time.sleep(0.3)
    raise AssertionError(f'job {jobid} still running after {timeout}s')


def outcome(srv, jobid):
    """(JobState, ExitCode) straight from scontrol."""
    out = srv.run(f'scontrol show job {jobid}', check=True).stdout
    return (
        out.split('JobState=')[1].split()[0],
        out.split('ExitCode=')[1].split()[0],
    )


# -- link layer ----------------------------------------------------------


def test_lazy_then_connect(srv):
    assert not srv.connected
    srv.connect()
    assert srv.connected
    assert srv.slurm_version.startswith('slurm')


def test_info_and_ping(srv):
    host = srv.info()
    assert host.hostname and host.kernel
    assert host.cpu_count and host.cpu_count > 0
    assert 0 < srv.ping() < 10


def test_run_and_check(srv):
    assert srv.run('echo hello').stdout.strip() == 'hello'

    bad = srv.run('exit 3')
    assert bad.returncode == 3 and not bad.ok        # no exception by default

    with pytest.raises(tether.RemoteCommandError) as exc:
        srv.run('echo boom >&2; exit 3', check=True)
    assert exc.value.result.returncode == 3
    assert 'boom' in exc.value.result.stderr


def test_channel_reuse_is_cheap(srv):
    """50 commands must cost one authentication, not fifty."""
    srv.connect()
    conn_id = id(srv.link._conn)
    start = time.perf_counter()
    for i in range(50):
        assert srv.run(f'echo {i}').stdout.strip() == str(i)
    elapsed = time.perf_counter() - start
    assert id(srv.link._conn) == conn_id        # same connection throughout
    assert elapsed < 15, f'50 commands took {elapsed:.1f}s'


def test_timeout_raises_and_kills_remote_process(srv):
    with pytest.raises(tether.LinkError, match='timed out'):
        srv.run('sleep 30', timeout=2)

    # The connection must survive the timeout and stay usable.
    assert srv.run('echo alive').stdout.strip() == 'alive'


def test_reconnect_is_invisible(srv):
    srv.connect()
    first = id(srv.link._conn)

    srv.link._conn.abort()          # simulate a link drop
    time.sleep(0.5)

    assert srv.run('echo back').stdout.strip() == 'back'
    assert id(srv.link._conn) != first


def test_transfer_roundtrip(srv, tmp_path):
    payload = 'phoebe passband table\n' * 100
    local = tmp_path / 'up.txt'
    local.write_text(payload)

    srv.put(str(local), 'tether-test.txt')
    assert srv.run('wc -c < tether-test.txt', check=True).stdout.strip() == str(
        len(payload)
    )

    back = tmp_path / 'down.txt'
    srv.get('tether-test.txt', str(back))
    assert back.read_text() == payload
    srv.run('rm -f tether-test.txt')


def test_context_manager_closes(rig):
    with tether.SlurmServer(ALIAS, ssh_config=rig) as s:
        assert s.connected
    assert not s.connected


def test_reuse_after_close(srv):
    srv.connect()
    srv.close()
    assert not srv.connected
    assert srv.run('echo again').stdout.strip() == 'again'   # loop recreated


def test_plain_server_has_no_slurm_methods(plain):
    assert isinstance(plain, tether.Server)
    assert not isinstance(plain, tether.SlurmServer)
    assert not hasattr(plain, 'queue')


def test_slurm_absence_is_a_hard_error(rig):
    s = tether.SlurmServer(ALIAS, ssh_config=rig)
    original = s.run

    def sabotaged(cmd, **kw):
        if cmd.startswith('sinfo --version'):
            return original('PATH=/nonexistent sinfo --version', **kw)
        return original(cmd, **kw)

    s.run = sabotaged
    with pytest.raises(tether.SlurmError, match='no usable Slurm'):
        s.connect()
    s.close()


def test_bad_host_raises_link_error(rig):
    s = tether.Server(host=HOST, port=1, ssh_config=rig)
    with pytest.raises(tether.LinkError, match='cannot connect'):
        s.connect()


def test_usable_from_inside_a_running_event_loop(rig):
    """The Jupyter case: a kernel cell already runs inside a loop, and
    `run_until_complete` cannot nest in one. Before the loop moved onto its own
    thread, every tether call raised here."""
    import asyncio

    async def cell():
        s = tether.SlurmServer(ALIAS, ssh_config=rig)
        try:
            s.connect()
            return s.run('echo from-a-cell', check=True).stdout.strip()
        finally:
            s.close()

    assert asyncio.run(cell()) == 'from-a-cell'


def test_an_abandoned_server_is_reclaimed(rig):
    """Rebinding a variable is the most ordinary thing in a notebook. Without
    a finalizer each abandoned Server keeps a thread and a live SSH session
    until the kernel dies."""
    import gc
    import threading

    before = threading.active_count()
    srv = tether.SlurmServer(ALIAS, ssh_config=rig)
    srv.connect()
    assert threading.active_count() > before

    srv = None            # what re-running a cell does
    gc.collect()
    assert threading.active_count() == before


def test_close_is_idempotent(rig):
    s = tether.SlurmServer(ALIAS, ssh_config=rig)
    s.connect()
    s.close()
    s.close()             # must not raise, nor wait on a thread already joined
    assert not s.connected


def test_the_loop_thread_is_cleaned_up_on_close(rig):
    import threading

    before = threading.active_count()
    s = tether.SlurmServer(ALIAS, ssh_config=rig)
    s.connect()
    assert threading.active_count() > before
    s.close()
    assert threading.active_count() == before


# -- phase 0: paths and transfers ----------------------------------------


def test_put_creates_every_missing_parent(srv, tmp_path):
    """`mkdir -p`, not one level.

    SFTP will not create a parent at all, and reports a missing one as a bare
    "No such file" -- which reads like the *source* is absent. The workdir
    itself is removed first, so this pins that the whole chain is created and
    cannot pass by accident because an earlier test made it.
    """
    srv.run(f'rm -rf {srv.path()}', check=True)
    local = tmp_path / 'payload.txt'
    local.write_text('staged\n')
    remote = srv.path('a/b/c/d/e/payload.txt')

    srv.put(str(local), remote)
    try:
        assert srv.run(f'cat {remote}', check=True).stdout == 'staged\n'
        made = srv.run(f'find {srv.path()} -type d', check=True).stdout.split()
        assert len(made) == 6            # the workdir itself, plus a..e
    finally:
        srv.run(f'rm -rf {srv.path()}')


def test_get_creates_every_missing_local_parent(srv, tmp_path):
    srv.run('printf fetched > ~/fetch-probe.txt', check=True)
    local = tmp_path / 'no' / 'such' / 'dir' / 'at' / 'all' / 'out.txt'
    try:
        srv.get('fetch-probe.txt', str(local))
        assert local.read_text() == 'fetched'
    finally:
        srv.run('rm -f ~/fetch-probe.txt')


def test_local_paths_accept_pathlib_objects(srv, tmp_path):
    """Local ends take a Path; the remote end stays str, since a local path
    object carries local separators and cannot describe the far side."""
    local = tmp_path / 'obj.txt'
    local.write_text('via Path\n')

    srv.put(local, srv.path('objs/obj.txt'))          # note: no str()
    back = tmp_path / 'fetched' / 'obj.txt'
    try:
        srv.get(srv.path('objs/obj.txt'), back)       # ...nor here
        assert back.read_text() == 'via Path\n'
    finally:
        srv.run(f'rm -rf {srv.path("objs")}')


def test_path_resolves_the_tilde_in_workdir(srv):
    """`workdir` defaults to `~/.tether`, and SFTP never expands `~`."""
    assert srv.workdir == '~/.tether'                  # as configured
    assert srv.path() == f'{srv.home}/.tether'         # as used
    assert srv.path('jobs', '42') == f'{srv.home}/.tether/jobs/42'
    assert srv.home.startswith('/')


def test_workdir_is_usable_end_to_end(srv, tmp_path):
    """The default workdir must survive a real transfer -- it did not before,
    because `put('~/.tether/x')` fails outright."""
    local = tmp_path / 'x.txt'
    local.write_text('ok\n')
    try:
        srv.put(str(local), srv.path('probe', 'x.txt'))
        assert srv.run(f'cat {srv.path("probe/x.txt")}', check=True).stdout == 'ok\n'
    finally:
        srv.run(f'rm -rf {srv.path("probe")}')


def test_absolute_workdir_is_left_alone(rig):
    s = tether.Server(ALIAS, ssh_config=rig, workdir='/tmp/tether-abs')
    try:
        assert s.path('a') == '/tmp/tether-abs/a'
    finally:
        s.close()


def test_server_timeout_is_the_default_for_run(rig):
    s = tether.Server(ALIAS, ssh_config=rig, timeout=2)
    try:
        assert s.timeout == 2
        with pytest.raises(tether.LinkError, match='timed out'):
            s.run('sleep 30')                      # uses the server default
        assert s.run('echo fine', timeout=30).stdout.strip() == 'fine'
    finally:
        s.close()


# -- scheduler -----------------------------------------------------------


def test_whoami_is_cached(srv):
    assert srv.whoami() == 'tether'
    assert srv.whoami() is srv._username


def test_partitions(srv):
    parts = {p.name: p for p in srv.partitions()}
    assert set(parts) == {'main', 'debug'}
    assert parts['main'].is_default and parts['main'].is_up
    assert not parts['debug'].is_default
    assert parts['main'].nodes_total == 1
    assert parts['debug'].timelimit.total_seconds() == 300


def test_queue_sees_a_real_job(srv):
    jobid = submit(srv, '#!/bin/bash\nsleep 3\n', name='queued')
    try:
        jobs = {j.jobid: j for j in srv.queue()}
        assert jobid in jobs
        job = jobs[jobid]
        assert job.user == 'tether'
        assert job.partition == 'main'          # the default partition
        assert job.is_pending or job.is_running
    finally:
        srv.run(f'scancel {jobid}')
        wait_for(srv, jobid)


def test_queue_filters_by_user(srv):
    jobid = submit(srv, '#!/bin/bash\nsleep 3\n', name='mine')
    try:
        assert jobid in [j.jobid for j in srv.queue(user='tether')]
        assert srv.queue(user='nobody') == []
    finally:
        srv.run(f'scancel {jobid}')
        wait_for(srv, jobid)


def test_a_finished_job_still_answers_and_a_fictional_one_does_not(srv):
    """The two used to be indistinguishable -- both came back None. `scontrol`
    still holds a finished job for MinJobAge, and `sacct` after that, so only
    a job Slurm genuinely never saw is empty now."""
    jobid = submit(srv, '#!/bin/bash\nsleep 2\n', name='lookup')
    found = srv.job(jobid)
    assert found is not None and found.jobid == jobid

    done = wait_for(srv, jobid)
    assert done is not None and done.is_finished and not done.is_failed
    assert done.exit_code == 0
    assert srv.job('999999') is None


def test_real_exit_code_survives(srv):
    """The wrapper/sentinel work in MS2 exists because of exactly this."""
    jobid = submit(srv, '#!/bin/bash\necho payload\nexit 5\n', name='rc')
    wait_for(srv, jobid)

    state, exit_code = outcome(srv, jobid)
    assert state == 'FAILED'
    assert exit_code == '5:0'
    assert 'payload' in srv.run(f'cat ~/jobs/slurm-{jobid}.out', check=True).stdout


def test_cancelled_job_leaves_no_exit_code(srv):
    """scancel kills the shell, so nothing the payload writes on exit happens.

    This is the case the exit-code sentinel structurally cannot cover, and why
    completion detection needs a second source.
    """
    jobid = submit(srv, '#!/bin/bash\nsleep 60\n', name='doomed')
    for _ in range(60):
        job = srv.job(jobid)
        if job and job.is_running:
            break
        time.sleep(0.3)

    srv.run(f'scancel {jobid}', check=True)
    wait_for(srv, jobid)

    state, _ = outcome(srv, jobid)
    assert state == 'CANCELLED'


# -- accounting ----------------------------------------------------------
#
# These are the reason slurmdbd is in the image. tether has no `sacct` support
# yet, so they drive it through `run()` -- what they establish is that the rig
# can express the cases MS2's completion detection has to resolve.


def sacct(srv, jobid, fields='JobID,State,ExitCode'):
    """The allocation row for one job, as a list of fields."""
    out = srv.run(
        f'sacct -j {jobid} -X -n --parsable2 --format={fields}', check=True
    ).stdout.strip()
    return out.split('|') if out else []


def test_a_failure_reaches_the_caller_with_its_exit_code(srv):
    """squeue could never have answered this: it drops a job the moment it
    finishes, and carries no exit code even while it has one."""
    jobid = submit(srv, '#!/bin/bash\nexit 5\n', name='acct')
    done = wait_for(srv, jobid)

    assert done.state == 'FAILED' and done.is_failed
    assert (done.exit_code, done.signal) == (5, 0)
    assert sacct(srv, jobid) == [jobid, 'FAILED', '5:0']   # independent check


def test_a_cancelled_job_is_failed_even_though_its_exit_code_is_zero(srv):
    """The trap `-X` leaves behind, pinned so nobody "fixes" it later.

    The allocation row reports `ExitCode 0:0` for a killed job -- only the
    `.batch` step records the signal. A caller reading `exit_code == 0` as
    success would be wrong; `state` and `is_failed` are the authority.
    """
    jobid = submit(srv, '#!/bin/bash\nsleep 60\n', name='killed')
    for _ in range(60):
        job = srv.job(jobid)
        if job and job.is_running:
            break
        time.sleep(0.3)

    srv.run(f'scancel {jobid}', check=True)
    wait_for(srv, jobid)

    state, exit_code = sacct(srv, jobid)[1:3]
    assert state.startswith('CANCELLED')
    assert exit_code == '0:0'          # ...and yet it plainly did not succeed

    steps = srv.run(
        f'sacct -j {jobid} -n --parsable2 --format=JobID,State,ExitCode',
        check=True,
    ).stdout
    batch = next(ln for ln in steps.splitlines() if ln.startswith(f'{jobid}.batch|'))
    state, code = batch.split('|')[1:3]
    # Killed by a signal, so there is no exit status -- which signal depends on
    # whether the job handled SIGTERM before Slurm escalated to SIGKILL, and
    # under load it often does not. Asserting 0:15 specifically made this flaky.
    assert state == 'CANCELLED'
    assert code.startswith('0:') and code != '0:0', f'expected a signal, got {code}'


def test_sacct_is_silent_about_a_job_that_never_existed(srv):
    """The other half of the ambiguity: "finished" and "never existed" have to
    stay distinguishable, and an empty sacct result is what says the latter."""
    assert sacct(srv, '999999') == []


def test_sacct_records_the_submitting_user_and_account(srv):
    jobid = submit(srv, '#!/bin/bash\ntrue\n', name='who')
    wait_for(srv, jobid)
    _, user, account = sacct(srv, jobid, fields='JobID,User,Account')
    assert user == 'tether'
    assert account == 'tether'          # the association sacctmgr created


def test_bad_partition_is_rejected_by_sbatch(srv):
    srv.run('mkdir -p ~/jobs', check=True)
    srv.run("printf '#!/bin/bash\\ntrue\\n' > ~/jobs/bad.sh", check=True)
    result = srv.run('cd ~/jobs && sbatch --parsable -p nosuchpartition bad.sh')

    assert not result.ok
    assert 'partition' in result.stderr.lower()


# -- environments --------------------------------------------------------


def test_venv_activates(rig, tmp_path):
    srv = env_server(rig, tmp_path, kind='venv', path=VENV)
    info = srv.verify_environment()
    srv.close()

    assert info.kind == 'venv'
    assert info.prefix == VENV
    assert info.python.startswith(f'{VENV}/bin/')
    assert info.version.startswith('3.')


def test_conda_activates_via_conda_base(rig, tmp_path):
    srv = env_server(rig, tmp_path, kind='conda', conda_env=CONDA_ENV, conda_base=CONDA_BASE)
    info = srv.verify_environment()
    srv.close()

    assert info.kind == 'conda'
    assert info.prefix == f'{CONDA_BASE}/envs/{CONDA_ENV}'
    assert info.version.startswith('3.12')


def test_conda_without_conda_base_fails_because_conda_is_not_on_path(rig, tmp_path):
    """The image deliberately keeps conda off PATH, as a real cluster does
    before the right module is loaded. Without `conda_base` there is nothing to
    source, and tether must say so rather than run against the wrong python."""
    srv = env_server(rig, tmp_path, kind='conda', conda_env=CONDA_ENV)
    with pytest.raises(tether.EnvActivationError, match='conda is not on PATH'):
        srv.verify_environment()
    srv.close()


def test_conda_found_via_pre_activation(rig, tmp_path):
    """...and sourcing the hook in pre_activation_cmds is the documented cure."""
    srv = env_server(
        rig,
        tmp_path,
        kind='conda',
        conda_env=CONDA_ENV,
        pre_activation_cmds=[f'source {CONDA_BASE}/etc/profile.d/conda.sh'],
    )
    info = srv.verify_environment()
    srv.close()
    assert info.prefix == f'{CONDA_BASE}/envs/{CONDA_ENV}'


def test_bare_metal_reports_system_python(rig, tmp_path):
    srv = env_server(rig, tmp_path, kind='none')
    info = srv.verify_environment()
    srv.close()

    assert info.kind == 'none'
    assert info.version.startswith('3.')
    assert info.prefix == '/usr'          # no environment was entered


def test_no_environment_configured_still_verifies(plain):
    info = plain.verify_environment()
    assert info.kind == 'none' and info.name == ''


def test_missing_venv_is_an_activation_error(rig, tmp_path):
    srv = env_server(rig, tmp_path, kind='venv', path='/nonexistent/venv')
    with pytest.raises(tether.EnvActivationError, match='could not activate venv'):
        srv.verify_environment()
    srv.close()


def test_a_venv_that_activates_but_does_nothing_is_caught(rig, tmp_path, plain):
    """The one failure `run_or_abort` structurally cannot see.

    An `activate` script that exists but is empty -- an interrupted `python -m
    venv`, a stale mount, a half-copied tree -- sources cleanly and returns 0.
    Without the evidence check this reports success and the payload then runs
    against the system interpreter.
    """
    faux = '/tmp/tether-faux-venv'
    plain.run(f'mkdir -p {faux}/bin && : > {faux}/bin/activate', check=True)
    try:
        srv = env_server(rig, tmp_path, kind='venv', path=faux)
        with pytest.raises(tether.EnvActivationError, match=r'\$VIRTUAL_ENV is unset'):
            srv.verify_environment()
        srv.close()
    finally:
        plain.run(f'rm -rf {faux}', check=True)


def test_missing_conda_env_is_an_activation_error(rig, tmp_path):
    srv = env_server(
        rig, tmp_path, kind='conda', conda_env='no-such-env', conda_base=CONDA_BASE
    )
    with pytest.raises(tether.EnvActivationError, match='could not activate conda'):
        srv.verify_environment()
    srv.close()


# -- modules: the reason pre_activation_cmds exists ---------------------------


def test_module_is_absent_from_a_non_interactive_shell(plain):
    """The premise behind `pre_activation_cmds`, asserted rather than assumed.

    `module` is a shell function from /etc/profile.d, and a non-interactive SSH
    command never sources it. This is real cluster behaviour, not a quirk of
    the container.
    """
    assert not plain.run('command -v module').ok


def test_module_load_fails_without_pre_activation(rig, tmp_path):
    srv = env_server(rig, tmp_path, kind='none', modules=[MODULE])
    with pytest.raises(tether.EnvActivationError, match='module load'):
        srv.verify_environment()
    srv.close()


def test_module_load_works_once_pre_activation_sources_it(rig, tmp_path):
    srv = env_server(
        rig,
        tmp_path,
        kind='none',
        modules=[MODULE],
        pre_activation_cmds=[f'source {MODULES_INIT}'],
    )
    srv.verify_environment()          # activation succeeds...
    loaded = srv.run('printf %s "${TETHER_MODULE_LOADED-unset}"', environment=True)
    srv.close()
    assert loaded.stdout == '1'       # ...and the modulefile really took effect


def test_unknown_module_is_an_activation_error(rig, tmp_path):
    srv = env_server(
        rig,
        tmp_path,
        kind='none',
        modules=['no-such-module/9.9'],
        pre_activation_cmds=[f'source {MODULES_INIT}'],
    )
    with pytest.raises(tether.EnvActivationError, match='module load'):
        srv.verify_environment()
    srv.close()


# -- preamble behaviour --------------------------------------------------


def test_env_exports_reach_the_command_only_with_environment_true(rig, tmp_path):
    srv = env_server(rig, tmp_path, kind='none', env={'TETHER_PROBE': '42'})
    with_env = srv.run('printf %s "${TETHER_PROBE-unset}"', environment=True)
    without = srv.run('printf %s "${TETHER_PROBE-unset}"')
    srv.close()

    assert with_env.stdout == '42'
    assert without.stdout == 'unset'       # run() stays raw by default


def test_verbatim_slots_land_on_the_right_side_of_activation(rig, tmp_path):
    srv = env_server(
        rig,
        tmp_path,
        kind='venv',
        path=VENV,
        pre_activation_cmds=['echo "pre:${VIRTUAL_ENV-unset}"'],
        post_activation_cmds=['echo "post:${VIRTUAL_ENV-unset}"'],
    )
    out = srv.run('true', check=True, environment=True).stdout
    srv.close()

    assert 'pre:unset' in out                        # before activation
    assert f'post:{VENV}' in out                     # after activation


def test_activation_failure_stops_before_the_payload(rig, tmp_path):
    """The guard must abort, not merely complain and carry on."""
    srv = env_server(rig, tmp_path, kind='venv', path='/nonexistent/venv')
    result = srv.run('echo PAYLOAD_RAN', environment=True)
    srv.close()

    assert result.returncode == 1
    assert 'PAYLOAD_RAN' not in result.stdout
    assert 'tether:' in result.stderr


def test_tilde_paths_expand_remotely(rig, tmp_path, plain):
    """`~` must survive quoting -- shlex.quote alone would break it."""
    home = plain.run('printf %s "$HOME"', check=True).stdout
    plain.run('python3 -m venv ~/tilde-venv', check=True, timeout=180)

    srv = env_server(rig, tmp_path, kind='venv', path='~/tilde-venv')
    try:
        assert srv.verify_environment().prefix == f'{home}/tilde-venv'
    finally:
        plain.run('rm -rf ~/tilde-venv')
        srv.close()


def test_environment_applies_to_a_real_job(rig, tmp_path, srv):
    """The whole point: a submitted job runs inside the environment."""
    env = env_server(rig, tmp_path, kind='venv', path=VENV)
    script = '#!/bin/bash\n' + env.preamble + '\npython3 -c "import sys; print(sys.prefix)"\n'
    env.close()

    jobid = submit(srv, script, name='inenv')
    wait_for(srv, jobid)

    state, _ = outcome(srv, jobid)
    assert state == 'COMPLETED'
    assert srv.run(f'cat ~/jobs/slurm-{jobid}.out', check=True).stdout.strip() == VENV


# -- conda discovery -----------------------------------------------------


def test_probe_conda_finds_nothing_it_cannot_reach(plain):
    """The rig keeps conda off PATH, and it is not in this user's
    environments.txt, so an honest probe reports nothing.

    A conventional-directory search would "find" /opt/conda here -- and be
    wrong on any cluster that puts conda somewhere else. Not looking is the
    point: an empty result means "nothing reachable", not "nothing exists".
    """
    assert plain.probe_conda() == []


def test_pre_activation_makes_conda_discoverable(rig, tmp_path):
    """...and naming how to reach it is what makes it visible. This is the
    case `pre_activation_cmds` exists for."""
    srv = env_server(rig, tmp_path, kind='none',
                     pre_activation_cmds=[f'source {CONDA_BASE}/etc/profile.d/conda.sh'])
    try:
        installs = {i.base: i for i in srv.probe_conda()}
        assert CONDA_BASE in installs
        found = installs[CONDA_BASE]
        assert found.version and found.version != 'unknown'
        assert CONDA_ENV in found.environments
    finally:
        srv.close()


def test_conda_is_found_by_path_alone(rig, tmp_path):
    """`type -P`, not `command -v`: once conda's hook is sourced, conda is a
    shell *function* and `command -v` answers "conda" rather than a path."""
    srv = env_server(rig, tmp_path, kind='none',
                     pre_activation_cmds=[f'export PATH={CONDA_BASE}/bin:$PATH'])
    try:
        assert CONDA_BASE in {i.base for i in srv.probe_conda()}
    finally:
        srv.close()


def test_probe_conda_sees_the_per_user_fallback(rig, tmp_path):
    """A named environment created against a base the user cannot write to
    lands in ~/.conda/envs, and `conda activate <name>` still finds it -- so
    the probe has to as well, or it would under-report what is usable."""
    srv = env_server(rig, tmp_path, kind='none',
                     pre_activation_cmds=[f'source {CONDA_BASE}/etc/profile.d/conda.sh'])
    try:
        srv.run(
            f'source {CONDA_BASE}/etc/profile.d/conda.sh && '
            f'conda create -y -n fallback-probe --no-default-packages',
            check=True, timeout=300,
        )
        where = srv.run('ls -d ~/.conda/envs/fallback-probe', check=True).stdout
        assert where.strip().endswith('.conda/envs/fallback-probe')   # premise

        found = next(i for i in srv.probe_conda() if i.base == CONDA_BASE)
        assert 'fallback-probe' in found.environments
    finally:
        srv.run('rm -rf ~/.conda/envs/fallback-probe')
        srv.close()


def test_probe_finds_a_conda_at_the_workdir_prefix(plain):
    """Where `install_conda()` puts one. This path was silently broken: the
    candidate was emitted with a literal backslash-n, so the base never
    matched -- and no test noticed, because nothing had ever installed there.
    """
    target = plain.path('conda')
    plain.run(f'mkdir -p {plain.path()} && ln -sfn {CONDA_BASE} {target}', check=True)
    try:
        found = {i.base: i for i in plain.probe_conda()}
        assert target in found, f'workdir conda not discovered; saw {list(found)}'
        assert found[target].version != 'unknown'
    finally:
        plain.run(f'rm -f {target}')


# -- submitting through Server.submit() -----------------------------------


def test_a_submitted_job_runs_and_leaves_its_whole_record_behind(srv):
    """Everything about the job in one directory -- which is what makes it
    findable again in a later session, after Slurm has forgotten it."""
    s = srv.submit('echo "ran on $(hostname)"', name='record', cpus=1, time='00:02:00')
    try:
        wait_for(srv, s.jobid)
        assert set(srv.run(f'ls -1 {s.directory}', check=True).stdout.split()) == {
            'job.sh', 'jobid', 'stdout', 'stderr',
        }
        assert 'ran on' in srv.run(f'cat {s.directory}/stdout', check=True).stdout
        assert srv.run(f'cat {s.directory}/jobid', check=True).stdout.strip() == s.jobid
    finally:
        srv.run(f'rm -rf {s.directory}', check=True)


def test_the_submission_records_what_was_asked_for(srv):
    s = srv.submit('true', name='asked', cpus=2, time='00:03:00', nodes=1)
    try:
        assert (s.name, s.cpus, s.time, s.nodes) == ('asked', 2, '00:03:00', 1)
        assert s.jobid.isdigit()
        assert '/asked.' in s.directory
    finally:
        srv.run(f'scancel {s.jobid} 2>/dev/null; rm -rf {s.directory}', check=False)


def test_two_jobs_of_one_name_in_one_second_get_a_directory_each(srv):
    """`mkdir -p` would have succeeded on the existing directory and the second
    job would have overwritten the first's script, jobid and output -- before
    the first had even started. Refusing would have been tether deciding what
    the caller meant, so the second is indexed instead."""
    first = srv.submit('echo first', name='clash', time='00:02:00')
    second = srv.submit('echo second', name='clash', time='00:02:00')
    try:
        assert first.directory != second.directory
        assert second.directory.startswith(first.directory)

        for sub, expected in ((first, 'echo first'), (second, 'echo second')):
            wait_for(srv, sub.jobid)
            kept = srv.run(f'cat {sub.directory}/job.sh', check=True).stdout
            assert expected in kept
            assert srv.run(f'cat {sub.directory}/jobid', check=True).stdout.strip() == sub.jobid
    finally:
        for sub in (first, second):
            srv.run(f'rm -rf {sub.directory}', check=True)


def test_the_environment_preamble_reaches_the_job(rig, tmp_path):
    """A job gets the environment an interactive `run()` would."""
    path = tmp_path / 'servers' / f'{ALIAS}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {'kind': 'slurm', 'environments': {'e': {'kind': 'venv', 'path': VENV}}}
    ))
    srv = tether.SlurmServer(
        ALIAS, ssh_config=rig, config_dir=str(tmp_path), environment='e'
    )
    s = srv.submit('python -c "import sys; print(sys.prefix)"', name='envjob', time='00:02:00')
    try:
        wait_for(srv, s.jobid)
        assert VENV in srv.run(f'cat {s.directory}/stdout', check=True).stdout
    finally:
        srv.run(f'rm -rf {s.directory}', check=True)
        srv.close()


# -- cancelling ------------------------------------------------------------


def test_a_running_job_can_be_cancelled(srv):
    s = srv.submit('sleep 120', name='cancelme', time='00:05:00')
    try:
        for _ in range(60):
            found = srv.job(s.jobid)
            if found and found.is_running:
                break
            time.sleep(0.3)

        assert srv.cancel(s.jobid) is None
        done = wait_for(srv, s.jobid)
        assert done.state == 'CANCELLED' and done.is_failed
    finally:
        srv.run(f'rm -rf {s.directory}', check=True)


def test_cancelling_a_finished_job_is_an_error(srv):
    """`scancel` would have been silent and exited 0. Without the check the
    caller could not tell "cancelled" from "was already done"."""
    s = srv.submit('true', name='alreadydone', time='00:02:00')
    try:
        wait_for(srv, s.jobid)
        with pytest.raises(tether.SlurmError, match='already finished'):
            srv.cancel(s.jobid)
    finally:
        srv.run(f'rm -rf {s.directory}', check=True)


def test_cancelling_a_job_that_never_existed_is_an_error(srv):
    """The case a typo produces, which `scancel` reports as success."""
    with pytest.raises(tether.SlurmError, match='no record of job'):
        srv.cancel('999999')
