"""Slurm command formats and parsers.

Deliberately free of I/O so it can be unit-tested without a cluster: it turns
strings into dataclasses and back.

Every command uses an explicit `-o` format string with `-h` (no header).
Default human-readable Slurm output is never parsed -- it is column-aligned,
locale-influenced, and free to change between versions.

`--json` exists on newer Slurm and would be more robust, but its schema
stability across versions is not something to rely on blindly. `Server.info()`
records the remote Slurm version so that decision can be made later on
evidence.
"""

from __future__ import annotations

import posixpath
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from .errors import SlurmError
from .shell import remote_path

SEP = '|'

SQUEUE_SPEC = {
    'jobid':         '%i',
    'state':         '%T',
    'partition':     '%P',
    'user':          '%u',
    'nodes':         '%D',
    'cpus':          '%C',
    'elapsed':       '%M',
    'timelimit':     '%l',
    'reason':        '%R',
    'workdir':       '%Z',
    'name':          '%j',  # last so that a `|` inside it is absorbed
}

SINFO_SPEC = {
    'partition':     '%P',
    'available':     '%a',
    'timelimit':     '%l',
    'cpus_per_node': '%c',
    'node_states':   '%F',  # "allocated/idle/other/total"
    'nodelist':      '%N',
}

#: Values Slurm uses for "no meaningful duration here".
_NON_DURATIONS = frozenset({'UNLIMITED', 'INVALID', 'NOT_SET', 'N/A', '', '-'})

PENDING_STATES = frozenset({
    'PENDING', 'SUSPENDED', 'REQUEUED', 'REQUEUE_FED', 'REQUEUE_HOLD',
    'RESV_DEL_HOLD', 'STOPPED', 'SPECIAL_EXIT',
})
RUNNING_STATES = frozenset({
    'RUNNING', 'COMPLETING', 'CONFIGURING', 'RESIZING', 'SIGNALING',
    'STAGE_OUT',
})
FINISHED_STATES = frozenset({
    'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'PREEMPTED',
    'BOOT_FAIL', 'DEADLINE', 'OUT_OF_MEMORY', 'REVOKED',
})
FAILED_STATES = FINISHED_STATES - {'COMPLETED'}


def spec_format(spec: dict[str, str]) -> str:
    """Turn a Slurm spec dict into a single `-o` format string."""
    return SEP.join(spec.values())


def parse_duration(raw: str) -> timedelta | None:
    """Parse a Slurm duration.

    Handles `SS`, `MM:SS`, `HH:MM:SS`, `D-HH`, `D-HH:MM` and `D-HH:MM:SS`.
    Returns `None` for UNLIMITED, INVALID, N/A and anything unrecognised --
    callers must treat `None` as "no number available", not as zero.
    """
    text = raw.strip()
    if text.upper() in _NON_DURATIONS:
        return None

    days = 0
    if '-' in text:
        head, _, text = text.partition('-')
        try:
            days = int(head)
        except ValueError:
            return None

    try:
        parts = [int(p) for p in text.split(':')]
    except ValueError:
        return None

    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        # Ambiguous without the day marker: "5:30" is MM:SS, "1-5:30" is HH:MM.
        (hours, minutes, seconds) = (
            (parts[0], parts[1], 0) if days else (0, parts[0], parts[1])
        )
    elif len(parts) == 1:
        (hours, minutes, seconds) = (parts[0], 0, 0) if days else (0, 0, parts[0])
    else:
        return None

    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def _int(raw: str) -> int | None:
    """Slurm emits things like '4+' or 'N/A' where an integer is expected."""
    text = raw.strip().rstrip('+')
    try:
        return int(text)
    except ValueError:
        return None


@dataclass(frozen=True)
class Job:
    jobid: str
    name: str
    state: str
    partition: str
    user: str
    nodes: int | None
    cpus: int | None
    elapsed: timedelta | None
    timelimit: timedelta | None
    reason: str
    workdir: str
    exit_code: int | None = None
    signal: int | None = None

    @property
    def is_pending(self) -> bool:
        return self.state in PENDING_STATES

    @property
    def is_running(self) -> bool:
        return self.state in RUNNING_STATES

    @property
    def is_finished(self) -> bool:
        return self.state in FINISHED_STATES

    @property
    def is_failed(self) -> bool:
        return self.state in FAILED_STATES

    def __str__(self) -> str:
        return f'{self.jobid} {self.state} {self.name!r} ({self.user})'


@dataclass(frozen=True)
class Partition:
    name: str
    is_default: bool
    available: str
    timelimit: timedelta | None
    cpus_per_node: int | None
    nodes_allocated: int | None
    nodes_idle: int | None
    nodes_other: int | None
    nodes_total: int | None
    nodelist: str

    @property
    def is_up(self) -> bool:
        return self.available.lower() == 'up'

    @property
    def load(self) -> float | None:
        """Fraction of nodes allocated, or None if Slurm did not say."""
        if not self.nodes_total or self.nodes_allocated is None:
            return None
        return self.nodes_allocated / self.nodes_total

    def __str__(self) -> str:
        return f'{self.name} {self.available} {self.nodes_idle}/{self.nodes_total} idle'


def _fields(line: str, count: int) -> list[str] | None:
    parts = line.split(SEP, count - 1)
    return parts if len(parts) == count else None


def parse_squeue(stdout: str) -> list[Job]:
    """Parse `squeue -h -o SQUEUE_FORMAT`. Unparseable lines are skipped."""
    jobs = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = _fields(line, len(SQUEUE_SPEC))
        if parts is None:
            continue
        f = dict(zip(SQUEUE_SPEC.keys(), (p.strip() for p in parts)))
        jobs.append(
            Job(
                jobid=f['jobid'],
                name=f['name'],
                state=f['state'].upper(),
                partition=f['partition'],
                user=f['user'],
                nodes=_int(f['nodes']),
                cpus=_int(f['cpus']),
                elapsed=parse_duration(f['elapsed']),
                timelimit=parse_duration(f['timelimit']),
                reason=f['reason'],
                workdir=f['workdir'],
            )
        )
    return jobs


def parse_sinfo(stdout: str) -> list[Partition]:
    """Parse `sinfo -h -s -o SINFO_FORMAT`. Unparseable lines are skipped."""
    partitions = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = _fields(line, len(SINFO_SPEC))
        if parts is None:
            continue
        f = dict(zip(SINFO_SPEC.keys(), (p.strip() for p in parts)))

        name = f['partition']
        is_default = name.endswith('*')

        counts = [_int(c) for c in f['node_states'].split('/')]
        counts += [None] * (4 - len(counts))
        allocated, idle, other, total = counts[:4]

        partitions.append(
            Partition(
                name=name.rstrip('*'),
                is_default=is_default,
                available=f['available'],
                timelimit=parse_duration(f['timelimit']),
                cpus_per_node=_int(f['cpus_per_node']),
                nodes_allocated=allocated,
                nodes_idle=idle,
                nodes_other=other,
                nodes_total=total,
                nodelist=f['nodelist'],
            )
        )
    return partitions


#: Where a job's files live, relative to the directory `--chdir` selects.
STDOUT = 'stdout'
STDERR = 'stderr'
SCRIPT = 'job.sh'

#: Written next to them, so tether knows a job existed even after Slurm has
#: forgotten it. Slurm owns process *state*; the filesystem owns the record
#: that there was a process at all -- which is what keeps "finished long ago"
#: distinguishable from "never submitted" on a cluster without accounting.
JOBID = 'jobid'

#: The common directives, in the order they will be written. Anything absent
#: from a submission is simply not emitted, so Slurm's own defaults apply.
_DIRECTIVES = {
    'name': 'job-name',
    'partition': 'partition',
    'nodes': 'nodes',
    'cpus': 'cpus-per-task',
    'time': 'time',
    'memory': 'mem',
}


@dataclass(frozen=True)
class Submission:
    """A complete account of what was sent to the cluster.

    Deliberately a record of the *request*, not of the job's state: it does
    not go stale, and it is what remains meaningful after the job has finished
    and Slurm has forgotten. Ask `Server.job()` for what the job is doing now.

    `directory` is where everything about this job lives -- the script as
    submitted, `stdout`, `stderr`, and whatever the payload wrote. It is also
    the handle for picking the job up again in a later session, which is the
    point of naming it something reconstructible rather than after the jobid.
    """

    jobid: str
    directory: str
    name: str
    partition: str | None = None
    nodes: int | None = None
    cpus: int | None = None
    time: str | None = None
    memory: str | None = None
    directives: tuple[tuple[str, str], ...] = ()
    inputs: tuple[str, ...] = ()

    def __str__(self) -> str:
        asked = ', '.join(
            f'{key}={value}' for key, value in (
                ('partition', self.partition), ('nodes', self.nodes),
                ('cpus', self.cpus), ('time', self.time), ('memory', self.memory),
            ) if value is not None
        )
        return f'{self.jobid} {self.name!r} in {self.directory}' + (f' ({asked})' if asked else '')


def job_directory(base: str, name: str, when: datetime) -> str:
    """`<base>/<name>.<timestamp>`, the directory one job owns.

    Named before submission because it has to be: `--chdir` needs a directory
    and the jobid does not exist until `sbatch` has already been told where to
    run. A timestamp rather than the jobid also means the name is meaningful
    to a human reading `ls`, and reconstructible without asking Slurm.
    """
    check_name(name)
    return f'{base.rstrip("/")}/{name}.{when:%Y%m%d-%H%M%S}'


def check_name(name: str) -> None:
    """A job name is also a directory component, so it is narrower than Slurm
    would allow. Raises `SlurmError` rather than producing a path with a `/`
    in the middle of it."""
    # Slurm accepts more than this, but a name becomes a directory here.
    if not re.fullmatch(r'[A-Za-z0-9._-]+', name or ''):
        raise SlurmError(
            f'{name!r} is not a usable job name: it becomes a directory, so '
            f'letters, digits, dot, dash and underscore only'
        )


def batch_script(
    payload: str,
    *,
    name: str,
    preamble: str = '',
    partition: str | None = None,
    nodes: int | None = None,
    cpus: int | None = None,
    time: str | None = None,
    memory: str | None = None,
    directives: Mapping[str, str] | None = None,
) -> str:
    """The batch script, exactly as it will be written to disk.

    Three parts in order: the `#SBATCH` block, the environment preamble, and
    the payload. The preamble is whatever `Environment.preamble()` produced,
    so a job gets the same environment an interactive `run()` would -- and the
    guards inside it abort the script rather than running the payload against
    the wrong interpreter.

    `directives` carries anything tether has not anticipated -- `gres`,
    `account`, `exclusive` -- as `{option: value}` rendered `--option=value`.
    A value of `''` emits a bare flag. Nothing here is quoted: `#SBATCH` lines
    are read by Slurm, not by a shell, and quoting them would make the quotes
    part of the value.

    Deliberately no exit-code trap. `sacct` records state and exit code
    durably where accounting exists, and where it does not, the job directory
    is what proves the job ran. A second record would be a second truth.
    """
    check_name(name)

    # check for protected directives:
    asked_for = directives or {}
    reserved = [name for name in ('output', 'error') if name in asked_for]
    if reserved:
        raise SlurmError(
            f'{", ".join(sorted(reserved))} cannot be set through directives: '
            f"tether writes the job's output into its own directory, and "
            f'`stdout()` reads it from there. Redirect inside the payload if '
            f'you need it elsewhere'
        )

    lines = ['#!/bin/bash']
    asked = {'name': name, 'partition': partition, 'nodes': nodes,
             'cpus': cpus, 'time': time, 'memory': memory}
    for key, option in _DIRECTIVES.items():
        if asked[key] is not None:
            lines.append(f'#SBATCH --{option}={asked[key]}')

    lines.append(f'#SBATCH --output={STDOUT}')
    lines.append(f'#SBATCH --error={STDERR}')

    for option, value in (directives or {}).items():
        lines.append(f'#SBATCH --{option}={value}' if value != '' else f'#SBATCH --{option}')

    if preamble:
        lines += ['', preamble]
    lines += ['', payload, '']

    return '\n'.join(lines)


def submit_command(directory: str) -> str:
    """Command submitting the script already written into `directory`.

    `--parsable` so the answer is a jobid rather than a sentence, and
    `--chdir` so the job runs where its files are: `stdout` and `stderr` are
    relative names, and Slurm resolves them against the working directory.
    """
    at = remote_path(directory)
    return f'sbatch --parsable --chdir={at} {at}/{SCRIPT}'


def parse_submit(stdout: str) -> str:
    """The jobid out of `sbatch --parsable`.

    Which prints `jobid` alone, or `jobid;cluster` on a federation -- the
    cluster name is not part of the id and would poison every later lookup.
    """
    first = stdout.strip().splitlines()[0] if stdout.strip() else ''
    jobid = first.split(';')[0].strip()
    if not jobid.isdigit():
        raise SlurmError(f'sbatch did not return a job id: {stdout.strip()!r}')
    return jobid


def claim_command(directory: str, *, limit: int = 999) -> str:
    """Shell that creates `directory`, or the first free `<directory>.N`.

    Two jobs of one name submitted inside one second want the same directory.
    Letting the second take it over loses the first's script, jobid and output
    -- possibly before it has even started -- and refusing the second would be
    tether deciding what the caller meant. An index does neither.

    The loop is in the shell deliberately. `mkdir` either creates a directory
    or fails, atomically, so looping there is safe even against another tether
    in another process; asking from Python and then creating would leave a
    window between the two. Variables are prefixed so the surrounding
    environment is not clobbered.

    Prints the directory it made, and nothing if it exhausted `limit` --
    which `parse_claim()` turns into an error rather than a silent reuse.
    """
    at = remote_path(directory)
    parent = remote_path(posixpath.dirname(directory.rstrip('/')) or '.')

    return '\n'.join((
        f'mkdir -p {parent} || exit 1',
        f'_tether_base={at}',
        '_tether_n=1',
        f'while [ "$_tether_n" -le {limit} ]; do',
        '    if [ "$_tether_n" -eq 1 ]; then _tether_dir="$_tether_base";',
        '    else _tether_dir="$_tether_base.$_tether_n"; fi',
        '    if mkdir "$_tether_dir" 2>/dev/null; then',
        "        printf '%s\\n' \"$_tether_dir\"; break",
        '    fi',
        '    _tether_n=$((_tether_n + 1))',
        'done',
    ))


def parse_claim(stdout: str) -> str:
    """The directory that was actually created.

    Which may carry an index the caller did not ask for, and is therefore the
    one everything afterwards must use -- the script, `--chdir`, and the
    `Submission` handed back.
    """
    made = stdout.strip().splitlines()
    if not made or not made[0].strip():
        raise SlurmError(
            'could not create a job directory: every candidate name was taken, '
            'or the parent is not writable'
        )
    return made[0].strip()


#: `scontrol show job` keys, mapped onto `Job` fields. It reads slurmctld's
#: memory rather than the accounting database, so it answers for any pending
#: or running job regardless of age, and for a finished one until `MinJobAge`
#: (300s by default) purges it. Unlike `squeue` it carries the exit code, and
#: unlike `sacct` it needs no accounting service.
SCONTROL_KEYS = {
    'jobid': 'JobId',
    'name': 'JobName',
    'state': 'JobState',
    'partition': 'Partition',
    'user': 'UserId',
    'nodes': 'NumNodes',
    'cpus': 'NumCPUs',
    'elapsed': 'RunTime',
    'timelimit': 'TimeLimit',
    'reason': 'Reason',
    'workdir': 'WorkDir',
    'exit': 'ExitCode',
}

#: `sacct` fields, in the order they are requested. The durable record: the
#: only source that still knows about a job that finished long ago, which is
#: the ordinary case for a detached job someone comes back to.
SACCT_SPEC = {
    'jobid': 'JobID',
    'state': 'State',
    'partition': 'Partition',
    'user': 'User',
    'nodes': 'NNodes',
    'cpus': 'NCPUS',
    'elapsed': 'Elapsed',
    'timelimit': 'Timelimit',
    'workdir': 'WorkDir',
    'exit': 'ExitCode',
    'name': 'JobName',
}


#: One `Key=Value` out of a `--oneliner` dump. Targeted rather than splitting
#: the whole line, because Slurm does not escape values containing spaces: a
#: `WorkDir=/a b` would swallow the next field. Every key read here has a
#: single-token value, and tether validates its own job names.
def _scontrol_value(line: str, key: str) -> str:
    found = re.search(rf'(?:^|\s){re.escape(key)}=(\S*)', line)
    return found.group(1) if found else ''


def scontrol_command(jobid: str | int) -> str:
    """Command asking slurmctld about one job."""
    return f'scontrol show job {shlex.quote(str(jobid))} --oneliner'


def sacct_command(jobid: str | int) -> str:
    """Command asking the accounting database about a job.

    `-X` keeps it to the job rather than also listing every step, and
    `--parsable2` gives `|`-separated fields with no trailing separator.

    The cost of `-X` is worth stating: a cancelled job's allocation row reports
    `ExitCode 0:0`, because only the `.batch` step records the signal that
    killed it. The *state* still says `CANCELLED`, so nothing is lost for a
    caller who reads `is_failed` -- but one who reads `exit_code == 0` as
    success would be wrong.
    """
    fields = ','.join(SACCT_SPEC.values())
    return f'sacct -n -X --parsable2 -o {fields} -j {shlex.quote(str(jobid))}'


def parse_scontrol(stdout: str) -> Job | None:
    """`Job` from `scontrol show job --oneliner`, or `None`.

    `None` covers both "no such job" and an error line, since slurmctld
    answers an unknown id with `slurm_load_jobs error` on stderr and nothing
    usable on stdout.
    """
    line = stdout.strip()
    if not line or not _scontrol_value(line, 'JobId'):
        return None

    f = {name: _scontrol_value(line, key) for name, key in SCONTROL_KEYS.items()}
    exit_code, signal = parse_exit_code(f['exit'])

    return Job(
        jobid=f['jobid'],
        name=f['name'],
        state=distill_state(f['state']),
        partition=f['partition'],
        user=f['user'].partition('(')[0],   # UserId=root(0)
        nodes=_int(f['nodes']),
        cpus=_int(f['cpus']),
        elapsed=parse_duration(f['elapsed']),
        timelimit=parse_duration(f['timelimit']),
        reason=f['reason'],
        workdir=f['workdir'],
        exit_code=exit_code,
        signal=signal,
    )


def parse_sacct(stdout: str) -> list[Job]:
    """Jobs from `sacct --parsable2`. Unparseable lines are skipped.

    `reason` comes back empty: sacct offers the field but never fills it, so
    asking would only produce a convincing blank. A pending job's reason has
    to come from `scontrol`, which still has it at any age.
    """
    jobs = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = _fields(line, len(SACCT_SPEC))
        if parts is None:
            continue
        f = dict(zip(SACCT_SPEC.keys(), (p.strip() for p in parts)))
        exit_code, signal = parse_exit_code(f['exit'])

        jobs.append(
            Job(
                jobid=f['jobid'],
                name=f['name'],
                state=distill_state(f['state']),
                partition=f['partition'],
                user=f['user'],
                nodes=_int(f['nodes']),
                cpus=_int(f['cpus']),
                elapsed=parse_duration(f['elapsed']),
                timelimit=parse_duration(f['timelimit']),
                reason='',
                workdir=f['workdir'],
                exit_code=exit_code,
                signal=signal,
            )
        )
    return jobs


def distill_state(raw: str) -> str:
    """The state as a bare token.

    sacct reports a cancelled job as `CANCELLED by 1000` -- the uid of whoever
    asked. Left whole it matches nothing in `FINISHED_STATES`, so a cancelled
    job would read as neither finished nor running.
    """
    return raw.strip().upper().split(' ', 1)[0]


def parse_exit_code(raw: str) -> tuple[int | None, int | None]:
    """`ExitCode=3:0` as `(exit status, signal)`.

    Both halves are kept: a job killed by a signal exits 0 with a non-zero
    signal, so collapsing them would make it indistinguishable from success.
    `(None, None)` when Slurm said nothing useful.
    """
    status, _, killed = raw.strip().partition(':')
    return _int(status), _int(killed)


def cancel_command(jobid: str | int) -> str:
    """Command asking Slurm to cancel one job.

    No `--signal`: with it, `scancel` *signals* the job and leaves it running,
    which is not what a method called cancel should do. A job that wants a
    chance to clean up asks for it at submission -- `--signal=B:TERM@60` sends
    TERM a minute before the time limit -- which the `directives` passthrough
    already carries.
    """
    return f'scancel {shlex.quote(str(jobid))}'
