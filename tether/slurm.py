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

#: A job name that is safe as both a directory component and an `#SBATCH`
#: value. Slurm accepts more than this, but a name becomes a path here.
_NAME = re.compile(r'[A-Za-z0-9._-]+\Z')


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
    if not _NAME.match(name or ''):
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
