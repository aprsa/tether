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

from dataclasses import dataclass
from datetime import timedelta

SEP = "|"

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
_NON_DURATIONS = frozenset({"UNLIMITED", "INVALID", "NOT_SET", "N/A", "", "-"})

PENDING_STATES = frozenset({
    "PENDING", "SUSPENDED", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD",
    "RESV_DEL_HOLD", "STOPPED", "SPECIAL_EXIT",
})
RUNNING_STATES = frozenset({
    "RUNNING", "COMPLETING", "CONFIGURING", "RESIZING", "SIGNALING",
    "STAGE_OUT",
})
FINISHED_STATES = frozenset({
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED",
    "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "REVOKED",
})
FAILED_STATES = FINISHED_STATES - {"COMPLETED"}


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
    if "-" in text:
        head, _, text = text.partition("-")
        try:
            days = int(head)
        except ValueError:
            return None

    try:
        parts = [int(p) for p in text.split(":")]
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
    text = raw.strip().rstrip("+")
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
        return f"{self.jobid} {self.state} {self.name!r} ({self.user})"


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
        return self.available.lower() == "up"

    @property
    def load(self) -> float | None:
        """Fraction of nodes allocated, or None if Slurm did not say."""
        if not self.nodes_total or self.nodes_allocated is None:
            return None
        return self.nodes_allocated / self.nodes_total

    def __str__(self) -> str:
        return f"{self.name} {self.available} {self.nodes_idle}/{self.nodes_total} idle"


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
                jobid=f["jobid"],
                name=f["name"],
                state=f["state"].upper(),
                partition=f["partition"],
                user=f["user"],
                nodes=_int(f["nodes"]),
                cpus=_int(f["cpus"]),
                elapsed=parse_duration(f["elapsed"]),
                timelimit=parse_duration(f["timelimit"]),
                reason=f["reason"],
                workdir=f["workdir"],
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

        name = f["partition"]
        is_default = name.endswith("*")

        counts = [_int(c) for c in f["node_states"].split("/")]
        counts += [None] * (4 - len(counts))
        allocated, idle, other, total = counts[:4]

        partitions.append(
            Partition(
                name=name.rstrip("*"),
                is_default=is_default,
                available=f["available"],
                timelimit=parse_duration(f["timelimit"]),
                cpus_per_node=_int(f["cpus_per_node"]),
                nodes_allocated=allocated,
                nodes_idle=idle,
                nodes_other=other,
                nodes_total=total,
                nodelist=f["nodelist"],
            )
        )
    return partitions
