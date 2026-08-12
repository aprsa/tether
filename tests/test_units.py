"""Unit tests for the parts of tether that need no cluster."""

from datetime import timedelta

import pytest

import tether
from tether.slurm import parse_duration, parse_sinfo, parse_squeue


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("30", timedelta(seconds=30)),
        ("5:30", timedelta(minutes=5, seconds=30)),
        ("1:05:30", timedelta(hours=1, minutes=5, seconds=30)),
        ("2-03", timedelta(days=2, hours=3)),
        ("2-03:04", timedelta(days=2, hours=3, minutes=4)),
        ("2-03:04:05", timedelta(days=2, hours=3, minutes=4, seconds=5)),
        ("  10:00  ", timedelta(minutes=10)),
        ("UNLIMITED", None),
        ("INVALID", None),
        ("N/A", None),
        ("", None),
        ("nonsense", None),
    ],
)
def test_parse_duration(raw, expected):
    assert parse_duration(raw) == expected


def test_squeue_parses_realistic_output():
    out = (
        "12345|RUNNING|main|andrej|2|48|1-02:03:04|3-00:00:00|node[01-02]|"
        "/home/andrej/run|phoebe-fit\n"
        "12346|PENDING|main|kelly|1|24|0:00|1:00:00|Resources|"
        "/home/kelly/x|glaze-sim\n"
        "12347_3|COMPLETING|gpu|andrej|1|8|10:00|UNLIMITED|node05|"
        "/tmp|array task\n"
    )
    jobs = parse_squeue(out)
    assert len(jobs) == 3

    a, b, c = jobs
    assert (a.jobid, a.name, a.user) == ("12345", "phoebe-fit", "andrej")
    assert a.nodes == 2 and a.cpus == 48
    assert a.elapsed == timedelta(days=1, hours=2, minutes=3, seconds=4)
    assert a.is_running and not a.is_pending and not a.is_finished

    assert b.is_pending and b.reason == "Resources"
    assert b.elapsed == timedelta(0)

    assert c.jobid == "12347_3"        # array task id survives
    assert c.timelimit is None          # UNLIMITED is not zero
    assert c.is_running                 # COMPLETING counts as active


def test_squeue_absorbs_pipe_in_job_name():
    """Name is the last field, so a `|` inside it must not shift the others."""
    out = "99|RUNNING|main|andrej|1|4|1:00|2:00|node01|/home/andrej|a|b\n"
    (job,) = parse_squeue(out)
    assert job.jobid == "99"
    assert job.workdir == "/home/andrej"
    assert job.name == "a|b"


def test_squeue_skips_malformed_lines():
    out = "12345|RUNNING|main\n" + "\n" + "1|RUNNING|m|u|1|1|0:01|1:00|n|/w|ok\n"
    (job,) = parse_squeue(out)
    assert job.name == "ok"


def test_sinfo_parses_and_computes_load():
    out = (
        "main*|up|7-00:00:00|48|12/4/0/16|node[01-16]\n"
        "gpu|down|1-00:00:00|64|0/2/1/3|gpu[01-03]\n"
    )
    main, gpu = parse_sinfo(out)

    assert main.name == "main" and main.is_default and main.is_up
    assert (main.nodes_allocated, main.nodes_idle, main.nodes_total) == (12, 4, 16)
    assert main.load == pytest.approx(0.75)
    assert main.timelimit == timedelta(days=7)
    assert main.cpus_per_node == 48

    assert not gpu.is_default and not gpu.is_up
    assert gpu.nodes_other == 1


def test_config_missing_file_is_not_an_error(tmp_path):
    cfg = tether.load_config(tmp_path)
    assert cfg.servers == {} and cfg.environments == {}


def test_config_roundtrip(tmp_path):
    (tmp_path / "servers.toml").write_text(
        """
        [server.terra]
        host = "terra.villanova.edu"
        user = "andrej"
        workdir = "~/.tether"
        default_environment = "phoebe"

        [server.laptop]
        kind = "plain"

        [environment.phoebe]
        kind = "conda"
        name = "phoebe-dev"
        modules = ["openmpi/4.1.5"]
        env = { OMP_NUM_THREADS = "1" }
        """
    )
    cfg = tether.load_config(tmp_path)

    terra = cfg.server("terra")
    assert terra.host == "terra.villanova.edu"
    assert terra.kind == "slurm"            # default
    assert cfg.server("laptop").host == "laptop"   # defaults to table key
    assert cfg.environments["phoebe"].modules == ("openmpi/4.1.5",)
    assert cfg.environments["phoebe"].env == {"OMP_NUM_THREADS": "1"}


@pytest.mark.parametrize(
    "body,fragment",
    [
        ('[server.a]\nhostt = "x"', "unknown key"),
        ('[server.a]\nkind = "pbs"', "kind must be one of"),
        ('[server.a]\ndefault_environment = "nope"', "unknown environment"),
        ('[environment.e]\nkind = "conda"', "requires 'name'"),
        ("[server.a\n", "servers.toml"),
    ],
)
def test_config_errors_are_loud(tmp_path, body, fragment):
    (tmp_path / "servers.toml").write_text(body)
    with pytest.raises(tether.ConfigError) as exc:
        tether.load_config(tmp_path)
    assert fragment in str(exc.value)


def test_server_needs_a_host(tmp_path):
    with pytest.raises(tether.ConfigError):
        tether.Server(config_dir=str(tmp_path))


def test_server_construction_does_no_io(tmp_path):
    s = tether.server("nonexistent.invalid", config_dir=str(tmp_path))
    assert isinstance(s, tether.SlurmServer)   # slurm is the default kind
    assert s.host == "nonexistent.invalid"
    assert not s.connected                     # lazy: nothing opened


def test_server_kind_is_normalized_to_strenum(tmp_path):
    plain = tether.server("localhost", kind="plain", config_dir=str(tmp_path))
    assert isinstance(plain, tether.Server)
    assert plain.host == "localhost"

    slurm = tether.server("localhost", kind=tether.ServerKind.SLURM, config_dir=str(tmp_path))
    assert isinstance(slurm, tether.SlurmServer)


def test_spec_formats_are_stable():
    from tether.slurm import SINFO_SPEC, SQUEUE_SPEC, spec_format
    assert spec_format(SQUEUE_SPEC) == "%i|%T|%P|%u|%D|%C|%M|%l|%R|%Z|%j"
    assert spec_format(SINFO_SPEC) == "%P|%a|%l|%c|%F|%N"
    assert list(SQUEUE_SPEC)[-1] == "name"   # name last: absorbs an embedded |
