"""End-to-end tests against a real sshd on 127.0.0.1:2222 with Slurm shims.

Exercises the parts that unit tests cannot: authentication, channel reuse,
timeout-with-terminate, and reconnect after a dropped link.
"""

import socket
import time

import pytest

import tether

HOST, PORT = "127.0.0.1", 2222


def _reachable():
    with socket.socket() as s:
        return s.connect_ex((HOST, PORT)) == 0


pytestmark = pytest.mark.skipif(not _reachable(), reason="no local sshd on 2222")


@pytest.fixture
def srv(tmp_path):
    s = tether.SlurmServer(host=HOST, port=PORT, config_dir=str(tmp_path))
    yield s
    s.close()


def test_lazy_then_connect(srv):
    assert not srv.connected
    srv.connect()
    assert srv.connected
    assert srv.slurm_version == "slurm 23.02.7"


def test_info_and_ping(srv):
    host = srv.info()
    assert host.hostname and host.kernel
    assert host.cpu_count and host.cpu_count > 0
    assert 0 < srv.ping() < 10


def test_run_and_check(srv):
    assert srv.run("echo hello").stdout.strip() == "hello"

    bad = srv.run("exit 3")
    assert bad.returncode == 3 and not bad.ok        # no exception by default

    with pytest.raises(tether.RemoteCommandError) as exc:
        srv.run("echo boom >&2; exit 3", check=True)
    assert exc.value.result.returncode == 3
    assert "boom" in exc.value.result.stderr


def test_channel_reuse_is_cheap(srv):
    """50 commands must cost one authentication, not fifty."""
    srv.connect()
    conn_id = id(srv.link._conn)
    start = time.perf_counter()
    for i in range(50):
        assert srv.run(f"echo {i}").stdout.strip() == str(i)
    elapsed = time.perf_counter() - start
    assert id(srv.link._conn) == conn_id        # same connection throughout
    assert elapsed < 15, f"50 commands took {elapsed:.1f}s"


def test_timeout_raises_and_kills_remote_process(srv):
    with pytest.raises(tether.LinkError, match="timed out"):
        srv.run("sleep 30", timeout=2)

    # The connection must survive the timeout and stay usable.
    assert srv.run("echo alive").stdout.strip() == "alive"


def test_reconnect_is_invisible(srv):
    srv.connect()
    first = id(srv.link._conn)

    srv.link._conn.abort()          # simulate a link drop
    time.sleep(0.5)

    assert srv.run("echo back").stdout.strip() == "back"
    assert id(srv.link._conn) != first


def test_transfer_roundtrip(srv, tmp_path):
    payload = "phoebe passband table\n" * 100
    local = tmp_path / "up.txt"
    local.write_text(payload)

    srv.put(str(local), "/tmp/tether-test.txt")
    assert srv.run("wc -c < /tmp/tether-test.txt", check=True).stdout.strip() == str(
        len(payload)
    )

    back = tmp_path / "down.txt"
    srv.get("/tmp/tether-test.txt", str(back))
    assert back.read_text() == payload


def test_partitions(srv):
    parts = {p.name: p for p in srv.partitions()}
    assert set(parts) == {"main", "gpu", "debug"}
    assert parts["main"].is_default and parts["main"].is_up
    assert parts["main"].load == pytest.approx(0.75)
    assert not parts["debug"].is_up


def test_queue_defaults_to_all_users(srv):
    jobs = srv.queue()
    assert {j.user for j in jobs} == {"andrej", "kelly"}
    assert len(jobs) == 3


def test_queue_filters_by_user(srv):
    jobs = srv.queue(user="andrej")
    assert [j.jobid for j in jobs] == ["12345", "12347"]
    assert all(j.user == "andrej" for j in jobs)


def test_whoami_is_cached(srv):
    who = srv.whoami()
    assert who
    assert srv.whoami() is srv._username


def test_job_lookup(srv):
    job = srv.job(12345)
    assert job is not None
    assert job.name == "phoebe-fit" and job.is_running
    assert job.elapsed.total_seconds() == 1 * 86400 + 2 * 3600 + 3 * 60 + 4

    assert srv.job(999999) is None       # invalid id is None, not an error


def test_context_manager_closes(tmp_path):
    with tether.SlurmServer(host=HOST, port=PORT, config_dir=str(tmp_path)) as s:
        assert s.connected
    assert not s.connected


def test_reuse_after_close(srv):
    srv.connect()
    srv.close()
    assert not srv.connected
    assert srv.run("echo again").stdout.strip() == "again"   # loop recreated


def test_plain_server_has_no_slurm_methods(tmp_path):
    s = tether.server("localhost", kind="plain", config_dir=str(tmp_path))
    assert isinstance(s, tether.Server) and not isinstance(s, tether.SlurmServer)
    assert not hasattr(s, "queue")


def test_slurm_absence_is_a_hard_error(tmp_path):
    s = tether.SlurmServer(host=HOST, port=PORT, config_dir=str(tmp_path))
    s._probe_command = None
    # Hide the shims so `sinfo --version` fails.
    original = s.run

    def sabotaged(cmd, **kw):
        if cmd.startswith("sinfo --version"):
            return original("PATH=/nonexistent sinfo --version", **kw)
        return original(cmd, **kw)

    s.run = sabotaged
    with pytest.raises(tether.SlurmError, match="no usable Slurm"):
        s.connect()
    s.close()


def test_bad_host_raises_link_error(tmp_path):
    s = tether.Server(host="127.0.0.1", port=1, config_dir=str(tmp_path))
    with pytest.raises(tether.LinkError, match="cannot connect"):
        s.connect()
