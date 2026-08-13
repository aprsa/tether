#!/bin/bash
# Bring the cluster up in dependency order, then hand PID 1 to sshd so the
# container dies with it:
#
#   munge -> mariadb -> slurmdbd -> slurmctld -> slurmd -> sshd
#
# slurmdbd must be answering before slurmctld starts, and the cluster has to be
# registered in the accounting database before slurmctld will record anything
# against it.
#
# Configuration lives in the image (/etc/slurm/*.conf). What is left here is
# what genuinely cannot be baked: secrets, the database schema, and host keys
# that arrive read-only over the bind mount at /rig.
set -euo pipefail

RIG=${RIG_DIR:-/rig}

# Every Slurm client resolves the local hostname to work out who it is, and a
# compose-assigned hostname is not always in /etc/hosts. Without this, sinfo
# and slurmd block on DNS for ~30s and the healthcheck never passes.
grep -q "[[:space:]]$(hostname)\$" /etc/hosts || echo "127.0.0.1 $(hostname)" >> /etc/hosts

wait_for() {
    local what=$1 attempts=$2; shift 2
    for _ in $(seq "$attempts"); do
        "$@" >/dev/null 2>&1 && return 0
        sleep 1
    done
    echo "timed out waiting for $what" >&2
    return 1
}

# -- munge ------------------------------------------------------------------
# munged refuses to start unless its socket directory is executable by all,
# which rules out the 0700 that looks right everywhere else.
install -d -o munge -g munge -m 0700 /etc/munge /var/lib/munge /var/log/munge
install -d -o munge -g munge -m 0755 /run/munge
if [ ! -s /etc/munge/munge.key ]; then
    dd if=/dev/urandom of=/etc/munge/munge.key bs=1024 count=1 status=none
    chown munge:munge /etc/munge/munge.key
    chmod 400 /etc/munge/munge.key
fi
setpriv --reuid=munge --regid=munge --clear-groups /usr/sbin/munged

install -d -o slurm -g slurm /var/spool/slurmctld /var/log/slurm
install -d /var/spool/slurmd
mkdir -p /sys/fs/cgroup/system.slice   # the directory systemd would have made

# -- mariadb ----------------------------------------------------------------
install -d -o mysql -g mysql /run/mysqld
setpriv --reuid=mysql --regid=mysql --clear-groups \
    /usr/sbin/mariadbd --datadir=/var/lib/mysql &
wait_for 'mariadb' 60 mariadb -e 'SELECT 1'

# Idempotent: the container can be restarted without wiping the database.
mariadb <<'SQL'
CREATE DATABASE IF NOT EXISTS slurm_acct_db;
CREATE USER IF NOT EXISTS 'slurm'@'localhost' IDENTIFIED BY 'tether-rig';
GRANT ALL ON slurm_acct_db.* TO 'slurm'@'localhost';
FLUSH PRIVILEGES;
SQL

# -- slurmdbd ---------------------------------------------------------------
/usr/sbin/slurmdbd
wait_for 'slurmdbd' 60 sacctmgr -n show cluster

# The cluster, account and user have to exist in the database before jobs can
# be recorded against them. All three are idempotent; ignore "already exists".
sacctmgr -i add cluster tether        >/dev/null 2>&1 || true
sacctmgr -i add account tether Description='tether live tests' >/dev/null 2>&1 || true
sacctmgr -i add user tether Account=tether >/dev/null 2>&1 || true

# -- slurmctld / slurmd -----------------------------------------------------
/usr/sbin/slurmctld

# slurmd (25.11+) migrates every pid out of the container's root cgroup before
# it will start, because cgroup v2 forbids processes in a non-leaf node. If one
# of those pids exits mid-enumeration it gives up with ESRCH -- and this script
# has been spawning short-lived children all the way down, so that is exactly
# what happens on a cold start.
#
# Note the check: slurmd daemonises, so the parent exits 0 whether or not the
# daemon survived. Its exit status says nothing; only pgrep does.
for attempt in 1 2 3 4 5; do
    /usr/sbin/slurmd || true
    sleep 1
    pgrep -x slurmd >/dev/null && break
    echo "slurmd did not stay up (attempt $attempt), retrying" >&2
done
pgrep -x slurmd >/dev/null || echo 'slurmd never started; see /var/log/slurm/slurmd.log' >&2

# -- sshd -------------------------------------------------------------------
# The host key arrives read-only over the bind mount; sshd insists on owning it
# and on 0600, so it is copied rather than used in place. The base image ships
# its own host keys -- drop them, or sshd matches our private key against a
# stale .pub and refuses to use it.
rm -f /etc/ssh/ssh_host_*
install -m 0600 -o root -g root "$RIG/hostkey" /etc/ssh/ssh_host_ed25519_key
install -m 0644 -o root -g root "$RIG/hostkey.pub" /etc/ssh/ssh_host_ed25519_key.pub
install -d -m 0700 -o tether -g tether /home/tether/.ssh
install -m 0600 -o tether -g tether "$RIG/authorized_keys" /home/tether/.ssh/authorized_keys

cat > /etc/ssh/sshd_config <<'CONF'
Port 22
HostKey /etc/ssh/ssh_host_ed25519_key
PubkeyAuthentication yes
PasswordAuthentication no
KbdInteractiveAuthentication no
UsePAM no
AuthorizedKeysFile .ssh/authorized_keys
PrintMotd no
AcceptEnv LANG LC_*
Subsystem sftp /usr/lib/openssh/sftp-server
CONF
mkdir -p /run/sshd

# The node registers a moment after slurmd starts; the compose healthcheck is
# what actually gates readiness, so just get out of its way.
exec /usr/sbin/sshd -D -e
