#!/bin/bash
# Local fake cluster for tests/test_live.py: an sshd on 127.0.0.1:2222 plus
# sinfo/squeue shims that emit realistic Slurm output. Lets layers 1 and 2 be
# tested in CI without touching a real cluster.
#
# Usage: bash tests/rig.sh start | stop
#
# Runs entirely as an unprivileged user and writes nothing outside $DIR --
# in particular it does not touch ~/.ssh. Everything the client needs is
# described by the generated $DIR/ssh_config, which the tests hand to tether
# as `ssh_config=`. sshd needs no root here because the only account it ever
# authenticates is the one running it.
set -euo pipefail

DIR=${TETHER_RIG_DIR:-/tmp/tether-rig}
PORT=${TETHER_RIG_PORT:-2222}
SSHD=${TETHER_RIG_SSHD:-/usr/sbin/sshd}

# Distros disagree about where sftp-server lives, and its absence would only
# show up much later as an opaque SFTP channel failure.
for candidate in /usr/lib/openssh/sftp-server /usr/libexec/openssh/sftp-server \
                 /usr/lib/ssh/sftp-server; do
    [ -x "$candidate" ] && { SFTP_SERVER=$candidate; break; }
done

start() {
    [ -x "$SSHD" ] || fail "no sshd at $SSHD (install openssh-server, or set TETHER_RIG_SSHD)"
    [ -n "${SFTP_SERVER:-}" ] || fail "cannot find sftp-server; SFTP tests would fail"
    listening && fail "something is already listening on 127.0.0.1:$PORT"

    rm -rf "$DIR"
    mkdir -p "$DIR/bin"

    # Two throwaway keypairs: one identifies the fake host, one the client.
    ssh-keygen -t ed25519 -N "" -f "$DIR/hostkey" -q
    ssh-keygen -t ed25519 -N "" -f "$DIR/id_ed25519" -q
    cp "$DIR/id_ed25519.pub" "$DIR/authorized_keys"
    chmod 600 "$DIR/authorized_keys" "$DIR/hostkey"

    # Derived from the key we just generated rather than scraped back with
    # ssh-keyscan: no round trip, no race against sshd coming up.
    printf '[127.0.0.1]:%s %s\n' "$PORT" "$(cut -d' ' -f1,2 "$DIR/hostkey.pub")" \
        > "$DIR/known_hosts"

    # StrictModes is off only because $DIR sits under a world-writable /tmp.
    # Host key checking on the client side stays fully enabled.
    cat > "$DIR/sshd_config" <<EOF
Port $PORT
ListenAddress 127.0.0.1
HostKey $DIR/hostkey
PubkeyAuthentication yes
PasswordAuthentication no
KbdInteractiveAuthentication no
AuthorizedKeysFile $DIR/authorized_keys
StrictModes no
UsePAM no
PidFile $DIR/sshd.pid
SetEnv PATH=$DIR/bin:/usr/local/bin:/usr/bin:/bin
Subsystem sftp $SFTP_SERVER
EOF

    # The tests connect to the alias `tether-rig`, so every credential the
    # client needs is named here instead of in the user's ~/.ssh.
    cat > "$DIR/ssh_config" <<EOF
Host tether-rig
    HostName 127.0.0.1
    Port $PORT
    User $(id -un)
    IdentityFile $DIR/id_ed25519
    IdentitiesOnly yes
    UserKnownHostsFile $DIR/known_hosts
EOF

    install_shims
    "$SSHD" -f "$DIR/sshd_config" -E "$DIR/sshd.log"

    for _ in $(seq 50); do listening && break; sleep 0.1; done
    listening || fail "sshd did not come up; see $DIR/sshd.log"
    echo "rig up on 127.0.0.1:$PORT (dir: $DIR)"
}

install_shims() {
    cat > "$DIR/bin/sinfo" <<'EOF'
#!/bin/bash
[ "$1" = "--version" ] && { echo "slurm 23.02.7"; exit 0; }
cat <<'ROWS'
main*|up|7-00:00:00|48|12/4/0/16|node[01-16]
gpu|up|1-00:00:00|64|1/2/0/3|gpu[01-03]
debug|down|30:00|16|0/0/2/2|dbg[01-02]
ROWS
EOF

    cat > "$DIR/bin/squeue" <<'EOF'
#!/bin/bash
USER_FILTER=""; JOB_FILTER=""; NEXT=""
for a in "$@"; do
  case "$a" in
    -u)       NEXT=user ;;
    --job=*)  JOB_FILTER="${a#--job=}" ;;
    *)        [ "$NEXT" = "user" ] && { USER_FILTER="$a"; NEXT=""; } ;;
  esac
done
ALL=$(cat <<'ROWS'
12345|RUNNING|main|andrej|2|48|1-02:03:04|3-00:00:00|node[01-02]|/home/andrej/run|phoebe-fit
12346|PENDING|main|kelly|1|24|0:00|1:00:00|Resources|/home/kelly/x|glaze-sim
12347|RUNNING|gpu|andrej|1|8|10:00|UNLIMITED|gpu01|/tmp|ndpolator
ROWS
)
if [ -n "$JOB_FILTER" ]; then
  [ "$JOB_FILTER" = "999999" ] && { echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1; }
  echo "$ALL" | awk -F'|' -v j="$JOB_FILTER" '$1==j'
elif [ -n "$USER_FILTER" ]; then
  echo "$ALL" | awk -F'|' -v u="$USER_FILTER" '$4==u'
else
  echo "$ALL"
fi
EOF
    chmod +x "$DIR/bin/sinfo" "$DIR/bin/squeue"
}

stop() {
    [ -f "$DIR/sshd.pid" ] && kill "$(cat "$DIR/sshd.pid")" 2>/dev/null || true
    rm -rf "$DIR"
    echo "rig down"
}

listening() {
    (exec 3<>/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null
}

fail() {
    echo "rig: $1" >&2
    exit 1
}

case "${1:-start}" in
    start) start ;;
    stop)  stop ;;
    *)     echo "usage: $0 start|stop" >&2; exit 2 ;;
esac
