#!/bin/bash
# Local fake cluster for tests/test_live.py: an sshd on 127.0.0.1:2222 plus
# sinfo/squeue shims that emit realistic Slurm output. Lets layers 1 and 2 be
# tested in CI without touching a real cluster.
#
# Usage: sudo bash tests/rig.sh start | stop
set -euo pipefail

DIR=${TETHER_RIG_DIR:-/tmp/tether-rig}
PORT=${TETHER_RIG_PORT:-2222}
CONF="$DIR/sshd_config"

start() {
    mkdir -p "$DIR" /run/sshd "$HOME/.ssh"
    chmod 700 "$HOME/.ssh"

    [ -f "$HOME/.ssh/id_ed25519" ] || ssh-keygen -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519" -q
    [ -f "$DIR/hostkey" ] || ssh-keygen -t ed25519 -N "" -f "$DIR/hostkey" -q
    cat "$HOME/.ssh/id_ed25519.pub" >> "$HOME/.ssh/authorized_keys"
    chmod 600 "$HOME/.ssh/authorized_keys"

    cat > "$CONF" <<EOF
Port $PORT
ListenAddress 127.0.0.1
HostKey $DIR/hostkey
PermitRootLogin yes
PubkeyAuthentication yes
PasswordAuthentication no
AuthorizedKeysFile $HOME/.ssh/authorized_keys
UsePAM no
PidFile $DIR/sshd.pid
Subsystem sftp /usr/lib/openssh/sftp-server
EOF

    install_shims
    /usr/sbin/sshd -f "$CONF"
    sleep 1
    # asyncssh validates host keys strictly and offers no opt-out.
    ssh-keyscan -p "$PORT" -t ssh-ed25519 127.0.0.1 2>/dev/null >> "$HOME/.ssh/known_hosts"
    echo "rig up on 127.0.0.1:$PORT"
}

install_shims() {
    cat > /usr/local/bin/sinfo <<'EOF'
#!/bin/bash
[ "$1" = "--version" ] && { echo "slurm 23.02.7"; exit 0; }
cat <<'ROWS'
main*|up|7-00:00:00|48|12/4/0/16|node[01-16]
gpu|up|1-00:00:00|64|1/2/0/3|gpu[01-03]
debug|down|30:00|16|0/0/2/2|dbg[01-02]
ROWS
EOF

    cat > /usr/local/bin/squeue <<'EOF'
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
    chmod +x /usr/local/bin/sinfo /usr/local/bin/squeue
}

stop() {
    [ -f "$DIR/sshd.pid" ] && kill "$(cat "$DIR/sshd.pid")" 2>/dev/null || true
    rm -f /usr/local/bin/sinfo /usr/local/bin/squeue
    echo "rig down"
}

case "${1:-start}" in
    start) start ;;
    stop)  stop ;;
    *)     echo "usage: $0 start|stop" >&2; exit 2 ;;
esac
