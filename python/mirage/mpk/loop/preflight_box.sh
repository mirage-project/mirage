#!/bin/bash
# The box checks before a job (docs/megakernel_dev_loop.md section 6, preflight_box; rules A7, H1, H5, H6), run on the box:
#   1. code     md5sum -c of MANIFEST ("md5  path" lines, paths relative to REPO, default ~/mk/mirage_upstream): the box runs
#               the code that was synced
#   2. alone    no keepalive / kda_prod process and no other compute process on the GPUs (a co-resident persistent kernel
#               hangs ours and pollutes every time: memory keepalive-contention-trap); stop the keepalive before this check
#   3. fabric   nvidia-fabricmanager active, and no FABRIC_STATE_OUT_OF_SYNC in dmesg (multicast binds fail after an apt
#               restart of the fabric manager: memory nvls-fabric-broken-by-apt-2026-09-08)
#   4. clocks   per GPU the SM clock and its maximum (printed, not judged)
#   5. disk     at least MIN_FREE_GB (default 20) free in ~; DATA_DIR (if set) present and on a mounted file system
# Prints one line per check, then "PREFLIGHT ok" (exit 0) or "PREFLIGHT FAIL <reasons>" (exit 1).
#   bash preflight_box.sh ~/mk/b.md5
MANIFEST=${1:?usage: preflight_box.sh MANIFEST}
REPO=${REPO:-$HOME/mk/mirage_upstream}
MIN_FREE_GB=${MIN_FREE_GB:-20}
FAIL=()

# 1. code
if [ ! -f "$MANIFEST" ]; then
  echo "code: no manifest $MANIFEST"; FAIL+=(manifest)
else
  bad=$(cd "$REPO" && md5sum -c --quiet "$MANIFEST" 2>&1 | grep -v "^md5sum: WARNING" | head -5)
  if [ -n "$bad" ]; then echo "code: md5 mismatch: $bad"; FAIL+=(md5); else echo "code: $(wc -l < "$MANIFEST") files match $MANIFEST"; fi
fi

# 2. alone (bracket patterns so pgrep does not match this script's own command line)
others=$(pgrep -a -f "[k]eepalive|[k]da_prod" | grep -v -E "^($$|$PPID) " | cut -c1-120)
if [ -n "$others" ]; then echo "alone: keepalive / kda_prod running: $others"; FAIL+=(keepalive); else echo "alone: no keepalive"; fi
if command -v nvidia-smi > /dev/null; then
  apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>&1)
  if [ -n "$apps" ]; then echo "alone: compute processes on the GPUs: $(echo "$apps" | tr '\n' ';')"; FAIL+=(gpu-busy)
  else echo "alone: no compute process on the GPUs"; fi
else
  echo "alone: no nvidia-smi"; FAIL+=(nvidia-smi)
fi

# 3. fabric
fm=$(systemctl is-active nvidia-fabricmanager 2>/dev/null)
if [ "$fm" = active ]; then echo "fabric: nvidia-fabricmanager active"; else echo "fabric: nvidia-fabricmanager $fm"; FAIL+=(fabricmanager); fi
oos=$(sudo -n dmesg > /tmp/preflight_dmesg.$$ 2>/dev/null && grep -c FABRIC_STATE_OUT_OF_SYNC /tmp/preflight_dmesg.$$)
rm -f /tmp/preflight_dmesg.$$
if [ -z "$oos" ]; then echo "fabric: dmesg not readable (sudo -n)"
elif [ "$oos" -gt 0 ]; then echo "fabric: $oos FABRIC_STATE_OUT_OF_SYNC lines in dmesg (reboot to recover)"; FAIL+=(fabric-sync)
else echo "fabric: no FABRIC_STATE_OUT_OF_SYNC"; fi

# 4. clocks
command -v nvidia-smi > /dev/null && echo "clocks (sm MHz, max sm MHz): $(nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm \
  --format=csv,noheader | tr '\n' ';')"

# 5. disk
free=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
if [ "${free:-0}" -lt "$MIN_FREE_GB" ]; then echo "disk: ${free} GB free in ~ (< $MIN_FREE_GB)"; FAIL+=(disk); else echo "disk: ${free} GB free in ~"; fi
if [ -n "$DATA_DIR" ]; then
  if [ -d "$DATA_DIR" ] && findmnt -T "$DATA_DIR" > /dev/null; then echo "data: $DATA_DIR on $(findmnt -n -o TARGET -T "$DATA_DIR")"
  else echo "data: $DATA_DIR missing or not mounted"; FAIL+=(data); fi
fi

if [ ${#FAIL[@]} -eq 0 ]; then echo "PREFLIGHT ok"; exit 0; fi
echo "PREFLIGHT FAIL ${FAIL[*]}"; exit 1
