#!/bin/bash
# Runs the behave suite in parallel shards, each on its own headless Weston compositor.
# Usage: [SHARDS=n] [CPUS=n] scripts/run_behave.sh [behave args...]

LOGICAL=$(nproc)
PHYSICAL=$(lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l)

# Total number of shards.
# One shard per 2 physical cores, at most 4.
SHARDS=${SHARDS:-$(( PHYSICAL / 2 < 4 ? PHYSICAL / 2 : 4 ))}
# The number of CPUs each AstraStack process is told to use.
# Reserve one logical CPU for each shard's behave/GTK process to accommodate test overhead.
CPUS=${CPUS:-$(( (LOGICAL - SHARDS) / SHARDS ))}

TMP_ROOT=$(mktemp -d -t astrastack-run-XXXXXX)
WESTONS=(); WORKERS=()

cleanup() {
    for pid in "${WORKERS[@]}"; do kill -KILL -- "-$pid" 2>/dev/null; done
    for pid in "${WESTONS[@]}"; do kill "$pid" 2>/dev/null; done
    wait 2>/dev/null
    rm -rf "$TMP_ROOT"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# Start one weston server for each shard.
for n in $(seq 0 $((SHARDS - 1))); do
    mkdir -p "$TMP_ROOT/$n/config" "$TMP_ROOT/$n/tmp"
    weston --backend=headless --socket="astrastack-test-$$-$n" > "$TMP_ROOT/$n/weston.log" 2>&1 &
    WESTONS+=($!)
done

# Wait until all weston servers are up and running.
for n in $(seq 0 $((SHARDS - 1))); do
    for _ in $(seq 100); do [ -S "$XDG_RUNTIME_DIR/astrastack-test-$$-$n" ] && break; sleep 0.1; done
    [ -S "$XDG_RUNTIME_DIR/astrastack-test-$$-$n" ] || { cat "$TMP_ROOT/$n/weston.log"; exit 1; }
done

echo "Running $SHARDS shards with $CPUS CPUs each"
for n in $(seq 0 $((SHARDS - 1))); do
    env -u DISPLAY GDK_BACKEND=wayland WAYLAND_DISPLAY="astrastack-test-$$-$n" XDG_CONFIG_HOME="$TMP_ROOT/$n/config" \
        TMPDIR="$TMP_ROOT/$n/tmp" ASTRASTACK_TEST_CPUS="$CPUS" ASTRASTACK_TEST_SHARD="$n/$SHARDS" \
        setsid python -m behave --no-skipped "$@" > "$TMP_ROOT/$n/behave.log" 2> "$TMP_ROOT/$n/stderr.log" &
    WORKERS+=($!)
done

status=0
for n in $(seq 0 $((SHARDS - 1))); do
    if wait "${WORKERS[$n]}"; then
        echo "== Shard $n passed"
        grep -E "^[0-9]+ (features?|scenarios?|steps?) passed|^Took" "$TMP_ROOT/$n/behave.log"
    else
        status=1
        echo "== Shard $n failed"
        cat "$TMP_ROOT/$n/behave.log"
    fi
done
exit $status
