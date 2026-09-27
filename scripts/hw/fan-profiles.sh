#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Mjolnir HW setup — Thor fan profiles (host HW provisioning step).
#
# Run this BEFORE any fan-mode selection (or benchmarking). One-time and
# idempotent — every part is gated so re-runs never duplicate anything:
#
#   1. BACKUP (first run only): untouched /etc/nvfancontrol.conf is dumped
#      to /etc/nvfancontrol.conf.bck. The .bck ALWAYS holds the ORIGINAL —
#      it is the clean rollback point and is never overwritten.
#   2. INJECT (per-profile gated): the two performance fan profiles
#      (tuned on the AGX Thor) are inserted before the THERMAL_GROUP
#      section — each ONLY if its block is not already present:
#        FAN_PROFILE recommended  — balanced warmth/acoustics
#        FAN_PROFILE max          — sustained full-load cooling (bench)
#   3. FAN MODE SELECTION: FAN_DEFAULT_PROFILE recommended (not the stock
#      "cool").
#   4. Restart nvfancontrol and print the daemon's active state.
#
# Usage:
#   mjolnir hw setup            # real run (sudo prompts where needed)
#   mjolnir hw setup --dry-run  # transform on a scratch copy + diff; no sudo,
#                               # no writes to /etc
#
# Env: MJOLNIR_NVFANCONF — conf path override (testing).
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

CONF="${MJOLNIR_NVFANCONF:-/etc/nvfancontrol.conf}"
BCK="$CONF.bck"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
[ -r "$CONF" ] || { echo "fan-profiles: cannot read $CONF" >&2; exit 4; }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/profile_recommended.conf" <<'EOF'
	FAN_PROFILE recommended {
		#TEMP	  HYST	PWM	RPM
		0	   0	255	5371
		10	   0	220	4700
		20	   0	180	3900
		30	   0	140	3000
		40	   0	102	2400
		55	   0	90	2100
		70	   0	80	1800
		115	   0	80	1800
	}
EOF

cat > "$TMP/profile_max.conf" <<'EOF'
	FAN_PROFILE max {
		#TEMP	  HYST	PWM	RPM
		0	   0	255	5371
		15	   0	240	5000
		25	   0	220	4700
		35	   0	195	4200
		50	   0	170	3700
		70	   0	120	2700
		115	   0	120	2700
	}
EOF

present() { grep -cE "FAN_PROFILE $1[[:space:]]*\{" "$CONF" || true; }
HAVE_REC="$(present recommended)"
HAVE_MAX="$(present max)"
CUR_DEFAULT="$(awk '/^[[:space:]]*FAN_DEFAULT_PROFILE/ {print $2; exit}' "$CONF")"
CUR_DEFAULT="${CUR_DEFAULT:-<unset>}"

DO_REC=0; [ "$HAVE_REC" -eq 0 ] && DO_REC=1
DO_MAX=0; [ "$HAVE_MAX" -eq 0 ] && DO_MAX=1

echo "fan profiles: $CONF"
if [ -e "$BCK" ]; then
  echo "  backup:      $BCK (exists — original preserved)"
else
  echo "  backup:      $CONF -> $BCK (first run — will be created)"
fi
if [ "$DO_REC" = 1 ]; then
  echo "  inject:      FAN_PROFILE recommended (not present — will be added)"
else
  echo "  inject:      FAN_PROFILE recommended (already present — skipped)"
fi
if [ "$DO_MAX" = 1 ]; then
  echo "  inject:      FAN_PROFILE max (not present — will be added)"
else
  echo "  inject:      FAN_PROFILE max (already present — skipped)"
fi
if [ "$CUR_DEFAULT" != "recommended" ]; then
  echo "  fan mode:    FAN_DEFAULT_PROFILE $CUR_DEFAULT -> recommended"
else
  echo "  fan mode:    FAN_DEFAULT_PROFILE recommended (already selected)"
fi

PENDING=0
[ -e "$BCK" ] || PENDING=1
[ "$DO_REC" = 1 ] && PENDING=1
[ "$DO_MAX" = 1 ] && PENDING=1
[ "$CUR_DEFAULT" != "recommended" ] && PENDING=1

# $1=src $2=dst — inserts the missing profile block(s) before the
# THERMAL_GROUP anchor and selects the fan mode (FAN_DEFAULT_PROFILE).
transform() {
  awk -v do_rec="$DO_REC" -v do_max="$DO_MAX" -v src="$1" \
      -v f_rec="$TMP/profile_recommended.conf" -v f_max="$TMP/profile_max.conf" '
    /^[[:space:]]*FAN_DEFAULT_PROFILE/ { sub(/FAN_DEFAULT_PROFILE.*/, "FAN_DEFAULT_PROFILE recommended") }
    !done && $0 ~ /^[[:space:]]*THERMAL_GROUP[[:space:]]0/ {
      done=1
      if (do_rec) while ((getline l < f_rec) > 0) print l
      if (do_max) while ((getline l < f_max) > 0) print l
    }
    { print }
    END {
      if (!done) {
        print "fan-profiles: no THERMAL_GROUP anchor in " src " — refusing to guess" > "/dev/stderr"
        exit 3
      }
    }
  ' "$1" > "$2"
}

if [ "$DRY_RUN" = 1 ]; then
  if [ "$PENDING" = 0 ]; then
    echo "  already configured — nothing to do"
    exit 0
  fi
  transform "$CONF" "$TMP/new.conf"
  if diff -q "$CONF" "$TMP/new.conf" >/dev/null; then
    echo "  (dry-run) conf already up to date (only the backup is pending)"
  else
    echo "  (dry-run) resulting diff:"
    diff -u "$CONF" "$TMP/new.conf" | sed 's/^/    /'
  fi
  echo "  (dry-run) would then: sudo cp -> $CONF + sudo systemctl restart nvfancontrol"
  exit 0
fi

if [ "$PENDING" = 0 ]; then
  echo "  already configured — nothing to do"
  exit 0
fi

[ -e "$BCK" ] || sudo cp -a -- "$CONF" "$BCK"
if [ "$DO_REC" = 1 ] || [ "$DO_MAX" = 1 ] || [ "$CUR_DEFAULT" != "recommended" ]; then
  transform "$CONF" "$TMP/new.conf"
  sudo cp -- "$TMP/new.conf" "$CONF"
fi
sudo systemctl restart nvfancontrol
echo "  restarted nvfancontrol — active state:"
sudo nvfancontrol -q 2>&1 | grep -iE 'profile|governor' | sed 's/^/    /' || true
echo "  done — fan mode: recommended"
