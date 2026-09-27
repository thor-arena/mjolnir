# I want to pre-configure a fresh Jetson Thor for mjolnir

**Story:** "I got a fresh Thor out of the box (or reflashed it). What do I
run to bring the *host* to the state the rest of the CLI assumes — headless,
Docker with the NVIDIA runtime, swap, fan profiles, locked clocks, MAXN
power mode?"

## One command

```bash
mjolnir hw setup
```

That runs `scripts/hw/setup-thor.sh` — the port of the standalone Thor setup
script, with the tuned fan profiles merged in — and walks the box through
the 10 steps (in order):

| # | Step | What it does |
|---|---|---|
| 1 | `gui` | boot target → `multi-user.target` (headless server; GUI off) |
| 2 | `upgrade` | `apt update` + `full-upgrade` (+ JetPack tools, curl, wget) |
| 3 | `pip` | `python3-pip`/setuptools/wheel, get-pip.py fallback, `~/.local/bin` on PATH |
| 4 | `docker` | docker-ce + `nvidia-container-toolkit`, user in the `docker` group, `nvidia` as the **default** runtime in `daemon.json`, service started + enabled at boot |
| 5 | `jtop` | `jetson-stats` (the thermal/telemetry monitor) |
| 6 | `memory` | 32 GB swap file at `/mnt` (zRAM disabled), fstab-persisted |
| 7 | `fan` | install the tuned `recommended` + `max` fan profiles into `/etc/nvfancontrol.conf` (per-profile gated — never duplicated; untouched conf backed up to `.bck` on first run), then select the default profile — **`recommended`** by default |
| 8 | `host` | disable `nvargus-daemon`, `cups`, `ModemManager` (camera/print/modem — dead weight on a server) |
| 9 | `clocks` | `jetson_clocks.service` — locks CPU/GPU/EMC at max |
| 10 | `power` | `nvpmodel` → MAXN (or MAXN_SUPER if present) — **applies after reboot** |

Every step is **idempotent** — re-running is a no-op where already
configured, so it's safe to run after a reflash or a partial setup.

## Options

```bash
mjolnir hw setup --dry-run             # print the 10-step plan, change nothing
mjolnir hw setup --yes                  # skip the "continue?" prompt
mjolnir hw setup --no-upgrade          # skip the apt full-upgrade
mjolnir hw setup --keep-gui            # don't switch the boot target to headless
mjolnir hw setup --fan-profile max     # recommended (default) | max | cool | quiet
mjolnir hw setup --swap-size 16        # swap file size in GB (default 32)
mjolnir hw setup --skip jtop,fan       # skip individual steps
mjolnir hw setup --reboot              # reboot at the end if one is required
```

`--skip` names: `gui,upgrade,pip,docker,jtop,memory,fan,host,clocks,power`
(in run order; `docker` covers install + group + runtime + service; `fan`
covers both the profile install and the default selection).

The stock `cool`/`quiet` profiles ship with L4T; `recommended` (balanced)
and `max` (sustained bench cooling) are the tuned ones this step installs.
For benchmarks, `--fan-profile max`; for daily serving, leave the default
`recommended`.

## Inspecting (no sudo)

```bash
mjolnir hw status        # installed fan profiles, the selected default,
                         # and the .bck rollback backup
sudo nvfancontrol -q     # the daemon's live state (needs sudo)
```

## What it does NOT do

- No state file, no model/image selection — that's `mjolnir model` /
  `mjolnir image` ([settings-precedence.md](settings-precedence.md)).
- It never touches the running vLLM container. If you're running
  `mjolnir serve`, expect the Docker step to restart the docker *service*
  (the container comes back with it) and the clocks/power steps to want a
  reboot — schedule setup deliberately, like `mjolnir bench ab`.
- It needs `sudo` and a TTY for the prompts (pass `--yes` from scripts; the
  reboot prompt becomes a printed note without a TTY, or `--reboot` to act).

## First boot afterwards

```bash
mjolnir serve up          # the baked-in Qwen3.8-27B NVFP4 + FA4 GEMV defaults
mjolnir gate --once       # confirm the clean-window gate sees the server
```

## Where it lives

`scripts/hw/setup-thor.sh` (self-contained, Thor-only constants: SoM=thor,
128 GB, NVIDIA carrier) driven by `src/mjolnir/thor.py` — the CLI maps its
flags to the script's options; `--dry-run` previews the plan. The conf path
is overridable for testing via `MJOLNIR_NVFANCONF` (also honored by
`mjolnir hw status`).
