# I want to pre-configure a fresh Jetson Thor for mjolnir

**Story:** "I got a fresh Thor out of the box (or reflashed it). What do I
run to bring the *host* to the state the rest of the CLI assumes — headless,
Docker with the NVIDIA runtime, swap, locked clocks, MAXN power mode?"

## One command

```bash
mjolnir setup
```

That runs `scripts/setup-thor.sh` — the port of the standalone Thor setup
script — and walks the box through the 10 steps (in order):

| # | Step | What it does |
|---|---|---|
| 1 | `gui` | boot target → `multi-user.target` (headless server; GUI off) |
| 2 | `upgrade` | `apt update` + `full-upgrade` (+ JetPack tools, curl, wget) |
| 3 | `pip` | `python3-pip`/setuptools/wheel, get-pip.py fallback, `~/.local/bin` on PATH |
| 4 | `docker` | docker-ce + `nvidia-container-toolkit`, user in the `docker` group, `nvidia` as the **default** runtime in `daemon.json`, service started + enabled at boot |
| 5 | `jtop` | `jetson-stats` (the thermal/telemetry monitor) |
| 6 | `memory` | 32 GB swap file at `/mnt` (zRAM disabled), fstab-persisted |
| 7 | `fan` | `nvfancontrol` profile → `cool` (only if configured) |
| 8 | `host` | disable `nvargus-daemon`, `cups`, `ModemManager` (camera/print/modem — dead weight on a server) |
| 9 | `clocks` | `jetson_clocks.service` — locks CPU/GPU/EMC at max |
| 10 | `power` | `nvpmodel` → MAXN (or MAXN_SUPER if present) — **applies after reboot** |

Every step is **idempotent** — re-running is a no-op where already
configured, so it's safe to run after a reflash or a partial setup.

## Options

```bash
mjolnir setup --dry-run            # print the plan, change nothing
mjolnir setup --yes                # skip the "continue?" prompt
mjolnir setup --no-upgrade         # skip the apt full-upgrade
mjolnir setup --keep-gui           # don't switch the boot target to headless
mjolnir setup --fan-profile quiet  # cool (default) | quiet
mjolnir setup --swap-size 16       # swap file size in GB (default 32)
mjolnir setup --skip jtop,fan      # skip individual steps
mjolnir setup --reboot             # reboot at the end if one is required
```

`--skip` names: `gui,upgrade,pip,docker,jtop,memory,fan,host,clocks,power`
(in run order; `docker` covers install + group + runtime + service).

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

`scripts/setup-thor.sh` (self-contained, Thor-only constants: SoM=thor,
128 GB, NVIDIA carrier) driven by `src/mjolnir/thor.py` — the CLI maps its
flags to the script's options; `--dry-run` previews the resolved command.
