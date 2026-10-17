"""``mjolnir autoresearch`` — an autonomous kernel-research loop with a ledger.

After karpathy/autoresearch: an agent edits ONE file, runs a FIXED,
time-boxed evaluation, keeps the change if the metric improved and resets it
otherwise — forever, while every experiment lands on a ledger. Here:

* the editable file  — ``src/mjolnir/mjolnir/t5_decode.py`` (the kernel);
* the fixed harness  — ``mjolnir autoresearch evaluate``: the quick
  correctness subset (``mjolnir verify correctness --quick``) + an ncu sweep
  of FlashInfer vs Mjolnir (harness default) at every L;
* the metric         — per L, FlashInfer time / Mjolnir time, both measured
  in the SAME run (drift on the shared Thor cancels out); the score is their
  geomean over L;
* the decision       — made by this module, not the agent (:func:`judge`),
  on the per-L ratios against the best kept row, not only on the geomean:
  ``keep`` needs a geomean gain > ``--min-gain`` (2%) that does not rest on a
  single L and no more than one L slower than the noise band; L where the
  FlashInfer reference itself jumped are set aside; ``noisy`` when the
  reference was unstable at too many L (re-evaluate the same commit),
  ``crash`` if a stage failed to produce data.
* the ledger         — ``research/ledger.jsonl`` (+ a ``ledger.tsv`` view):
  one row per experiment with the hypothesis, the agents involved, the
  sources (URLs / arXiv ids), the commit, the per-L numbers and the verify
  run ids. Untracked while the loop runs (a discard is a ``git reset
  --hard``, which must not rewind the notebook); ``mjolnir autoresearch
  export`` snapshots it into ``research/runs/<tag>/`` for committing.

The loop itself is an OpenCode session (``mjolnir autoresearch run``)
driving ``.opencode/agent/autoresearch.md`` (the orchestrator; the helpers it delegates to are user-defined OpenCode agents)
→ ``web-researcher`` / ``arxiv-researcher`` (ideas + sources) →
``python-engineer`` (the change) → ``debugger`` (crashes, correctness
failures). The program they follow is ``research/program.md``.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from mjolnir_cli.config import load_layout
from mjolnir_cli.legs import KERNEL
from mjolnir_cli.verify import results

console = Console()
app = typer.Typer(
    help="Autonomous kernel research: OpenCode agents propose, implement and test kernel changes; "
    "every experiment is scored by a fixed harness and recorded on the ledger.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

KERNEL_FILE = Path("src/mjolnir/mjolnir/t5_decode.py")
LEDGER_DIR = Path("research")
TSV_COLUMNS = ["id", "ts", "commit", "status", "score", "best_before", "correctness", "agent", "hypothesis", "reason"]


# ── ledger ──────────────────────────────────────────────────────────────────


def ledger_path(root: Path) -> Path:
    return root / LEDGER_DIR / "ledger.jsonl"


def load_ledger(root: Path) -> list[dict]:
    p = ledger_path(root)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def best_kept(rows: list[dict]) -> dict | None:
    kept = [r for r in rows if r.get("status") in ("keep", "baseline") and r.get("score")]
    return max(kept, key=lambda r: r["score"]) if kept else None


def append_ledger(root: Path, row: dict) -> None:
    d = root / LEDGER_DIR
    d.mkdir(parents=True, exist_ok=True)
    with (d / "ledger.jsonl").open("a") as f:
        f.write(json.dumps(row, sort_keys=False) + "\n")
    tsv = d / "ledger.tsv"
    new = not tsv.exists()
    with tsv.open("a") as f:
        if new:
            f.write("\t".join(TSV_COLUMNS) + "\n")
        vals = []
        for c in TSV_COLUMNS:
            v = row.get(c, "")
            if isinstance(v, float):
                v = f"{v:.4f}"
            vals.append(str(v).replace("\t", " ").replace("\n", " "))
        f.write("\t".join(vals) + "\n")


def decide(score: float | None, correct: bool, best: float | None, min_gain: float) -> str:
    """The fixed keep/discard rule (the agent never grades its own work)."""
    if score is None:
        return "crash"
    if not correct:
        return "discard"
    if best is None:
        return "baseline"
    return "keep" if score > best * (1 + min_gain) else "discard"


NOISE = 0.03  # a same-run ratio moves ±3% between runs (verify.stages.NOISE_PCT)
REF_JUMP = 0.15  # the FlashInfer leg moved more than this vs the best row's run: that L is suspect
MAX_SUSPECT = 2  # more suspect L than this and the measurement itself is rejected ("noisy")


def _gm(xs) -> float:
    xs = list(xs)
    return results.geomean(xs) if xs else 1.0


def _pct(x: float) -> str:
    return f"{(x - 1) * 100:+.1f}%"


def _L(k) -> str:
    k = int(k)
    return f"{k // 1024}K" if k >= 1024 and k % 1024 == 0 else str(k)


def judge(row: dict, best: dict | None, min_gain: float = 0.02, noise: float = NOISE) -> tuple[str, str, dict]:
    """The fixed keep/discard rule, per L. Returns (status, reason, rel) where
    ``rel[L]`` = this row's FlashInfer÷Mjolnir ratio ÷ the best kept row's.

    A single global geomean can be moved by one L — a FlashInfer outlier at
    256K once turned a no-op into "+8%" — so a keep needs all of:

    1. the geomean of ``rel`` over the trusted L beats ``1 + min_gain``;
    2. it still beats ``1 + min_gain / 2`` with the single best L left out
       (the win does not rest on one context length);
    3. at most one L is slower than ``1 - noise`` (one is tolerated: 16K is
       bimodal run to run).

    An L is *suspect* — set aside — when the FlashInfer reference time moved
    by more than ``REF_JUMP`` against the best row's run (the reference never
    changes between experiments, so a jump there is noise in the measurement,
    not a property of the kernel). More than ``MAX_SUSPECT`` suspect L and the
    run is ``noisy``: nothing is decided, evaluate the same commit again.
    """
    score, correct = row.get("score"), row.get("correctness") == "pass"
    if score is None:
        return "crash", "no score (a stage failed)", {}
    if not correct:
        return "discard", "correctness failed", {}
    if best is None:
        return "baseline", "the starting kernel", {}
    new, old = row.get("per_L") or {}, best.get("per_L") or {}
    common = [L for L in new if L in old and old[L]]
    if not common:  # rows written before per-L judging: the plain geomean rule
        st = decide(score, correct, best.get("score"), min_gain)
        return st, f"geomean {_pct(score / best['score'])} (no per-L reference)", {}
    rel = {L: new[L] / old[L] for L in common}
    fi_new, fi_old = row.get("fi_us") or {}, best.get("fi_us") or {}
    suspect = [L for L in common if L in fi_new and fi_old.get(L) and abs(fi_new[L] / fi_old[L] - 1) > REF_JUMP]
    if len(suspect) > MAX_SUSPECT:
        return "noisy", "FlashInfer reference unstable at " + ", ".join(_L(L) for L in suspect) + " — re-evaluate", rel
    trusted = {L: v for L, v in rel.items() if L not in suspect}
    note = f" (set aside, reference jumped: {', '.join(_L(L) for L in suspect)})" if suspect else ""
    gain = _gm(trusted.values())
    if gain <= 1 + min_gain:
        return "discard", f"geomean {_pct(gain)} ≤ +{min_gain * 100:g}%{note}", rel
    top = max(trusted, key=trusted.get)
    loo = _gm(v for L, v in trusted.items() if L != top)
    if loo <= 1 + min_gain / 2:
        return (
            "discard",
            f"geomean {_pct(gain)} rests on {_L(top)} ({_pct(trusted[top])}): {_pct(loo)} without it{note}",
            rel,
        )
    slower = sorted((L for L, v in trusted.items() if v < 1 - noise), key=int)
    if len(slower) > 1:
        return (
            "discard",
            f"geomean {_pct(gain)} but slower at " + ", ".join(f"{_L(L)} ({_pct(trusted[L])})" for L in slower) + note,
            rel,
        )
    return "keep", f"geomean {_pct(gain)}, {_pct(loo)} without its best L ({_L(top)}){note}", rel


# ── git helpers ─────────────────────────────────────────────────────────────


def _git(root: Path, *args: str, check: bool = True) -> str:
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout.strip()


def _root() -> Path:
    return load_layout().require_repo("autoresearch")


# ── commands ────────────────────────────────────────────────────────────────


def _init(root: Path, tag: str, base: str, evaluate_baseline: bool, image: Optional[str]) -> int:
    branch = f"autoresearch/{tag}"
    if _git(root, "branch", "--list", branch):
        console.print(f"[red]{branch} already exists — pick a fresh tag (a run never reuses a branch)[/red]")
        return 4
    if _git(root, "status", "--porcelain", "--", str(KERNEL_FILE)):
        console.print(f"[red]{KERNEL_FILE} has uncommitted changes — commit or stash them first[/red]")
        return 4
    if load_ledger(root):
        console.print(
            "[red]research/ledger.jsonl holds a previous run — `mjolnir autoresearch export` it, then delete it[/red]"
        )
        return 4
    _git(root, "checkout", "-b", branch, base)
    console.print(f"[green]on {branch}[/green] (from {base})")
    if evaluate_baseline:
        return _evaluate(root, hypothesis="baseline: the starting kernel", agent="human", sources="", image=image)
    return 0


@app.command()
def init(
    tag: str = typer.Argument(..., help="run tag, e.g. oct11 — the branch is autoresearch/<tag>"),
    base: str = typer.Option("main", "--base", help="branch to start from"),
    evaluate_baseline: bool = typer.Option(True, "--baseline/--no-baseline", help="score the starting kernel now"),
    image: Optional[str] = typer.Option(None, "--image"),
):
    """Create the research branch and score the starting kernel (the baseline row)."""
    raise typer.Exit(_init(_root(), tag, base, evaluate_baseline, image))


def _evaluate(
    root: Path,
    hypothesis: str,
    agent: str,
    sources: str,
    image: Optional[str],
    min_gain: float = 0.02,
    ls: Optional[str] = None,
    iters: int = 50,
    gate: bool = True,
    skip_correctness: bool = False,
) -> int:
    from mjolnir_cli.verify.cli import PREFLIGHT_RC, execute
    from mjolnir_cli.verify.stages import ALL_LS

    rows = load_ledger(root)
    best = best_kept(rows)
    commit = _git(root, "rev-parse", "--short", "HEAD")
    dirty = bool(_git(root, "status", "--porcelain", "--", str(KERNEL_FILE)))
    if dirty:
        console.print(
            "[yellow]the kernel has uncommitted changes — commit the experiment first "
            "(the ledger records commits)[/yellow]"
        )
        return 4
    correct, corr_run = True, None
    if not skip_correctness:
        rc, doc = execute(
            "correctness",
            {"dtypes": "fp8 bf16", "match": "", "fail_fast": True, "timeout": 300.0, "quick": True},
            {},
            image=image,
            ls=None,
            iters=iters,
            gate=gate,
            dry_run=False,
            timeout=300.0,
            save=True,
            plot=False,
        )
        if rc == PREFLIGHT_RC:
            return rc  # no docker / no image: an environment problem, not an experiment — no ledger row
        correct = rc == 0
        corr_run = doc["run_id"] if doc else None
    score, per_L, sweep_run, fi_us, mj_us = None, {}, None, {}, {}
    if correct:
        rc, doc = execute(
            "sweep",
            {"legs": KERNEL["autoresearch.evaluate"]},
            {},  # never a headline run
            image=image,
            ls=ls,
            iters=iters,
            gate=gate,
            dry_run=False,
            timeout=300.0,
            save=True,
            plot=False,
        )
        if doc:
            by = results.series_by_id(doc)
            if "fi" in by and "mjolnir" in by:
                sp = results.speedup(by["fi"], by["mjolnir"])
                want = set(int(x) for x in (ls or " ".join(map(str, ALL_LS))).replace(",", " ").split())
                if sp and set(sp) >= want:
                    score = results.geomean(sp.values())
                    per_L = {str(k): round(v, 4) for k, v in sp.items()}
                    fi_us = {str(k): round(v, 2) for k, v in results.point_map(by["fi"]).items()}
                    mj_us = {str(k): round(v, 2) for k, v in results.point_map(by["mjolnir"]).items()}
            sweep_run = doc["run_id"]
    row = {
        "id": len(rows) + 1,
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": commit,
        "status": None,
        "reason": None,
        "score": round(score, 4) if score else None,
        "best_before": best["score"] if best else None,
        "min_gain": min_gain,
        "correctness": "pass" if correct else "fail",
        "per_L": per_L,
        "fi_us": fi_us,
        "mj_us": mj_us,
        "agent": agent,
        "hypothesis": hypothesis,
        "sources": [s.strip() for s in sources.split(",") if s.strip()],
        "verify_runs": {"correctness": corr_run, "sweep": sweep_run},
    }
    status, reason, rel = judge(row, best, min_gain)
    row["status"], row["reason"] = status, reason
    row["vs_best"] = {L: round(v, 4) for L, v in rel.items()}
    append_ledger(root, row)
    color = STATUS_COLORS[status]
    console.print(
        f"\n[bold {color}]{status.upper()}[/bold {color}]  score {row['score']}  "
        f"(best kept {row['best_before']})  correctness {row['correctness']}  commit {commit}\n  {reason}"
    )
    if rel:
        console.print(
            "  vs best, per L: "
            + "  ".join(f"{_L(L)} {_pct(v)}" for L, v in sorted(rel.items(), key=lambda kv: int(kv[0])))
        )
    if status in ("discard", "crash"):
        console.print(f"[dim]  revert: git reset --hard {best['commit'] if best else 'HEAD~1'}[/dim]")
    elif status == "noisy":
        console.print("[dim]  the measurement was rejected, not the idea: evaluate the same commit again[/dim]")
    render_progress(root)
    return 0


STATUS_COLORS = {"keep": "green", "baseline": "cyan", "discard": "yellow", "crash": "red", "noisy": "magenta"}
PROGRESS = LEDGER_DIR / "progress.png"


def render_progress(root: Path, rows: list[dict] | None = None, out: Path | None = None) -> Path | None:
    """research/progress.png (+ -dark): every experiment on one chart, the
    kept staircase, and the per-L split vs the baseline. Best effort — a
    missing matplotlib never fails an evaluation."""
    rows = rows if rows is not None else load_ledger(root)
    try:
        from mjolnir_cli import charts, theme

        plt = charts.pyplot()
        out = out or root / PROGRESS
        for mode in theme.MODES:
            fig = charts.chart_autoresearch(rows, mode)
            if fig is None:
                return None
            charts.save_png(fig, out.with_name(f"{out.stem}{mode.suffix}{out.suffix}"))
            plt.close(fig)
        return out
    except Exception as e:  # noqa: BLE001 — a chart must never cost a ledger row
        console.print(f"[dim]progress chart skipped: {e!r}[/dim]")
        return None


def latest_rows(root: Path) -> list[dict]:
    """The live ledger, else the newest exported run (research/runs/<tag>/)."""
    rows = load_ledger(root)
    if rows:
        return rows
    runs = sorted((root / LEDGER_DIR / "runs").glob("*/ledger.jsonl"), key=lambda p: p.stat().st_mtime)
    return [json.loads(x) for x in runs[-1].read_text().splitlines() if x.strip()] if runs else []


@app.command()
def evaluate(
    hypothesis: str = typer.Option(..., "--hypothesis", "-m", help="what this experiment tries, one line"),
    agent: str = typer.Option("autoresearch", "--agent", help="who proposed/implemented it"),
    sources: str = typer.Option("", "--sources", help="comma-separated URLs / arXiv ids behind the idea"),
    min_gain: float = typer.Option(0.02, "--min-gain", help="required geomean gain over the best kept score"),
    ls: Optional[str] = typer.Option(None, "--ls", help="L values (default: all 8; a subset is never a keep)"),
    iters: int = typer.Option(50, "--iters"),
    gate: bool = typer.Option(True, "--gate/--no-gate"),
    image: Optional[str] = typer.Option(None, "--image"),
):
    """Score the committed kernel: quick correctness + FlashInfer vs Mjolnir sweep;
    append the result and the fixed keep/discard decision to the ledger."""
    raise typer.Exit(_evaluate(_root(), hypothesis, agent, sources, image, min_gain, ls, iters, gate))


@app.command()
def ledger(
    tail: int = typer.Option(20, "--tail"),
    as_json: bool = typer.Option(False, "--json", "-j"),
):
    """Show the ledger (newest last)."""
    rows = load_ledger(_root())
    if as_json:
        typer.echo(json.dumps(rows[-tail:], indent=2))
        return
    if not rows:
        console.print("[yellow]empty ledger — start with: mjolnir autoresearch init <tag>[/yellow]")
        return
    best = best_kept(rows)
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False)
    for c in ("#", "when", "commit", "status", "score", "agent", "hypothesis", "why"):
        t.add_column(c, justify="right" if c in ("#", "score") else "left")
    colors = STATUS_COLORS
    for r in rows[-tail:]:
        st = r.get("status", "")
        mark = " ★" if best and r is best else ""
        t.add_row(
            str(r["id"]),
            r["ts"][5:16].replace("T", " "),
            r["commit"],
            f"[{colors.get(st, 'white')}]{st}[/]",
            f"{r['score']:.4f}{mark}" if r.get("score") else "—",
            r.get("agent", ""),
            r.get("hypothesis", "")[:60],
            (r.get("reason") or "")[:60],
        )
    console.print(t)


@app.command()
def status():
    """Best kept score, experiment counts, the current branch."""
    root = _root()
    rows = load_ledger(root)
    best = best_kept(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    console.print(
        f"branch {_git(root, 'rev-parse', '--abbrev-ref', 'HEAD')} · {len(rows)} experiments · "
        + " · ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    )
    if best:
        console.print(
            f"best: [bold]{best['score']:.4f}×[/bold] vs FlashInfer (geomean over L) at {best['commit']} "
            f"— {best['hypothesis']}"
        )


@app.command()
def chart():
    """Re-render research/progress.png from the live ledger (or the newest exported run)."""
    root = _root()
    p = render_progress(root, latest_rows(root))
    console.print(f"[green]{p}[/green]" if p else "[yellow]no experiments yet[/yellow]")


@app.command()
def run(
    tag: Optional[str] = typer.Option(None, "--tag", help="start a new run first (init <tag>)"),
    model: Optional[str] = typer.Option(None, "--model", help="OpenCode model (provider/model) for the orchestrator"),
    prompt: str = typer.Option(
        "Read research/program.md and run the loop: setup is done when the ledger has a baseline row. "
        "Never stop to ask; record every experiment with `mjolnir autoresearch evaluate`.",
        "--prompt",
    ),
):
    """Launch the loop: an OpenCode session with the `autoresearch` agent."""
    root = _root()
    if shutil.which("opencode") is None:
        console.print("[red]opencode not found — install it: https://opencode.ai (npm i -g opencode-ai)[/red]")
        raise typer.Exit(2)
    if tag:
        rc = _init(root, tag, "main", True, None)
        if rc != 0:
            raise typer.Exit(rc)
    elif not load_ledger(root):
        console.print("[yellow]no ledger yet — run: mjolnir autoresearch init <tag>  (or pass --tag)[/yellow]")
        raise typer.Exit(4)
    cmd = ["opencode", "run", "--agent", "autoresearch"]
    if model:
        cmd += ["--model", model]
    cmd += [prompt]
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    raise typer.Exit(subprocess.call(cmd, cwd=root))


@app.command()
def export(
    tag: Optional[str] = typer.Option(None, "--tag", help="run tag (default: from the autoresearch/<tag> branch)"),
):
    """Snapshot the ledger into research/runs/<tag>/ (ledger.jsonl, ledger.tsv,
    SUMMARY.md) — commit that to keep the run's notebook with the repo."""
    root = _root()
    rows = load_ledger(root)
    if not rows:
        console.print("[yellow]empty ledger[/yellow]")
        raise typer.Exit(4)
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    tag = tag or (branch.split("/", 1)[1] if branch.startswith("autoresearch/") else branch)
    out = root / LEDGER_DIR / "runs" / tag
    out.mkdir(parents=True, exist_ok=True)
    for name in ("ledger.jsonl", "ledger.tsv"):
        src = root / LEDGER_DIR / name
        if src.exists():
            shutil.copy(src, out / name)
    best = best_kept(rows)
    base = next((r for r in rows if r["status"] == "baseline"), None)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines = [
        f"# autoresearch run `{tag}`",
        "",
        f"- experiments: {len(rows)} (" + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + ")",
        f"- baseline: {base['score']}× vs FlashInfer at `{base['commit']}`" if base and base.get("score") else "",
        f"- best: **{best['score']}×** at `{best['commit']}` — {best['hypothesis']}" if best else "",
        "",
        "![progress](progress.png)" if render_progress(root, rows, out / "progress.png") else "",
        "",
        "| # | commit | status | score | why | agent | hypothesis | sources |",
        "|--:|---|---|--:|---|---|---|---|",
    ]
    for r in rows:
        src = " ".join(f"[{i + 1}]({u})" if u.startswith("http") else u for i, u in enumerate(r.get("sources", [])))
        lines.append(
            f"| {r['id']} | `{r['commit']}` | {r['status']} | {r.get('score') or '—'} "
            f"| {(r.get('reason') or '').replace('|', '/')} | {r.get('agent', '')} "
            f"| {r.get('hypothesis', '').replace('|', '/')} | {src} |"
        )
    (out / "SUMMARY.md").write_text("\n".join(x for x in lines if x is not None) + "\n")
    console.print(f"[green]exported {len(rows)} rows -> {out}[/green]")
