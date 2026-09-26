"""A minimal arrow-key selection menu on ``curses`` (stdlib — no new deps).

Used by ``mjolnir model`` / ``mjolnir image``: pick the active model
config or the vLLM image with the arrow keys and Enter. Esc/q cancels.
"""
from __future__ import annotations

import curses
import sys


class Cancelled(Exception):
    """The user cancelled the picker (Esc / q), or the terminal could not
    run the curses UI."""


_FOOTER = "↑/↓ move · enter select · esc cancel"


def _run(stdscr, choices: list[str], start: int, title: str) -> str:
    curses.curs_set(0)
    cur = start
    while True:
        stdscr.erase()
        cols = curses.COLS
        lines = curses.LINES
        stdscr.addstr(0, 0, title[: cols - 1])
        room = max(lines - 3, 1)          # title row + footer row
        top = max(0, min(cur - room + 1, len(choices) - room))
        for i in range(top, min(top + room, len(choices))):
            y = i - top + 1
            line = " " + choices[i]
            if i == cur:
                stdscr.addstr(y, 0, line[: cols - 1], curses.A_REVERSE)
            else:
                stdscr.addstr(y, 0, line[: cols - 1])
        stdscr.addstr(lines - 1, 0, _FOOTER[: cols - 1])
        stdscr.refresh()

        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            cur = max(0, cur - 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            cur = min(len(choices) - 1, cur + 1)
        elif key == curses.KEY_HOME:
            cur = 0
        elif key == curses.KEY_END:
            cur = len(choices) - 1
        elif key in (curses.KEY_ENTER, 10, 13, ord(" ")):
            return choices[cur]
        elif key in (27, ord("q")):
            raise Cancelled


def select(title: str, choices: list[str],
           default: str | None = None) -> str:
    """Arrow-key menu over ``choices``; returns the chosen item.

    Raises ``Cancelled`` on Esc/q or when the terminal can't run curses
    (non-tty, TERM=dumb). Callers should check ``isatty`` first and offer
    the explicit-argument path instead."""
    if not choices:
        raise ValueError("select() needs at least one choice")
    start = choices.index(default) if default in choices else 0
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise Cancelled
    try:
        return curses.wrapper(_run, choices, start, title)
    except (curses.error, OSError) as e:
        raise Cancelled from e
