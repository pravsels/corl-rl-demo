"""Reads the four-button module over USB serial. This is the reward signal.

The module prints a state line at 9600 baud once a second, and at once whenever
a button changes:

    Button States: D2=1 D3=1 D4=1 D5=1

Its inputs use pull-ups, so 0 means the button is down. Other lines (the startup
banner, the ``buttons_module`` heartbeat) are ignored. The module debounces for
50 ms, so a contact shorter than that may not be reported.

A press is over long before a control loop looks again, so presses are latched
as they arrive and kept until the next episode begins.
"""

from __future__ import annotations

import argparse
import re
import threading
import time
from collections.abc import Callable
from typing import Protocol

BAUD = 9600
# Found by pressing each button on this module (2026-11-10).
COLOURS = {"D2": "green", "D3": "red", "D4": "yellow", "D5": "white"}
GREEN = "D2"
_PREFIX = "Button States:"
_PIN = re.compile(r"(D\d+)=([01])")


def parse_line(line: str) -> dict[str, bool] | None:
    """Which pins are down according to one line; None if it is not a state line."""
    if not line.startswith(_PREFIX):
        return None
    pins = {pin: level == "0" for pin, level in _PIN.findall(line)}
    return pins or None


class PressLatch:
    """Remembers which buttons were pressed since the episode began, in order.

    A press is a button going from up to down. A button already held when the
    episode begins does not count until it is released and pressed again.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._down: dict[str, bool] = {}
        self._presses: list[str] = []

    def update(self, down: dict[str, bool]) -> list[tuple[str, bool]]:
        """Take in one reading. Returns the pins that changed, as (pin, is_down)."""
        changes = []
        with self._lock:
            for pin, is_down in down.items():
                if is_down != self._down.get(pin, False):
                    changes.append((pin, is_down))
                    if is_down and pin not in self._presses:
                        self._presses.append(pin)
                self._down[pin] = is_down
        return changes

    def begin_episode(self) -> None:
        with self._lock:
            self._presses.clear()

    def presses(self) -> tuple[str, ...]:
        """Pins pressed this episode, first press first."""
        with self._lock:
            return tuple(self._presses)

    def down(self) -> dict[str, bool]:
        with self._lock:
            return dict(self._down)


class _Stream(Protocol):
    def readline(self) -> bytes: ...
    def close(self) -> None: ...


class ButtonModule:
    """Reads the module on a background thread and latches presses.

    ``seconds_since_reading`` is how the caller tells a quiet module from an
    unplugged one: the module reports every second, so several seconds of
    silence means there is no reward signal and rollouts should pause.

    ``on_change(pin, is_down, when)`` is called on the reader thread for every
    change, with the ``time.monotonic()`` at which its line arrived.
    """

    def __init__(
        self,
        stream: _Stream,
        on_change: Callable[[str, bool, float], None] | None = None,
    ) -> None:
        self._stream = stream
        self._on_change = on_change
        self.latch = PressLatch()
        # Until the first state line, silence is counted from now.
        self._last_reading = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    @classmethod
    def open(
        cls,
        port: str,
        baud: int = BAUD,
        on_change: Callable[[str, bool, float], None] | None = None,
    ) -> "ButtonModule":
        import serial

        # Opening the port resets the module's microcontroller; it reports again
        # about two seconds later.
        return cls(serial.Serial(port, baud, timeout=0.5), on_change)

    def _read(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self._stream.readline()
            except Exception:  # unplugged mid-read; silence is reported by the caller
                return
            down = parse_line(raw.decode("ascii", "replace").strip())
            if down is not None:
                now = time.monotonic()
                changes = self.latch.update(down)
                self._last_reading = now
                if self._on_change is not None:
                    for pin, is_down in changes:
                        self._on_change(pin, is_down, now)

    def seconds_since_reading(self) -> float:
        """Seconds since the last state line, or since opening if there was none."""
        return time.monotonic() - self._last_reading

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._stream.close()

    def __enter__(self) -> "ButtonModule":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _label(pin: str) -> str:
    return f"{pin} ({COLOURS[pin]})" if pin in COLOURS else pin


class TapReport:
    """Prints each press and release as it arrives, and tallies them per pin.

    The tally answers the tap question: tap a button five times, and it should
    show five presses with holds well under the one-second report interval.
    """

    def __init__(self, start: float, echo: Callable[[str], None] = print) -> None:
        self._start = start
        self._echo = echo
        self._pressed_at: dict[str, float] = {}
        self._count: dict[str, int] = {}
        self._shortest: dict[str, float] = {}

    def __call__(self, pin: str, is_down: bool, when: float) -> None:
        stamp = f"{when - self._start:7.2f} s  {_label(pin)}"
        if is_down:
            self._pressed_at[pin] = when
            self._count[pin] = self._count.get(pin, 0) + 1
            self._echo(f"{stamp} pressed")
        elif pin in self._pressed_at:
            held = when - self._pressed_at.pop(pin)
            self._shortest[pin] = min(held, self._shortest.get(pin, held))
            self._echo(f"{stamp} released after {held:.2f} s")

    def summary(self) -> list[str]:
        lines = []
        for pin in sorted(self._count):
            count = self._count[pin]
            presses = f"{count} press{'' if count == 1 else 'es'}"
            if pin in self._shortest:
                lines.append(f"{_label(pin)}: {presses}, shortest hold {self._shortest[pin]:.2f} s")
            else:
                lines.append(f"{_label(pin)}: {presses}, still down")
        return lines


def main(argv: list[str] | None = None) -> None:
    """Print each press and release, to check the module and its pin colours."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--port", required=True, help="e.g. /dev/cu.usbserial-110")
    parser.add_argument("--seconds", type=float, help="stop after this long (default: Ctrl-C)")
    args = parser.parse_args(argv)

    start = time.monotonic()
    report = TapReport(start, echo=lambda line: print(line, flush=True))
    with ButtonModule.open(args.port, on_change=report) as module:
        print(f"listening on {args.port}; press buttons (Ctrl-C to stop)", flush=True)
        deadline = None if args.seconds is None else start + args.seconds
        warned = False
        try:
            while deadline is None or time.monotonic() < deadline:
                time.sleep(0.1)
                silent = module.seconds_since_reading() > 5.0
                if silent and not warned:
                    print("no state line for 5 s: wrong port, or the module is unplugged")
                warned = silent
        except KeyboardInterrupt:
            pass
        pins = sorted(module.latch.down())
    print(f"\npins seen: {', '.join(pins) or 'none'}")
    for line in report.summary():
        print(line)


if __name__ == "__main__":
    main()
