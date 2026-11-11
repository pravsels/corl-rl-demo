"""Reads the four-button module over USB serial. This is the reward signal.

The module prints a state line at 9600 baud, at least once a second:

    Button States: D2=1 D3=1 D4=1 D5=1

Its inputs use pull-ups, so 0 means the button is down. Other lines (the startup
banner, the ``buttons_module`` heartbeat) are ignored.

A press is over long before a control loop looks again, so presses are latched
as they arrive and kept until the next episode begins.
"""

from __future__ import annotations

import argparse
import re
import threading
import time
from typing import Protocol

BAUD = 9600
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

    def update(self, down: dict[str, bool]) -> None:
        with self._lock:
            for pin, is_down in down.items():
                if is_down and not self._down.get(pin, False) and pin not in self._presses:
                    self._presses.append(pin)
                self._down[pin] = is_down

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
    """

    def __init__(self, stream: _Stream) -> None:
        self._stream = stream
        self.latch = PressLatch()
        # Until the first state line, silence is counted from now.
        self._last_reading = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    @classmethod
    def open(cls, port: str, baud: int = BAUD) -> "ButtonModule":
        import serial

        # Opening the port resets the module's microcontroller; it reports again
        # about two seconds later.
        return cls(serial.Serial(port, baud, timeout=0.5))

    def _read(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self._stream.readline()
            except Exception:  # unplugged mid-read; silence is reported by the caller
                return
            down = parse_line(raw.decode("ascii", "replace").strip())
            if down is not None:
                self.latch.update(down)
                self._last_reading = time.monotonic()

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


def main(argv: list[str] | None = None) -> None:
    """Print each press and release. Use it to find which pin is which colour."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--port", required=True, help="e.g. /dev/cu.usbserial-110")
    parser.add_argument("--seconds", type=float, help="stop after this long (default: Ctrl-C)")
    args = parser.parse_args(argv)

    with ButtonModule.open(args.port) as module:
        print(f"listening on {args.port}; press buttons (Ctrl-C to stop)")
        deadline = None if args.seconds is None else time.monotonic() + args.seconds
        previous: dict[str, bool] = {}
        warned = False
        try:
            while deadline is None or time.monotonic() < deadline:
                time.sleep(0.02)
                down = module.latch.down()
                for pin in sorted(down):
                    if down[pin] != previous.get(pin, False):
                        print(f"{pin} {'pressed' if down[pin] else 'released'}")
                previous = down
                silent = module.seconds_since_reading() > 5.0
                if silent and not warned:
                    print("no state line for 5 s: wrong port, or the module is unplugged")
                warned = silent
        except KeyboardInterrupt:
            pass
        print(f"pins seen: {', '.join(sorted(previous)) or 'none'}")


if __name__ == "__main__":
    main()
