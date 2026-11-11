import time

import pytest

from corl_rl_demo.buttons import ButtonModule
from corl_rl_demo.rig import CheckFailed, check_buttons

IDLE = "Button States: D2=1 D3=1 D4=1 D5=1"
D3_DOWN = "Button States: D2=1 D3=0 D4=1 D5=1"


class FakeStream:
    def __init__(self, lines):
        self._lines = [line.encode() + b"\r\n" for line in lines]

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        time.sleep(0.01)
        return b""

    def close(self):
        pass


def test_idle_module_passes():
    with ButtonModule(FakeStream(["buttons_module", IDLE])) as module:
        assert check_buttons(module, wait=1.0) == "4 buttons, all up"


def test_silent_module_fails():
    with ButtonModule(FakeStream(["buttons_module"])) as module:
        with pytest.raises(CheckFailed, match="no state line"):
            check_buttons(module, wait=0.2)


def test_held_button_fails_and_is_named():
    with ButtonModule(FakeStream([D3_DOWN])) as module:
        with pytest.raises(CheckFailed, match=r"D3 \(red\) down"):
            check_buttons(module, wait=1.0)


def test_missing_pins_fail():
    with ButtonModule(FakeStream(["Button States: D2=1 D3=1"])) as module:
        with pytest.raises(CheckFailed, match="expected pins"):
            check_buttons(module, wait=1.0)
