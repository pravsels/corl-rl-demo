import time

import pytest

from corl_rl_demo.buttons import ButtonModule
from corl_rl_demo.rig import CAMERAS, CheckFailed, check_buttons, pick_button_port

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


def test_the_only_usb_serial_adapter_is_the_button_module():
    # Its name follows the USB socket, so it is found, not fixed.
    assert pick_button_port(["/dev/cu.usbserial-1120"]) == "/dev/cu.usbserial-1120"


def test_no_usb_serial_adapter_fails():
    with pytest.raises(CheckFailed, match="plugged in"):
        pick_button_port([])


def test_several_usb_serial_adapters_fail_and_are_named():
    with pytest.raises(CheckFailed, match="usbserial-10.*usbserial-20.*--button-port"):
        pick_button_port(["/dev/cu.usbserial-10", "/dev/cu.usbserial-20"])


def test_the_rig_has_a_top_and_a_wrist_camera_on_different_indices():
    assert set(CAMERAS) == {"top", "wrist"}
    assert CAMERAS["top"] != CAMERAS["wrist"]
