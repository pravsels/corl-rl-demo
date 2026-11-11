import time

from corl_rl_demo.buttons import COLOURS, GREEN, ButtonModule, PressLatch, TapReport, parse_line

# Lines captured from the module.
BANNER = [
    "Starting input monitor (D2..D5) - output 1=HIGH, 0=LOW",
    "USE_INTERNAL_PULLUPS=true (active=LOW)",
    "DEBOUNCE_MS=50",
    "buttons_module",
]
IDLE = "Button States: D2=1 D3=1 D4=1 D5=1"
D4_DOWN = "Button States: D2=1 D3=1 D4=0 D5=1"


def test_state_line_reports_low_as_down():
    assert parse_line(IDLE) == {"D2": False, "D3": False, "D4": False, "D5": False}
    assert parse_line(D4_DOWN) == {"D2": False, "D3": False, "D4": True, "D5": False}


def test_other_lines_are_ignored():
    for line in BANNER + ["", "Button States:", "\x00\x00"]:
        assert parse_line(line) is None


def test_latch_keeps_a_press_after_release():
    latch = PressLatch()
    latch.update(parse_line(IDLE))
    latch.begin_episode()
    latch.update(parse_line(D4_DOWN))
    latch.update(parse_line(IDLE))
    assert latch.presses() == ("D4",)


def test_latch_orders_presses_and_counts_each_pin_once():
    latch = PressLatch()
    latch.update({"D2": False, "D4": False})
    latch.update({"D2": True, "D4": False})
    latch.update({"D2": False, "D4": True})
    latch.update({"D2": True, "D4": False})
    assert latch.presses() == ("D2", "D4")


def test_button_held_at_episode_start_is_not_a_press():
    latch = PressLatch()
    latch.update(parse_line(D4_DOWN))
    latch.begin_episode()
    latch.update(parse_line(D4_DOWN))
    assert latch.presses() == ()
    latch.update(parse_line(IDLE))
    latch.update(parse_line(D4_DOWN))
    assert latch.presses() == ("D4",)


def test_green_is_one_of_the_four_pins_the_module_reports():
    assert set(COLOURS) == set(parse_line(IDLE))
    assert COLOURS[GREEN] == "green"


def test_update_returns_the_pins_that_changed():
    latch = PressLatch()
    assert latch.update(parse_line(IDLE)) == []
    assert latch.update(parse_line(D4_DOWN)) == [("D4", True)]
    assert latch.update(parse_line(D4_DOWN)) == []
    assert latch.update(parse_line(IDLE)) == [("D4", False)]


class FakeStream:
    def __init__(self, lines):
        self._lines = [line.encode() + b"\r\n" for line in lines]
        self.closed = False

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        time.sleep(0.01)
        return b""

    def close(self):
        self.closed = True


def _wait_for(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_module_latches_presses_from_the_stream():
    stream = FakeStream(BANNER + [IDLE, D4_DOWN, IDLE])
    with ButtonModule(stream) as module:
        assert _wait_for(lambda: module.latch.presses() == ("D4",))
    assert stream.closed


def test_module_reports_each_change_with_its_arrival_time():
    changes = []
    stream = FakeStream(BANNER + [IDLE, D4_DOWN, IDLE])
    with ButtonModule(stream, on_change=lambda *change: changes.append(change)):
        assert _wait_for(lambda: len(changes) == 2)
    (pin_a, down_a, t_a), (pin_b, down_b, t_b) = changes
    assert (pin_a, down_a, pin_b, down_b) == ("D4", True, "D4", False)
    assert t_a <= t_b <= time.monotonic()


def test_tap_report_counts_presses_and_times_the_shortest_hold():
    report = TapReport(start=100.0, echo=lambda _: None)
    report("D4", True, 101.0)
    report("D4", False, 101.3)
    report("D4", True, 102.0)
    report("D4", False, 102.1)
    report("D2", True, 103.0)
    assert report.summary() == [
        "D2 (green): 1 press, still down",
        "D4 (yellow): 2 presses, shortest hold 0.10 s",
    ]


def test_silence_grows_until_a_state_line_arrives():
    with ButtonModule(FakeStream(BANNER)) as module:
        time.sleep(0.2)
        assert module.seconds_since_reading() >= 0.2

    with ButtonModule(FakeStream([IDLE])) as module:
        assert _wait_for(lambda: module.latch.down() != {})
        time.sleep(0.2)
        # Counted from the state line, not from opening: the line arrived first.
        assert 0.2 <= module.seconds_since_reading() < 2.0
