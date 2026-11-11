"""The rig: which port and camera is which, and a check that all of it answers.

``rig-check`` only reads. It never enables torque and never moves the arm, so it
is safe to run at any time: after plugging things in, and first thing at the venue.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from corl_rl_demo.buttons import COLOURS, ButtonModule

# macOS names the arm's port after its adapter's serial number, so it is stable.
# The button module's adapter has no serial number: its name follows the USB
# socket it is plugged into, and changes if it moves to another one.
ARM_PORT = "/dev/cu.usbmodem5AB01584891"
BUTTON_PORT = "/dev/cu.usbserial-110"

# OpenCV index of the top camera. macOS renumbers cameras when one comes or goes
# (an iPhone acting as a Continuity Camera counts), so look at the saved frame.
TOP_CAMERA = 0
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 320, 240, 30

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
HOT_C = 55


class CheckFailed(Exception):
    """One part of the rig did not answer as expected. The message says what to do."""


def check_buttons(module: ButtonModule, wait: float = 4.0) -> str:
    deadline = time.monotonic() + wait
    while not (down := module.latch.down()):
        if time.monotonic() > deadline:
            raise CheckFailed("no state line: wrong port, or the module is unplugged")
        time.sleep(0.05)
    if set(down) != set(COLOURS):
        raise CheckFailed(f"expected pins {sorted(COLOURS)}, got {sorted(down)}")
    held = [f"{pin} ({COLOURS[pin]})" for pin in sorted(down) if down[pin]]
    if held:
        raise CheckFailed(f"{', '.join(held)} down: stuck, or something is resting on it")
    return "4 buttons, all up"


def check_arm(port: str) -> str:
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    if not Path(port).exists():
        raise CheckFailed(f"{port} not found: is the arm's USB cable plugged in?")
    motors = {name: Motor(i + 1, "sts3215", MotorNormMode.DEGREES) for i, name in enumerate(JOINTS)}
    bus = FeetechMotorsBus(port=port, motors=motors)
    try:
        bus.connect()
    except Exception as error:
        if "Missing motor IDs" not in str(error):
            raise CheckFailed(str(error).splitlines()[0]) from error
        found = str(error).rsplit("\n", 1)[-1].replace("{}", "none")
        raise CheckFailed(
            f"the adapter answers but not all six motors do (found {found}): "
            "is the arm's power supply plugged in and on, and the cable to the first motor seated?"
        ) from error
    try:
        volts = bus.sync_read("Present_Voltage", normalize=False)
        temps = bus.sync_read("Present_Temperature", normalize=False)
    finally:
        bus.disconnect(disable_torque=False)
    hottest = max(temps, key=temps.get)
    summary = f"6 motors, {min(volts.values()) / 10:.1f} V, hottest {hottest} at {temps[hottest]} C"
    if temps[hottest] >= HOT_C:
        raise CheckFailed(f"{summary}: let it cool before running")
    return summary


def watch_arm(port: str, interval: float = 1.0) -> None:
    """Ping each motor once a second and print who answers, until Ctrl-C.

    For finding a bad cable: the motors are chained, so everything past a break
    goes silent. ``garbled`` means bytes came back that were not a valid reply,
    which points at the data line itself (a pinched or shorted cable) or at low
    supply voltage. Wiggle or unplug cables and watch the line change.
    """
    import scservo_sdk as scs

    handler = scs.PortHandler(port)
    packets = scs.PacketHandler(0)
    if not handler.openPort():
        raise CheckFailed(f"cannot open {port}")
    handler.setBaudRate(1_000_000)
    try:
        while True:
            cells = []
            for motor_id, name in enumerate(JOINTS, start=1):
                _, result, _ = packets.ping(handler, motor_id)
                if result == scs.COMM_SUCCESS:
                    volts, result, _ = packets.read1ByteTxRx(handler, motor_id, 62)
                    state = f"ok {volts / 10:.1f}V" if result == scs.COMM_SUCCESS else "ok"
                else:
                    state = "silent" if result == scs.COMM_RX_TIMEOUT else "garbled"
                cells.append(f"{motor_id} {name}: {state}")
            print(" | ".join(cells), flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        handler.closePort()


def check_camera(index: int, frame_path: Path) -> str:
    import cv2
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    config = OpenCVCameraConfig(
        index_or_path=index, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS
    )
    camera = OpenCVCamera(config)
    try:
        camera.connect()
    except Exception as error:
        raise CheckFailed(
            f"{str(error).splitlines()[0]} (on macOS, also check that this terminal "
            "is allowed under Privacy & Security > Camera)"
        ) from error
    try:
        frames = 30
        start = time.monotonic()
        for _ in range(frames):
            frame = camera.read()
        fps = frames / (time.monotonic() - start)
    finally:
        camera.disconnect()
    if frame.max() == 0:
        raise CheckFailed("frames are black: lens cap, or camera access was denied")
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    summary = f"{frame.shape[1]}x{frame.shape[0]} at {fps:.0f} fps"
    if fps < 0.8 * CAMERA_FPS:
        raise CheckFailed(f"{summary}, wanted {CAMERA_FPS}")
    return f"{summary}; look at {frame_path} to confirm it is the top view"


def main(argv: list[str] | None = None) -> None:
    """Check that the button module, the arm and the top camera all answer."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--arm-port", default=ARM_PORT)
    parser.add_argument("--button-port", default=BUTTON_PORT)
    parser.add_argument("--top-camera", type=int, default=TOP_CAMERA, help="OpenCV index")
    parser.add_argument("--frame", type=Path, default=Path("outputs/rig_check_top.png"))
    parser.add_argument(
        "--watch-arm", action="store_true", help="only ping the motors, once a second"
    )
    args = parser.parse_args(argv)

    if args.watch_arm:
        try:
            watch_arm(args.arm_port)
        except CheckFailed as error:
            sys.exit(f"FAIL  arm: {error}")
        return

    def buttons() -> str:
        if not Path(args.button_port).exists():
            raise CheckFailed(f"{args.button_port} not found: unplugged, or in a different socket")
        with ButtonModule.open(args.button_port) as module:
            return check_buttons(module)

    checks = {
        "buttons": buttons,
        "arm": lambda: check_arm(args.arm_port),
        "top camera": lambda: check_camera(args.top_camera, args.frame),
    }
    failed = False
    for name, check in checks.items():
        try:
            print(f"ok    {name}: {check()}", flush=True)
        except CheckFailed as error:
            failed = True
            print(f"FAIL  {name}: {error}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
