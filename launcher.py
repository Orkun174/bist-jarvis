# launcher.py
"""Windows launcher for BIST Jarvis (jarvis_app.py) inside a native pywebview window."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import logging
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import webview


# Resolve adjacent application files from the installed executable directory,
# not PyInstaller's temporary extraction directory or the working directory.
BASE_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)
APP_PATH = BASE_DIR / "jarvis_app.py"
WINDOW_TITLE = "BIST Jarvis"
MUTEX_NAME = r"Local\BIST_Jarvis_Launcher"

STREAMLIT_PORT = 8765
STREAMLIT_URL = f"http://localhost:{STREAMLIT_PORT}"
STREAMLIT_HEALTH_URL = f"http://127.0.0.1:{STREAMLIT_PORT}/_stcore/health"

LOG_DIR = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    / "BIST_AI_Radar"
)

# Local service requests must bypass any configured HTTP proxy.
LOCAL_HTTP = urllib.request.build_opener(
    urllib.request.ProxyHandler({})
)


def port_is_listening(port: int) -> bool:
    """Check both loopback address families without creating a process."""
    for family, address in (
        (socket.AF_INET, ("127.0.0.1", port)),
        (socket.AF_INET6, ("::1", port, 0, 0)),
    ):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as connection:
                connection.settimeout(0.5)
                if connection.connect_ex(address) == 0:
                    return True
        except OSError:
            # IPv6 may be disabled; continue checking available interfaces.
            continue
    return False


def streamlit_ready() -> bool:
    """Confirm the Streamlit health endpoint is responding successfully."""
    try:
        with LOCAL_HTTP.open(
            STREAMLIT_HEALTH_URL, timeout=3
        ) as response:
            return (
                response.status == 200
                and response.read(128).decode("utf-8").strip().lower() == "ok"
            )
    except (OSError, ValueError, urllib.error.URLError):
        return False


def wait_until_ready(
    check,
    timeout: float,
    service_name: str,
    process: subprocess.Popen | None = None,
) -> None:
    """Wait for readiness, stopping early if an owned process exits."""
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"{service_name} exited with code {process.returncode}.\n"
                f"Check the logs in:\n{LOG_DIR}"
            )

        if check():
            return

        time.sleep(0.5)

    raise RuntimeError(
        f"{service_name} did not become ready within {timeout:.0f} seconds.\n"
        f"Check the logs in:\n{LOG_DIR}\n\n"
        "Its port may be occupied by another application."
    )


def start_hidden(
    arguments: list[str],
    log_name: str,
    environment: dict | None = None,
) -> subprocess.Popen:
    """Start a service without a console window and preserve its output."""
    with (LOG_DIR / log_name).open("ab", buffering=0) as output:
        process = subprocess.Popen(
            arguments,
            cwd=str(BASE_DIR),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            shell=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
            close_fds=True,
        )

    logging.info("Started PID %s: %s", process.pid, arguments)
    return process


def stop_owned_streamlit(process: subprocess.Popen | None) -> None:
    """
    Terminate only the Streamlit process created by this launcher.

    The python -m streamlit command runs the server directly, without an
    intermediary shell. Existing services discovered by port scanning are
    never adopted or terminated.
    """
    if process is None or process.poll() is not None:
        return

    logging.info("Stopping owned Streamlit process PID %s", process.pid)
    try:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logging.warning(
                "Streamlit PID %s did not exit; forcing termination.",
                process.pid,
            )
            process.kill()
            process.wait(timeout=5)
    except ProcessLookupError:
        # The process exited between polling and termination.
        return
    except OSError:
        if process.poll() is None:
            logging.exception(
                "Could not terminate Streamlit PID %s", process.pid
            )
    except subprocess.TimeoutExpired:
        logging.error(
            "Streamlit PID %s did not confirm termination.", process.pid
        )


def find_python() -> Path:
    """Find external Python without recursively invoking the frozen launcher."""
    explicit = os.environ.get("BIST_RADAR_PYTHON")

    if explicit:
        interpreter = Path(
            os.path.expandvars(explicit)
        ).expanduser()

        if not interpreter.is_file():
            raise FileNotFoundError(
                "BIST_RADAR_PYTHON does not point to a file:\n"
                f"{interpreter}"
            )

        candidates = [interpreter]
    else:
        candidates = [
            BASE_DIR / ".venv" / "Scripts" / "python.exe",
            BASE_DIR / "venv" / "Scripts" / "python.exe",
        ]

        if not getattr(sys, "frozen", False):
            candidates.append(Path(sys.executable))

        if discovered := shutil.which("python.exe"):
            candidates.append(Path(discovered))

    for candidate in candidates:
        if not candidate.is_file():
            continue

        resolved = candidate.resolve()

        # Avoid Windows Store execution aliases.
        if "windowsapps" in str(resolved).lower():
            continue

        if (
            getattr(sys, "frozen", False)
            and resolved == Path(sys.executable).resolve()
        ):
            continue

        return resolved

    raise FileNotFoundError(
        "No external Python interpreter was found.\n"
        "Provide .venv\\Scripts\\python.exe beside the launcher or set "
        "BIST_RADAR_PYTHON to an absolute interpreter path."
    )


def ensure_streamlit() -> subprocess.Popen | None:
    """
    Return the owned server process, or None when reusing an existing service.

    If startup fails after spawning, clean up immediately so no owned server
    is left behind without a desktop window.
    """
    process = None

    try:
        if not port_is_listening(STREAMLIT_PORT):
            if not APP_PATH.is_file():
                raise FileNotFoundError(
                    f"{APP_PATH.name} must be located beside the launcher:\n"
                    f"{APP_PATH}"
                )

            interpreter = find_python()

            # A service may have appeared during interpreter discovery.
            if not port_is_listening(STREAMLIT_PORT):
                environment = os.environ.copy()
                environment["PYTHONUNBUFFERED"] = "1"
                environment["PYTHONIOENCODING"] = "utf-8"

                process = start_hidden(
                    [
                        str(interpreter),
                        "-m", "streamlit",
                        "run", str(APP_PATH),
                        "--server.address=127.0.0.1",
                        f"--server.port={STREAMLIT_PORT}",
                        "--server.headless=true",
                        "--browser.gatherUsageStats=false",
                    ],
                    "streamlit.log",
                    environment,
                )

        wait_until_ready(
            streamlit_ready,
            timeout=90,
            service_name="Streamlit",
            process=process,
        )
        return process

    except BaseException:
        stop_owned_streamlit(process)
        raise


def show_error(message: str) -> None:
    """Display failures visibly when compiled as a windowless executable."""
    ctypes.windll.user32.MessageBoxW(
        None,
        message,
        "BIST Jarvis — Launcher Error",
        0x10,
    )


def main() -> None:
    """Prepare services, run the native window, and clean up on window closure."""
    if os.name != "nt":
        raise RuntimeError("This launcher supports Windows only.")

    # Hold the mutex for the entire desktop window lifetime. Repeated launches
    # cannot race to start duplicate servers or create conflicting ownership.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
    ]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel32.ReleaseMutex.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    mutex = kernel32.CreateMutexW(
        None, False, MUTEX_NAME
    )
    if not mutex:
        raise ctypes.WinError(ctypes.get_last_error())

    acquired = False
    streamlit_process = None

    try:
        status = kernel32.WaitForSingleObject(mutex, 0)

        if status == 0x00000102:  # Another launcher/window owns the mutex.
            return
        if status not in (0x00000000, 0x00000080):
            raise ctypes.WinError(ctypes.get_last_error())

        acquired = True

        streamlit_process = ensure_streamlit()

        if not streamlit_ready():
            raise RuntimeError("A background service became unavailable.")

        # Native desktop window without browser tabs, address bars, or toolbars.
        # Standard native window controls remain available for closing the app.
        webview.create_window(
            WINDOW_TITLE,
            STREAMLIT_URL,
            width=1280,
            height=800,
        )

        # Run on the main thread. This blocks until the desktop window closes.
        webview.start()

    finally:
        # Runs on normal window closure and on webview initialization failures.
        # Only the Streamlit server this launcher started is stopped.
        stop_owned_streamlit(streamlit_process)

        if acquired:
            kernel32.ReleaseMutex(mutex)
        kernel32.CloseHandle(mutex)


if __name__ == "__main__":
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(LOG_DIR / "launcher.log"),
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            encoding="utf-8",
        )
        main()
    except Exception as exc:
        logging.exception("Launcher failed")
        if os.name == "nt":
            show_error(f"{exc}\n\nDiagnostic logs:\n{LOG_DIR}")
        raise SystemExit(1)
