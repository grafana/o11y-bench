import threading

from o11y_bench import harbor


class _DummyProcess:
    def __init__(self) -> None:
        self.wait_calls = 0
        self.signals: list[int] = []

    def wait(self) -> int:
        self.wait_calls += 1
        return 0

    def send_signal(self, signum: int) -> None:
        self.signals.append(signum)


def test_run_skips_signal_handlers_when_disabled(monkeypatch) -> None:
    process = _DummyProcess()
    signal_calls: list[tuple[int, object]] = []
    cleanup_calls: list[bool] = []

    monkeypatch.setattr(harbor.subprocess, "Popen", lambda command: process)
    monkeypatch.setattr(
        harbor.signal, "signal", lambda signum, handler: signal_calls.append((signum, handler))
    )
    monkeypatch.setattr(harbor, "run_cleanup", lambda quiet=True: cleanup_calls.append(quiet))

    assert harbor.run(["uv", "run", "harbor", "run"], forward_signals=False) == 0
    assert process.wait_calls == 1
    assert signal_calls == []
    assert cleanup_calls == [True]


def test_run_rejects_signal_forwarding_off_main_thread(monkeypatch) -> None:
    popen_calls: list[list[str]] = []

    monkeypatch.setattr(harbor.subprocess, "Popen", lambda command: popen_calls.append(command))

    errors: list[Exception] = []

    def target() -> None:
        try:
            harbor.run(["uv", "run", "harbor", "run"])
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "signal forwarding requires the main thread" in str(errors[0])
    assert "forward_signals=False" in str(errors[0])
    assert popen_calls == []


def test_run_installs_signal_handlers_on_main_thread(monkeypatch) -> None:
    process = _DummyProcess()
    signal_calls: list[tuple[int, object]] = []
    cleanup_calls: list[bool] = []

    monkeypatch.setattr(harbor.subprocess, "Popen", lambda command: process)
    monkeypatch.setattr(harbor.signal, "getsignal", lambda signum: f"orig-{signum}")
    monkeypatch.setattr(
        harbor.signal, "signal", lambda signum, handler: signal_calls.append((signum, handler))
    )
    monkeypatch.setattr(harbor, "run_cleanup", lambda quiet=True: cleanup_calls.append(quiet))

    assert harbor.run(["uv", "run", "harbor", "run"]) == 0
    assert process.wait_calls == 1
    assert [call[0] for call in signal_calls] == [
        harbor.signal.SIGINT,
        harbor.signal.SIGTERM,
        harbor.signal.SIGINT,
        harbor.signal.SIGTERM,
    ]
    assert cleanup_calls == [True]


def test_run_preflight_cleanup_only_adds_flag(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)

        class _Result:
            returncode = 0
            stderr = ""
            stdout = ""

        return _Result()

    monkeypatch.setattr(harbor.subprocess, "run", fake_run)

    harbor.run_cleanup()

    assert calls == [["bash", str(harbor.PREFLIGHT_SCRIPT), "--cleanup-only"]]
