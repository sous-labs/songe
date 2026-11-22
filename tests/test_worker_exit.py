"""A learning child that ends with an error says why."""

from songe.schedule import exit_reason


def test_a_kill_names_the_signal(tmp_path):
    err = tmp_path / "worker.err"
    err.write_text("")
    assert exit_reason(-9, err) == "killed by SIGKILL (out of memory?)"
    assert exit_reason(-15, err) == "killed by SIGTERM"


def test_an_error_exit_shows_the_last_line_of_stderr(tmp_path):
    err = tmp_path / "worker.err"
    err.write_text("Traceback (most recent call last):\n  File \"x.py\", line 1\n"
                   "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.48 GiB\n")
    assert exit_reason(1, err) == "exit code 1 · torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.48 GiB"
    assert exit_reason(1, tmp_path / "missing.err") == "exit code 1"
