from __future__ import annotations

from types import SimpleNamespace

import pytest

from dart import cli
from dart.runtime import server
from tests.test_cli_check import local_model as local_model
from tests.test_sampling_devices import DEVICES
from tests.test_server import post


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("interrupt", [False, True])
def test_demo_command_uses_local_presets_and_closes_the_engine(local_model, monkeypatch, device, interrupt):
    directory, built = local_model
    closed = []

    def serve(engine, host="127.0.0.1", port=8400):
        assert host == "127.0.0.1" and port == 0
        close = engine.close

        def record_close():
            closed.append(True)
            close()

        monkeypatch.setattr(engine, "close", record_close)
        response = post(server.make_handler(engine), "/v1/dart_demo", {"prompt_ids": [7, 8, 7], "max_tokens": 3})
        assert response["identical"] and response["baseline"]["tokens"] == response["dart"]["tokens"]
        if interrupt:
            raise KeyboardInterrupt

    monkeypatch.setattr(server, "serve", serve)
    device = "cuda" if device.startswith("cuda") else device
    assert cli.main(["demo", "--model", str(directory), "--device", device, "--port", "0"]) == (130 if interrupt else 0)
    assert closed == [True] and len(built) == 1
    config, _ = built[0]
    assert config["target"]["local_files_only"] is True
    assert config["target"]["attn_implementation"] == ("sdpa" if device == "cpu" else "eager")
    assert config["decode"]["max_new_tokens"] == 32


@pytest.mark.parametrize("port", [-1, 65536])
def test_invalid_demo_ports_fail_before_loading(local_model, port):
    directory, built = local_model
    with pytest.raises(SystemExit) as error:
        cli.main(["demo", "--model", str(directory), "--port", str(port)])
    assert error.value.code == 2 and not built


@pytest.mark.parametrize("interrupt", [False, True])
def test_server_closes_after_normal_stop_or_interrupt(monkeypatch, capsys, interrupt):
    class Listener:
        server_port = 12345
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def serve_forever(self):
            if interrupt:
                raise KeyboardInterrupt

    listener = Listener()

    def create(address, handler):
        assert address == ("127.0.0.1", 0)
        return listener

    monkeypatch.setattr(server, "HTTPServer", create)
    engine = SimpleNamespace(target=SimpleNamespace(tokenizer=None))
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            server.serve(engine, port=0)
    else:
        server.serve(engine, port=0)
    assert listener.closed
    assert "127.0.0.1:12345" in capsys.readouterr().out
