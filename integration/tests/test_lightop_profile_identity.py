"""Loaded-module identity tests, using temporary fixture binaries, never GPUs."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from integration.lightop import profile_identity as identity
from integration.lightop.silu_mul import LightopSiluMulAdapter, declared_binding


@pytest.fixture
def loaded(tmp_path, monkeypatch):
    package = tmp_path / "lightop"
    package.mkdir()
    init, native = package / "__init__.py", package / "op.so"
    init.write_bytes(b"CPU fixture init")
    native.write_bytes(b"CPU fixture native")
    binding = declared_binding()
    expected = {"op_so_path": "lightop/op.so", "init_sha256": identity.digest(init),
                "op_so_sha256": identity.digest(native)}
    binding["op_so_sha256"] = expected["op_so_sha256"]
    operator = Mock()
    adapter = object.__new__(LightopSiluMulAdapter)
    adapter.implementation, adapter.operator = binding, operator
    module = SimpleNamespace(__file__=str(init), op=SimpleNamespace(__file__=str(native), silu_and_mul_opt=operator))
    monkeypatch.setattr(identity, "check_installed", lambda: {"version": "CPU fixture"})
    monkeypatch.setattr(identity, "catalog", lambda: expected)
    monkeypatch.setattr(identity, "declared_binding", lambda: binding)
    monkeypatch.setattr(identity.importlib.metadata, "distribution", lambda _: SimpleNamespace(locate_file=lambda path: tmp_path / path))
    monkeypatch.setattr(identity.importlib, "import_module", lambda _: module)
    return adapter, module, native


def test_actual_imported_files_and_callable_are_observed(loaded):
    adapter, module, native = loaded
    observed = identity.loaded_implementation(adapter)
    assert observed["source_sha256"]["native"] == identity.digest(native)
    assert observed["paths"]["native"] == str(native.resolve())
    adapter.operator.assert_not_called()


@pytest.mark.parametrize("change", ["shadow_path", "wrong_callable", "changed_binary", "changed_adapter"])
def test_loaded_identity_mismatch_is_rejected(loaded, tmp_path, change):
    adapter, module, native = loaded
    if change == "shadow_path":
        module.op.__file__ = str(tmp_path / "shadow.so")
    elif change == "wrong_callable":
        adapter.operator = Mock()
    elif change == "changed_binary":
        native.write_bytes(b"different CPU fixture binary")
    else:
        adapter.implementation["version"] = "sha256:wrong"
    with pytest.raises(ValueError):
        identity.loaded_implementation(adapter)


def test_cpu_identity_is_rejected_before_loading_lightop(monkeypatch):
    load = Mock(side_effect=AssertionError("must not load on CPU"))
    monkeypatch.setattr(identity, "loaded_implementation", load)
    with pytest.raises(RuntimeError, match="explicit HIP GPU"):
        identity.observe_runtime(None, "cpu")
    load.assert_not_called()


def test_runtime_observation_keeps_unknown_architecture_and_provenance_uuid(loaded, monkeypatch):
    adapter, _, _ = loaded
    monkeypatch.setattr(identity.torch.version, "hip", "CPU fixture HIP version")
    monkeypatch.setattr(identity.torch.cuda, "get_device_properties", lambda _: SimpleNamespace(
        name="CPU fixture GPU", total_memory=123, uuid="CPU fixture UUID"))
    monkeypatch.setattr(identity, "driver_version", lambda: None)
    monkeypatch.setattr(identity.importlib.metadata, "version", lambda name: "CPU fixture " + name)
    actual = identity.observe_runtime(adapter, "cuda:0")
    assert actual["device"]["architecture"] is None and actual["runtime"]["driver_version"] is None
    assert actual["provenance"]["device_uuid"] == "CPU fixture UUID"
    assert actual["runtime"]["execution_device"] == "cuda:0"
    assert "uuid" not in actual["runtime"] and actual["runtime"]["flags"]["grad_enabled"] is False
    assert "source_sha256" in actual["loaded_source"]
