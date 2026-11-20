"""Trusted child process for executing a single Wasmtime store."""

from __future__ import annotations

import contextlib
import importlib.metadata
import sys
import tempfile
import threading
from pathlib import Path

import wasmtime

from desman.core.atomic import replace_file
from desman.core.sandbox import MOUNT_POINT, SandboxUnavailable, _WasiResult, _WasiSpec


def _engine() -> wasmtime.Engine:
    config = wasmtime.Config()
    config.epoch_interruption = True
    return wasmtime.Engine(config)


def _run(spec: _WasiSpec) -> _WasiResult:
    with _engine() as engine, _wasi_module(engine, spec.root) as module:
        spec.ready.touch()
        return _start(engine, module, spec)


def warm_cache(root: Path) -> None:
    try:
        with _engine() as engine, _wasi_module(engine, root, require_cache=True):
            pass
    except wasmtime.WasmtimeError as exc:
        raise SandboxUnavailable(f"Cannot prepare the WebAssembly sandbox: {exc}") from exc


def _wasi_module(
    engine: wasmtime.Engine, root: Path, *, require_cache: bool = False
) -> wasmtime.Module:
    # Wasmtime deserialization trusts native code. This cache is written only by DESMAN
    # in its runtime directory, which is outside the guest's preopened workspace.
    cache = root / f"wasmtime-{importlib.metadata.version('wasmtime')}" / "python.cwasm"
    if cache.is_file():
        with contextlib.suppress(OSError, wasmtime.WasmtimeError):
            return wasmtime.Module.deserialize_file(engine, str(cache))
    module = wasmtime.Module.from_file(engine, root / "py/python.wasm")
    try:
        cache.parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache.parent) as scratch:
            temporary = Path(scratch) / cache.name
            temporary.write_bytes(module.serialize())
            replace_file(temporary, cache)
    except OSError:
        if require_cache:
            module.close()
            raise
    return module


def _start(engine: wasmtime.Engine, module: wasmtime.Module, spec: _WasiSpec) -> _WasiResult:
    code: int | None = 0
    trap = ""
    with wasmtime.Linker(engine) as linker, wasmtime.Store(engine) as store:
        linker.define_wasi()
        with wasmtime.WasiConfig() as wasi:
            wasi.argv = spec.argv
            wasi.env = [
                ("PYTHONPATH", f"/site:{MOUNT_POINT}"),
                ("PYTHONDONTWRITEBYTECODE", "1"),
                ("HOME", MOUNT_POINT),
            ]
            wasi.preopen_dir(str(spec.root / "py/lib"), "/lib", False)
            wasi.preopen_dir(str(spec.root / "site"), "/site", False)
            wasi.preopen_dir(str(spec.workdir), MOUNT_POINT, True)
            wasi.stdin_file, wasi.stdout_file, wasi.stderr_file = (
                str(spec.stdin),
                str(spec.stdout),
                str(spec.stderr),
            )
            store.set_wasi(wasi)
        store.set_limits(memory_size=spec.memory_bytes)
        store.set_epoch_deadline(1)
        timer = threading.Timer(spec.timeout, engine.increment_epoch)
        timer.start()
        try:
            start = linker.instantiate(store, module).exports(store)["_start"]
            assert isinstance(start, wasmtime.Func)
            start(store)
        except wasmtime.ExitTrap as exc:
            code = exc.code
        except wasmtime.Trap as exc:
            if exc.trap_code == wasmtime.TrapCode.INTERRUPT:
                code = None
            else:
                code = 1
                trap = str(exc)
        finally:
            timer.cancel()
            timer.join()
    return _WasiResult(exit_code=code, trap=trap)


def main() -> None:
    spec = _WasiSpec.model_validate_json(Path(sys.argv[1]).read_bytes())
    try:
        result = _run(spec)
    except (OSError, wasmtime.WasmtimeError) as exc:
        result = _WasiResult(error=str(exc))
    spec.result.write_text(result.model_dump_json(), encoding="utf-8")


if __name__ == "__main__":
    main()
