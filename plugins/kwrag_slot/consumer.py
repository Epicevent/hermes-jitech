"""Hermes-owned consumer for an explicitly invoked KWRAG slot runtime.

This module owns correlation and consumption receipts only.  The caller owns
whether retrieval runs, the backend implementation, query construction,
stopping, prompt assembly, and whether returned evidence is shown to a model.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from plugins.kwrag_slot.manifest import canonical_json_bytes, load_component_manifest


_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_BINDING_FIELDS = {
    "schema_version",
    "enabled",
    "component_digest",
    "runtime_binding_digest",
    "expected_index_manifest",
    "expected_pipeline_fingerprint",
    "max_result_characters",
}


def _effective_user_id() -> int:
    getter = getattr(os, "geteuid", None)
    if getter is None:
        raise OSError("POSIX effective user identity is unavailable")
    return int(getter())


class HermesSlotRetrievalError(ValueError):
    """The explicit Hermes/KWRAG boundary could not be verified."""


class SlotRuntimeProtocol(Protocol):
    def search_exchange(self, request: Any) -> Any: ...


class ConsumptionReceiptSink(Protocol):
    def write(self, receipt: Mapping[str, Any]) -> str: ...

    def write_once(self, identity: str, receipt: Mapping[str, Any]) -> str: ...


class FileConsumptionReceiptSink:
    """Persist canonical consumption receipts using the KWRAG POSIX writer."""

    def __init__(self, path: Path):
        if not path.is_absolute():
            raise HermesSlotRetrievalError("consumption receipt path must be absolute")
        try:
            from kwrag.operation import ReceiptWriter
        except ImportError as exc:
            raise HermesSlotRetrievalError("embedded KWRAG component is unavailable") from exc
        self._path = path
        self._writer = ReceiptWriter(path)
        self._write_once_lock = threading.Lock()
        self._trusted_parent_identity: tuple[int, int] | None = None
        self._trusted_receipt_identity: tuple[int, int] | None = None
        self._trusted_outcome_root_identity: tuple[int, int] | None = None

    def write(self, receipt: Mapping[str, Any]) -> str:
        raw = canonical_json_bytes(dict(receipt)) + b"\n"
        receipt_digest = "sha256:" + hashlib.sha256(raw[:-1]).hexdigest()
        if os.name == "posix":
            try:
                with self._write_once_lock:
                    parent_identity, receipt_identity = self._append_receipt_posix(raw)
                    if (
                        self._trusted_parent_identity is not None
                        and self._trusted_parent_identity != parent_identity
                    ):
                        raise OSError("consumption receipt parent identity changed")
                    if (
                        self._trusted_receipt_identity is not None
                        and self._trusted_receipt_identity != receipt_identity
                    ):
                        raise OSError("consumption receipt file identity changed")
                    self._trusted_parent_identity = parent_identity
                    self._trusted_receipt_identity = receipt_identity
            except OSError as exc:
                raise HermesSlotRetrievalError(
                    "consumption receipt parent is not an approved POSIX ledger boundary"
                ) from exc
        else:
            parent_identity = None
            result = self._writer.write(dict(receipt))
            if result.status != "written" or result.digest != receipt_digest:
                raise HermesSlotRetrievalError("consumption receipt was not written")
        return receipt_digest

    def write_once(self, identity: str, receipt: Mapping[str, Any]) -> str:
        identity_digest = _digest(identity, "receipt identity")
        raw = canonical_json_bytes(dict(receipt)) + b"\n"
        receipt_digest = "sha256:" + hashlib.sha256(raw[:-1]).hexdigest()
        outcome_root = self._path.parent / f"{self._path.name}.outcomes"
        outcome_name = f"{identity_digest.removeprefix('sha256:')}.json"
        if os.name != "posix":
            raise HermesSlotRetrievalError(
                "provider attempt outcome persistence requires the POSIX slot runtime"
            )
        try:
            with self._write_once_lock:
                outcome_root_identity = self._write_once_posix(
                    outcome_root,
                    outcome_name,
                    raw,
                )
                if (
                    self._trusted_outcome_root_identity is not None
                    and self._trusted_outcome_root_identity != outcome_root_identity
                ):
                    raise OSError("provider attempt outcome directory identity changed")
                self._trusted_outcome_root_identity = outcome_root_identity
        except HermesSlotRetrievalError:
            raise
        except OSError as exc:
            raise HermesSlotRetrievalError(
                "provider attempt outcome could not be persisted safely"
            ) from exc
        return receipt_digest

    @staticmethod
    def _write_all(descriptor: int, raw: bytes) -> None:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("provider attempt outcome made no forward progress")
            view = view[written:]

    @staticmethod
    def _read_all(descriptor: int) -> bytes:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)

    @staticmethod
    def _verify_outcome_file(descriptor: int, *, require_owner_mode: bool) -> None:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise OSError("provider attempt outcome is not a single-link regular file")
        if require_owner_mode and (
            info.st_uid != _effective_user_id()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise OSError("provider attempt outcome owner or mode is invalid")

    def _open_existing_posix_receipt_parent(self) -> tuple[list[int], int]:
        if not self._path.parent.is_absolute():
            raise OSError("POSIX receipt parent path must be absolute")
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
            raise OSError("POSIX receipt parent requires no-follow opens")
        directory_flags = (
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        )
        opened: list[int] = []
        try:
            current = os.open("/", directory_flags)
            opened.append(current)
            for part in self._path.parent.parts[1:]:
                following = os.open(part, directory_flags, dir_fd=current)
                info = os.fstat(following)
                if not stat.S_ISDIR(info.st_mode) or info.st_nlink < 1:
                    os.close(following)
                    raise OSError("consumption receipt parent identity is invalid")
                opened.append(following)
                current = following
            parent_info = os.fstat(current)
            if (
                parent_info.st_uid != _effective_user_id()
                or stat.S_IMODE(parent_info.st_mode) != 0o700
            ):
                raise OSError("consumption receipt parent owner or mode is invalid")
            return opened, current
        except BaseException:
            for descriptor in reversed(opened):
                os.close(descriptor)
            raise

    def _assert_receipt_parent_path_identity(
        self,
        expected: tuple[int, int],
    ) -> None:
        opened, current = self._open_existing_posix_receipt_parent()
        try:
            info = os.fstat(current)
            if (info.st_dev, info.st_ino) != expected:
                raise OSError("consumption receipt parent was substituted")
        finally:
            for descriptor in reversed(opened):
                os.close(descriptor)

    def _assert_receipt_file_path_identity(
        self,
        parent: int,
        expected: tuple[int, int],
    ) -> None:
        info = os.stat(
            self._path.name,
            dir_fd=parent,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or (info.st_dev, info.st_ino) != expected
        ):
            raise OSError("consumption receipt file was substituted")

    @staticmethod
    def _open_public_file_no_symlinks(path: Path) -> int:
        """Open one complete Linux pathname without following any symlink."""

        if sys.platform != "linux":
            raise OSError(
                errno.ENOTSUP,
                "openat2 public-path verification is unavailable",
            )
        machine = os.uname().machine.lower()
        if machine not in {"x86_64", "amd64", "aarch64", "arm64"}:
            raise OSError(
                errno.ENOTSUP,
                f"openat2 syscall number is not bound for {machine}",
            )

        class _OpenHow(ctypes.Structure):
            _fields_ = [
                ("flags", ctypes.c_uint64),
                ("mode", ctypes.c_uint64),
                ("resolve", ctypes.c_uint64),
            ]

        # openat2 is syscall 437 on the customer amd64/arm64 Linux targets.
        syscall_openat2 = 437
        at_fdcwd = -100
        resolve_no_symlinks = 0x04
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        how = _OpenHow(flags=flags, mode=0, resolve=resolve_no_symlinks)
        raw_path = os.fsencode(os.path.abspath(os.fspath(path)))
        libc = ctypes.CDLL(None, use_errno=True)
        descriptor = libc.syscall(
            ctypes.c_long(syscall_openat2),
            ctypes.c_int(at_fdcwd),
            ctypes.c_char_p(raw_path),
            ctypes.byref(how),
            ctypes.c_size_t(ctypes.sizeof(how)),
        )
        if descriptor < 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number), raw_path)
        return int(descriptor)

    def _assert_public_file_path_identity(
        self,
        path: Path,
        expected_file: tuple[int, int],
        *,
        label: str,
    ) -> None:
        """Linearize a full public path and file inode in one openat2 lookup."""

        descriptor = self._open_public_file_no_symlinks(path)
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_uid != _effective_user_id()
                or stat.S_IMODE(info.st_mode) != 0o600
                or (info.st_dev, info.st_ino) != expected_file
            ):
                raise OSError(f"{label} public pathname was substituted")
        finally:
            os.close(descriptor)

    @staticmethod
    def _assert_named_directory_identity(
        parent: int,
        name: str,
        expected: tuple[int, int],
    ) -> None:
        flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(name, flags, dir_fd=parent)
        try:
            info = os.fstat(descriptor)
            if (info.st_dev, info.st_ino) != expected:
                raise OSError("provider attempt outcome directory was substituted")
        finally:
            os.close(descriptor)

    @staticmethod
    def _assert_named_outcome_file_identity(
        parent: int,
        name: str,
        expected: tuple[int, int],
    ) -> None:
        info = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != _effective_user_id()
            or stat.S_IMODE(info.st_mode) != 0o600
            or (info.st_dev, info.st_ino) != expected
        ):
            raise OSError("provider attempt outcome file was substituted")

    def _append_receipt_posix(
        self,
        raw: bytes,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        opened, parent = self._open_existing_posix_receipt_parent()
        try:
            parent_info = os.fstat(parent)
            parent_identity = (parent_info.st_dev, parent_info.st_ino)
            if (
                self._trusted_parent_identity is not None
                and self._trusted_parent_identity != parent_identity
            ):
                raise OSError("consumption receipt parent identity changed")
            self._assert_receipt_parent_path_identity(parent_identity)

            existed = True
            try:
                existing_info = os.stat(
                    self._path.name,
                    dir_fd=parent,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if self._trusted_receipt_identity is not None:
                    raise OSError("consumption receipt file identity changed")
                existed = False
            else:
                if (
                    not stat.S_ISREG(existing_info.st_mode)
                    or existing_info.st_nlink != 1
                    or existing_info.st_uid != _effective_user_id()
                    or stat.S_IMODE(existing_info.st_mode) != 0o600
                ):
                    raise OSError("consumption receipt file identity is invalid")
                if self._trusted_receipt_identity is not None and (
                    existing_info.st_dev,
                    existing_info.st_ino,
                ) != self._trusted_receipt_identity:
                    raise OSError("consumption receipt file identity changed")

            open_flags = os.O_RDWR | os.O_APPEND | os.O_CLOEXEC | os.O_NOFOLLOW
            if existed:
                descriptor = os.open(
                    self._path.name,
                    open_flags,
                    dir_fd=parent,
                )
            else:
                try:
                    descriptor = os.open(
                        self._path.name,
                        open_flags | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=parent,
                    )
                except FileExistsError:
                    existed = True
                    descriptor = os.open(
                        self._path.name,
                        open_flags,
                        dir_fd=parent,
                    )
            try:
                if not existed:
                    os.fchmod(descriptor, 0o600)
                self._verify_outcome_file(descriptor, require_owner_mode=True)
                receipt_info = os.fstat(descriptor)
                receipt_identity = (receipt_info.st_dev, receipt_info.st_ino)
                if (
                    self._trusted_receipt_identity is not None
                    and self._trusted_receipt_identity != receipt_identity
                ):
                    raise OSError("consumption receipt file identity changed")
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX)
                original_size = os.fstat(descriptor).st_size
                try:
                    # Keep the approved directory fd open, and prove the
                    # pathname still resolves to that inode immediately
                    # before publishing any receipt bytes.
                    self._assert_receipt_parent_path_identity(parent_identity)
                    self._assert_receipt_file_path_identity(parent, receipt_identity)
                    try:
                        self._write_all(descriptor, raw)
                        os.fsync(descriptor)
                        # A same-UID peer can rename the directory while the
                        # held fd remains valid.  Success therefore requires a
                        # post-publication identity check as well.  If the
                        # pathname moved during the write, remove exactly this
                        # append before returning fail-closed.
                        self._assert_receipt_parent_path_identity(parent_identity)
                        if not existed:
                            os.fsync(parent)
                        # Final observable publication identity operation.
                        self._assert_receipt_file_path_identity(
                            parent,
                            receipt_identity,
                        )
                        self._assert_public_file_path_identity(
                            self._path,
                            receipt_identity,
                            label="consumption receipt file",
                        )
                    except BaseException:
                        os.ftruncate(descriptor, original_size)
                        os.fsync(descriptor)
                        if not existed:
                            try:
                                self._assert_receipt_file_path_identity(
                                    parent,
                                    receipt_identity,
                                )
                            except (FileNotFoundError, OSError):
                                pass
                            else:
                                os.unlink(self._path.name, dir_fd=parent)
                                os.fsync(parent)
                        raise
                finally:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
            return parent_identity, receipt_identity
        finally:
            for descriptor in reversed(opened):
                os.close(descriptor)

    def _write_once_posix(
        self,
        outcome_root: Path,
        outcome_name: str,
        raw: bytes,
    ) -> tuple[int, int]:
        directory_flags = (
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        )
        if self._trusted_parent_identity is None:
            raise OSError("provider attempt outcome has no trusted receipt parent")
        opened, current = self._open_existing_posix_receipt_parent()
        try:
            current_info = os.fstat(current)
            if (current_info.st_dev, current_info.st_ino) != (
                self._trusted_parent_identity
            ):
                raise OSError("consumption receipt parent was substituted")

            outcome_root_created = False
            try:
                os.mkdir(outcome_root.name, 0o700, dir_fd=current)
                outcome_root_created = True
            except FileExistsError:
                pass
            outcome_directory = os.open(
                outcome_root.name,
                directory_flags,
                dir_fd=current,
            )
            opened.append(outcome_directory)
            root_info = os.fstat(outcome_directory)
            outcome_root_identity = (root_info.st_dev, root_info.st_ino)
            if (
                not stat.S_ISDIR(root_info.st_mode)
                or root_info.st_nlink < 1
                or root_info.st_uid != _effective_user_id()
                or stat.S_IMODE(root_info.st_mode) != 0o700
            ):
                raise OSError("provider attempt outcome directory identity is invalid")
            if (
                self._trusted_outcome_root_identity is not None
                and self._trusted_outcome_root_identity != outcome_root_identity
            ):
                raise OSError("provider attempt outcome directory identity changed")
            if outcome_root_created:
                os.fsync(current)
            self._assert_named_directory_identity(
                current,
                outcome_root.name,
                outcome_root_identity,
            )

            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_CLOEXEC
                | os.O_NOFOLLOW
            )
            try:
                descriptor = os.open(
                    outcome_name,
                    flags,
                    0o600,
                    dir_fd=outcome_directory,
                )
            except FileExistsError:
                existing_info = os.stat(
                    outcome_name,
                    dir_fd=outcome_directory,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(existing_info.st_mode)
                    or existing_info.st_nlink != 1
                    or existing_info.st_uid != _effective_user_id()
                    or stat.S_IMODE(existing_info.st_mode) != 0o600
                ):
                    raise OSError("existing provider attempt outcome identity is invalid")
                descriptor = os.open(
                    outcome_name,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=outcome_directory,
                )
                try:
                    self._verify_outcome_file(descriptor, require_owner_mode=True)
                    descriptor_info = os.fstat(descriptor)
                    outcome_file_identity = (
                        descriptor_info.st_dev,
                        descriptor_info.st_ino,
                    )
                    if outcome_file_identity != (
                        existing_info.st_dev,
                        existing_info.st_ino,
                    ):
                        raise OSError(
                            "existing provider attempt outcome was substituted"
                        )
                    self._assert_named_outcome_file_identity(
                        outcome_directory,
                        outcome_name,
                        outcome_file_identity,
                    )
                    existing = self._read_all(descriptor)
                    if existing != raw:
                        raise HermesSlotRetrievalError(
                            "provider attempt outcome identity collision"
                        )
                    self._assert_receipt_parent_path_identity(
                        self._trusted_parent_identity
                    )
                    self._assert_named_directory_identity(
                        current,
                        outcome_root.name,
                        outcome_root_identity,
                    )
                    # Final observable replay identity operation.
                    self._assert_named_outcome_file_identity(
                        outcome_directory,
                        outcome_name,
                        outcome_file_identity,
                    )
                    self._assert_public_file_path_identity(
                        outcome_root / outcome_name,
                        outcome_file_identity,
                        label="provider attempt outcome file",
                    )
                finally:
                    os.close(descriptor)
                return outcome_root_identity
            try:
                self._verify_outcome_file(descriptor, require_owner_mode=True)
                descriptor_info = os.fstat(descriptor)
                outcome_file_identity = (
                    descriptor_info.st_dev,
                    descriptor_info.st_ino,
                )
                try:
                    self._assert_named_outcome_file_identity(
                        outcome_directory,
                        outcome_name,
                        outcome_file_identity,
                    )
                    self._write_all(descriptor, raw)
                    os.fsync(descriptor)
                    self._assert_receipt_parent_path_identity(
                        self._trusted_parent_identity
                    )
                    self._assert_named_directory_identity(
                        current,
                        outcome_root.name,
                        outcome_root_identity,
                    )
                    os.fsync(outcome_directory)
                    # Final observable publication identity operation.
                    self._assert_named_outcome_file_identity(
                        outcome_directory,
                        outcome_name,
                        outcome_file_identity,
                    )
                    self._assert_public_file_path_identity(
                        outcome_root / outcome_name,
                        outcome_file_identity,
                        label="provider attempt outcome file",
                    )
                except BaseException:
                    os.ftruncate(descriptor, 0)
                    os.fsync(descriptor)
                    try:
                        file_info = os.stat(
                            outcome_name,
                            dir_fd=outcome_directory,
                            follow_symlinks=False,
                        )
                        if (file_info.st_dev, file_info.st_ino) == (
                            outcome_file_identity[0],
                            outcome_file_identity[1],
                        ):
                            os.unlink(outcome_name, dir_fd=outcome_directory)
                            os.fsync(outcome_directory)
                    except FileNotFoundError:
                        pass
                    raise
            finally:
                os.close(descriptor)
            return outcome_root_identity
        finally:
            for descriptor in reversed(opened):
                os.close(descriptor)


def _digest(value: object, field: str) -> str:
    text = str(value or "")
    if not _SHA256.fullmatch(text):
        raise HermesSlotRetrievalError(f"{field} is not a canonical SHA-256 digest")
    return text


@dataclass(frozen=True)
class HermesSlotRetrievalBinding:
    enabled: bool
    component_digest: str
    runtime_binding_digest: str | None
    expected_index_manifest: str | None
    expected_pipeline_fingerprint: str | None
    max_result_characters: int

    @classmethod
    def from_mapping(cls, raw: Any) -> "HermesSlotRetrievalBinding":
        if not isinstance(raw, Mapping) or set(raw) != _BINDING_FIELDS:
            raise HermesSlotRetrievalError("Hermes slot retrieval binding fields are invalid")
        if raw.get("schema_version") != "hermes-kwrag-slot-binding-v1":
            raise HermesSlotRetrievalError("Hermes slot retrieval binding schema is invalid")
        enabled = raw.get("enabled")
        if not isinstance(enabled, bool):
            raise HermesSlotRetrievalError("Hermes slot retrieval enabled flag is invalid")
        manifest = load_component_manifest()
        component_digest = _digest(raw.get("component_digest"), "component digest")
        if component_digest != manifest["component_wheel"]["sha256"]:
            raise HermesSlotRetrievalError("Hermes binding does not name the embedded component")
        max_chars = raw.get("max_result_characters")
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 0 <= max_chars <= 80_000:
            raise HermesSlotRetrievalError("Hermes result character budget is invalid")
        runtime_digest = raw.get("runtime_binding_digest")
        manifest_digest = raw.get("expected_index_manifest")
        pipeline_digest = raw.get("expected_pipeline_fingerprint")
        if enabled:
            return cls(
                enabled=True,
                component_digest=component_digest,
                runtime_binding_digest=_digest(runtime_digest, "runtime binding digest"),
                expected_index_manifest=_digest(manifest_digest, "index manifest digest"),
                expected_pipeline_fingerprint=_digest(pipeline_digest, "pipeline fingerprint"),
                max_result_characters=max_chars,
            )
        if any(value is not None for value in (runtime_digest, manifest_digest, pipeline_digest)):
            raise HermesSlotRetrievalError("disabled Hermes retrieval must not retain a runtime binding")
        if max_chars != 0:
            raise HermesSlotRetrievalError("disabled Hermes retrieval must have a zero result budget")
        return cls(False, component_digest, None, None, None, 0)


@dataclass
class HermesSlotRetrievalResult:
    results: tuple[dict[str, Any], ...]
    result_receipt: dict[str, Any]
    result_receipt_digest: str
    result_receipt_status: str
    _canonical_results_bytes: bytes = field(repr=False, compare=False)
    _canonical_result_receipt_bytes: bytes = field(repr=False, compare=False)
    consumption_receipt: dict[str, Any] | None = None
    consumption_receipt_digest: str | None = None
    consumption_receipt_status: str = "pending"
    provider_attempt_outcome_receipt: dict[str, Any] | None = None
    provider_attempt_outcome_receipt_digest: str | None = None
    provider_attempt_outcome_status: str = "pending"
    _receipt_sink: ConsumptionReceiptSink | None = field(repr=False, compare=False, default=None)

    def _verified_evidence(self) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
        """Reject caller mutation and return private canonical evidence copies."""

        try:
            current_results_bytes = canonical_json_bytes(list(self.results))
            current_receipt_bytes = canonical_json_bytes(self.result_receipt)
        except (TypeError, ValueError) as exc:
            raise HermesSlotRetrievalError("verified retrieval evidence is not canonical") from exc
        if current_results_bytes != self._canonical_results_bytes:
            raise HermesSlotRetrievalError("verified retrieval results were mutated")
        if current_receipt_bytes != self._canonical_result_receipt_bytes:
            raise HermesSlotRetrievalError("verified result receipt was mutated")
        receipt_digest = "sha256:" + hashlib.sha256(
            self._canonical_result_receipt_bytes
        ).hexdigest()
        if receipt_digest != self.result_receipt_digest:
            raise HermesSlotRetrievalError("verified result receipt digest is not bound")

        canonical_results = json.loads(self._canonical_results_bytes.decode("utf-8"))
        canonical_receipt = json.loads(
            self._canonical_result_receipt_bytes.decode("utf-8")
        )
        results_digest = "sha256:" + hashlib.sha256(
            self._canonical_results_bytes
        ).hexdigest()
        if canonical_receipt.get("result_digest") != results_digest:
            raise HermesSlotRetrievalError("verified result payload digest is not bound")
        if canonical_receipt.get("result_count") != len(canonical_results):
            raise HermesSlotRetrievalError("verified result count is not bound")
        try:
            result_characters = len(self._canonical_results_bytes.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise HermesSlotRetrievalError("verified result payload is not UTF-8") from exc
        if canonical_receipt.get("result_characters") != result_characters:
            raise HermesSlotRetrievalError("verified result character budget is not bound")
        return tuple(canonical_results), canonical_receipt

    def record_prompt_consumption(
        self,
        *,
        session_binding_digest: str,
        prompt_context_digest: str,
        provider_attempt_binding: Mapping[str, Any],
    ) -> str:
        verified_results, verified_receipt = self._verified_evidence()
        if self.consumption_receipt_status != "pending" or self.consumption_receipt is not None:
            raise HermesSlotRetrievalError("retrieval evidence was already consumed")
        if verified_receipt.get("result_status") != "hits" or not verified_results:
            raise HermesSlotRetrievalError("retrieval evidence has no verified hits to consume")
        if not isinstance(provider_attempt_binding, Mapping):
            raise HermesSlotRetrievalError("provider attempt binding is unavailable")
        required_attempt_fields = {
            "schema",
            "providerAttemptId",
            "providerCallId",
            "configuredProvider",
            "configuredModel",
            "provider",
            "apiMode",
            "model",
            "sdkMethod",
            "leafAdapter",
            "endpointIdentity",
            "fallbackIndex",
            "configuredRouteChainDigest",
            "finalRequestKwargsDigest",
            "providerAttemptBindingDigest",
        }
        if set(provider_attempt_binding) != required_attempt_fields:
            raise HermesSlotRetrievalError("provider attempt binding fields are invalid")
        binding = dict(provider_attempt_binding)
        binding_digest = _digest(
            binding.pop("providerAttemptBindingDigest", None),
            "provider attempt binding digest",
        )
        computed_binding_digest = "sha256:" + hashlib.sha256(
            canonical_json_bytes(binding)
        ).hexdigest()
        if binding_digest != computed_binding_digest:
            raise HermesSlotRetrievalError("provider attempt binding digest is not bound")
        if binding.get("schema") != "jitech-provider-sdk-request-attempt-binding/v1":
            raise HermesSlotRetrievalError("provider attempt binding schema is invalid")
        if binding.get("providerAttemptId") != 1:
            raise HermesSlotRetrievalError("retrieval evidence requires provider attempt 1")
        provider_call_id = binding.get("providerCallId")
        if (
            not isinstance(provider_call_id, str)
            or not re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
                provider_call_id,
            )
        ):
            raise HermesSlotRetrievalError("provider call identity is invalid")
        for field_name in (
            "configuredProvider",
            "configuredModel",
            "provider",
            "apiMode",
            "model",
            "sdkMethod",
            "leafAdapter",
            "endpointIdentity",
        ):
            if not isinstance(binding.get(field_name), str) or not binding[field_name]:
                raise HermesSlotRetrievalError(
                    f"provider attempt {field_name} is invalid"
                )
        fallback_index = binding.get("fallbackIndex")
        if (
            isinstance(fallback_index, bool)
            or not isinstance(fallback_index, int)
            or fallback_index < 0
        ):
            raise HermesSlotRetrievalError(
                "provider attempt fallbackIndex is invalid"
            )
        request_kwargs_digest = _digest(
            binding.get("finalRequestKwargsDigest"),
            "final provider request kwargs digest",
        )
        route_chain_digest = _digest(
            binding.get("configuredRouteChainDigest"),
            "configured provider route chain digest",
        )
        receipt = {
            "schema_version": "hermes-kwrag-consumption-receipt-v1",
            "consumer_family": "hermes",
            "consumption_status": "evidence_dispatch_handoff_committed",
            "evidence_projection_status": "verified_hits",
            "dispatch_handoff_status": "evidence_dispatch_handoff_committed",
            "transport_outcome_status": "unknown",
            "provider_attestation_status": "unavailable",
            "billing_status": "unavailable",
            "component_digest": verified_receipt["component_digest"],
            "runtime_binding_digest": verified_receipt["runtime_binding_digest"],
            "index_manifest": verified_receipt["index_manifest"],
            "session_binding_digest": _digest(session_binding_digest, "session binding digest"),
            "prompt_context_digest": _digest(prompt_context_digest, "prompt context digest"),
            "provider_attempt_id": 1,
            "provider_call_id": provider_call_id,
            "provider_attempt_binding_digest": binding_digest,
            "provider_request_kwargs_digest": request_kwargs_digest,
            "configured_route_chain_digest": route_chain_digest,
            "configured_provider": binding["configuredProvider"],
            "configured_model": binding["configuredModel"],
            "provider": binding["provider"],
            "api_mode": binding["apiMode"],
            "model": binding["model"],
            "sdk_method": binding["sdkMethod"],
            "leaf_adapter": binding["leafAdapter"],
            "endpoint_identity": binding["endpointIdentity"],
            "fallback_index": fallback_index,
            "request_id": verified_receipt["request_id"],
            "operation_id": verified_receipt["operation_id"],
            "run_id": verified_receipt["run_id"],
            "attempt": verified_receipt["attempt"],
            "result_digest": verified_receipt["result_digest"],
            "operation_receipt_digest": verified_receipt["operation_receipt_digest"],
            "result_receipt_digest": self.result_receipt_digest,
        }
        receipt_digest = "sha256:" + hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
        if self._receipt_sink is None:
            raise HermesSlotRetrievalError("consumption receipt sink is unavailable")
        written_digest = self._receipt_sink.write(receipt)
        if written_digest != receipt_digest:
            raise HermesSlotRetrievalError("written consumption receipt digest is not bound")
        self.consumption_receipt = receipt
        self.consumption_receipt_digest = receipt_digest
        self.consumption_receipt_status = "written"
        return receipt_digest

    def record_provider_attempt_outcome(
        self,
        *,
        provider_attempt_binding_digest: str,
        transport_outcome_status: str,
        error_category: str | None = None,
    ) -> str:
        if self.consumption_receipt_status != "written" or self.consumption_receipt is None:
            raise HermesSlotRetrievalError("provider attempt outcome has no dispatch receipt")
        if self.provider_attempt_outcome_status != "pending":
            raise HermesSlotRetrievalError("provider attempt outcome was already recorded")
        binding_digest = _digest(
            provider_attempt_binding_digest,
            "provider attempt binding digest",
        )
        if binding_digest != self.consumption_receipt.get(
            "provider_attempt_binding_digest"
        ):
            raise HermesSlotRetrievalError("provider attempt outcome binding mismatch")
        if transport_outcome_status not in {
            "response_observed",
            "sdk_exception",
            "interrupted",
            "unknown",
        }:
            raise HermesSlotRetrievalError("provider attempt outcome status is invalid")
        if error_category is not None and (
            not isinstance(error_category, str) or not error_category
        ):
            raise HermesSlotRetrievalError("provider attempt error category is invalid")
        if (
            transport_outcome_status == "response_observed"
            and error_category is not None
        ):
            raise HermesSlotRetrievalError(
                "observed provider response must not carry an error category"
            )
        receipt = {
            "schema_version": "hermes-kwrag-provider-attempt-outcome-receipt-v1",
            "consumer_family": "hermes",
            "provider_attempt_id": 1,
            "provider_call_id": self.consumption_receipt["provider_call_id"],
            "provider_attempt_binding_digest": binding_digest,
            "transport_outcome_status": transport_outcome_status,
            "error_category": error_category,
        }
        receipt_digest = "sha256:" + hashlib.sha256(
            canonical_json_bytes(receipt)
        ).hexdigest()
        if self._receipt_sink is None:
            raise HermesSlotRetrievalError("provider attempt outcome sink is unavailable")
        write_once = getattr(self._receipt_sink, "write_once", None)
        if not callable(write_once):
            raise HermesSlotRetrievalError(
                "provider attempt outcome sink is not insert-once capable"
            )
        written_digest = write_once(binding_digest, receipt)
        if written_digest != receipt_digest:
            raise HermesSlotRetrievalError(
                "written provider attempt outcome receipt digest is not bound"
            )
        self.provider_attempt_outcome_receipt = receipt
        self.provider_attempt_outcome_receipt_digest = receipt_digest
        self.provider_attempt_outcome_status = "written"
        return receipt_digest

    def content_free_attestation(self) -> dict[str, Any]:
        """Project exact result/consumption lineage for an enabled canary."""

        _verified_results, verified_receipt = self._verified_evidence()
        result_status = verified_receipt.get("result_status")
        if result_status == "zero_hits":
            projection_status = "verified_zero_hits"
            dispatch_handoff_status = "not_committed"
            transport_outcome_status = "not_attempted"
        elif self.consumption_receipt_status == "written":
            projection_status = "verified_hits"
            dispatch_handoff_status = "evidence_dispatch_handoff_committed"
            transport_outcome_status = (
                self.provider_attempt_outcome_receipt.get("transport_outcome_status")
                if self.provider_attempt_outcome_receipt is not None
                else "unknown"
            )
        else:
            projection_status = "verified_hits"
            dispatch_handoff_status = "not_committed"
            transport_outcome_status = "not_attempted"
        return {
            "schema": "jitech-hermes-kwrag-consumption-attestation/v1",
            "componentDigest": verified_receipt["component_digest"],
            "runtimeBindingDigest": verified_receipt["runtime_binding_digest"],
            "indexManifestDigest": verified_receipt["index_manifest"],
            "resultStatus": result_status,
            "operationReceiptDigest": verified_receipt["operation_receipt_digest"],
            "resultReceiptDigest": self.result_receipt_digest,
            "consumptionReceiptDigest": self.consumption_receipt_digest,
            "providerAttemptId": (
                self.consumption_receipt.get("provider_attempt_id")
                if self.consumption_receipt is not None
                else None
            ),
            "providerCallId": (
                self.consumption_receipt.get("provider_call_id")
                if self.consumption_receipt is not None
                else None
            ),
            "providerAttemptBindingDigest": (
                self.consumption_receipt.get("provider_attempt_binding_digest")
                if self.consumption_receipt is not None
                else None
            ),
            "providerAttemptOutcomeReceiptDigest": (
                self.provider_attempt_outcome_receipt_digest
            ),
            "evidenceProjectionStatus": projection_status,
            "dispatchHandoffStatus": dispatch_handoff_status,
            "transportOutcomeStatus": transport_outcome_status,
            "providerAttestationStatus": "unavailable",
            "billingStatus": "unavailable",
        }


class HermesSlotRetrievalConsumer:
    """Execute only a caller-authorized search and bind the consumed evidence."""

    def __init__(
        self,
        binding: HermesSlotRetrievalBinding,
        runtime: SlotRuntimeProtocol | None,
        receipt_sink: ConsumptionReceiptSink | None,
    ):
        if binding.enabled != (runtime is not None) or binding.enabled != (receipt_sink is not None):
            raise HermesSlotRetrievalError("runtime and receipt sink do not match the enabled binding")
        self._binding = binding
        self._runtime = runtime
        self._receipt_sink = receipt_sink

    def search(self, request: Mapping[str, Any]) -> HermesSlotRetrievalResult:
        if not self._binding.enabled or self._runtime is None or self._receipt_sink is None:
            raise HermesSlotRetrievalError("Hermes slot retrieval is disabled")
        try:
            from kwrag.slot_consumer import verify_slot_search_exchange
        except ImportError as exc:
            raise HermesSlotRetrievalError("embedded KWRAG component is unavailable") from exc
        exchange = self._runtime.search_exchange(dict(request))
        verified = verify_slot_search_exchange(
            request,
            exchange.response,
            exchange.operation_receipt,
            expected_index_manifest=self._binding.expected_index_manifest,
            expected_pipeline_fingerprint=self._binding.expected_pipeline_fingerprint,
            max_result_characters=self._binding.max_result_characters,
        )
        receipt = {
            "schema_version": "hermes-kwrag-result-receipt-v1",
            "consumer_family": "hermes",
            "adapter_status": "verified_by_product_adapter",
            "component_digest": self._binding.component_digest,
            "runtime_binding_digest": self._binding.runtime_binding_digest,
            "request_id": verified.request_id,
            "operation_id": verified.operation_id,
            "run_id": verified.run_id,
            "attempt": verified.attempt,
            "index_manifest": verified.index_manifest,
            "pipeline_fingerprint": verified.pipeline_fingerprint,
            "result_status": verified.result_status,
            "result_digest": verified.result_digest,
            "operation_receipt_digest": verified.operation_receipt_digest,
            "result_count": verified.result_count,
            "result_characters": verified.result_characters,
        }
        receipt_bytes = canonical_json_bytes(receipt)
        receipt_digest = "sha256:" + hashlib.sha256(receipt_bytes).hexdigest()
        written_digest = self._receipt_sink.write(receipt)
        if written_digest != receipt_digest:
            raise HermesSlotRetrievalError("written result receipt digest is not bound")
        verified_results = verified.results()
        canonical_results_bytes = canonical_json_bytes(verified_results)
        canonical_result_receipt_bytes = canonical_json_bytes(receipt)
        if (
            "sha256:" + hashlib.sha256(canonical_results_bytes).hexdigest()
            != receipt["result_digest"]
        ):
            raise HermesSlotRetrievalError("verified result payload digest is not bound")
        return HermesSlotRetrievalResult(
            results=tuple(json.loads(canonical_results_bytes.decode("utf-8"))),
            result_receipt=json.loads(canonical_result_receipt_bytes.decode("utf-8")),
            result_receipt_digest=receipt_digest,
            result_receipt_status="written",
            _canonical_results_bytes=canonical_results_bytes,
            _canonical_result_receipt_bytes=canonical_result_receipt_bytes,
            _receipt_sink=self._receipt_sink,
        )
