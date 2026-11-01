"""Signed requests from a node to the registry (ADR 0004, decision 3; ROADMAP G.4).

There are no passwords or tokens: every request a node sends carries its public key and the
signature, by its node key, of what the request is — method, path (with the query), the
SHA-256 of the body, the time and a nonce — as canonical JSON (`sign.canonical`). The registry
checks all of it before it reads the body (`verify_request`):

- the headers are there, once each, and well formed;
- the signature is that key's over exactly this method, path and body;
- the time is within `sign.SKEW` of the registry's clock, either way;
- the nonce was not used by this key before (`sign.Seen`; kept until the time window closes,
  after which the time check refuses the request anyway).

A request that fails is `sign.Refused` with its `Reason`, signed by the key it claims
(`Refused.signer`) when that much could be read. The path signed is the one the node sent: a
reverse proxy in front of the registry passes it on unchanged (docs/registry-deploy.md).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from desman.net import sign
from desman.net.sign import Reason, Refused

VERSION: Final = 1
KEY: Final = "X-Desman-Key"
"""The node's public key, base64 of its 32 raw bytes (`sign.encode_public`)."""
TIME: Final = "X-Desman-Time"
"""When the request was signed: UTC, whole seconds, `2026-01-01T12:00:00Z`."""
NONCE: Final = "X-Desman-Nonce"
"""32 hex digits, new for every request."""
SIGNATURE: Final = "X-Desman-Signature"
HEADERS: Final = (KEY, TIME, NONCE, SIGNATURE)
MAX_PATH: Final = 2048

_TIME_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"
_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_NONCE = re.compile(r"^[0-9a-f]{32}$")
_KEY = re.compile(r"^[A-Za-z0-9+/]{43}=$")
_SIGNATURE = re.compile(r"^[A-Za-z0-9+/]{86}==$")
_METHOD = re.compile(r"^[A-Z]{1,16}$")
_PATH = re.compile(r"^/[\x21-\x7e]*$")
"""Printable ASCII without spaces: what a request line carries."""


def body_hash(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def format_time(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError("a time without a time zone")
    return moment.astimezone(UTC).strftime(_TIME_FORMAT)


def message(method: str, path: str, body: bytes, key: str, time: str, nonce: str) -> bytes:
    """What is signed."""
    return sign.canonical(
        {
            "v": VERSION,
            "method": method,
            "path": path,
            "body": body_hash(body),
            "key": key,
            "time": time,
            "nonce": nonce,
        }
    )


def sign_request(
    key: sign.NodeKey,
    method: str,
    path: str,
    body: bytes,
    *,
    now: datetime,
    nonce: str | None = None,
) -> dict[str, str]:
    """The headers that make this request the node's."""
    if not _METHOD.match(method):
        raise ValueError(f"not a method: {method!r}")
    if len(path) > MAX_PATH or not _PATH.match(path):
        raise ValueError(f"not a request path: {path!r}")
    public = sign.encode_public(key.public)
    time = format_time(now)
    nonce = uuid.uuid4().hex if nonce is None else nonce
    signature = key.sign(message(method, path, body, public, time, nonce))
    return {
        KEY: public,
        TIME: time,
        NONCE: nonce,
        SIGNATURE: base64.b64encode(signature).decode("ascii"),
    }


@dataclass(frozen=True)
class Caller:
    """Who sent a request that passed every check."""

    key: Ed25519PublicKey
    public: str
    """As the header carried it (`sign.encode_public`)."""
    fingerprint: str
    time: datetime
    nonce: str


def _headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Ours by lower-case name; one of them twice is refused (which one would count?)."""
    wanted = {name.lower() for name in HEADERS}
    found: dict[str, str] = {}
    for name, value in headers:
        lower = name.lower()
        if lower not in wanted:
            continue
        if lower in found:
            raise Refused(Reason.MALFORMED, f"{name} given twice")
        found[lower] = value.strip()
    if missing := [name for name in HEADERS if name.lower() not in found]:
        raise Refused(Reason.MALFORMED, f"missing {', '.join(missing)}")
    return found


def verify_request(
    method: str,
    path: str,
    body: bytes,
    headers: Iterable[tuple[str, str]],
    *,
    now: datetime,
    seen: sign.Seen,
) -> Caller:
    """The node that signed this request; anything else is `Refused`."""
    found = _headers(headers)
    public, time_text = found[KEY.lower()], found[TIME.lower()]
    nonce, signature_text = found[NONCE.lower()], found[SIGNATURE.lower()]
    if not _METHOD.match(method) or len(path) > MAX_PATH or not _PATH.match(path):
        raise Refused(Reason.MALFORMED, "not a request line")
    if not _KEY.match(public):
        raise Refused(Reason.MALFORMED, f"{KEY} is not a public key")
    try:
        key = sign.decode_public(public)
    except sign.KeyFileError as e:
        raise Refused(Reason.MALFORMED, str(e)) from e
    fingerprint = sign.fingerprint(key)

    def refuse(reason: Reason, detail: str) -> Refused:
        return Refused(reason, detail, signer=fingerprint)

    if not _TIME.match(time_text):
        raise refuse(Reason.MALFORMED, f"{TIME} is not a UTC time")
    try:
        time = datetime.strptime(time_text, _TIME_FORMAT).replace(tzinfo=UTC)
    except ValueError as e:
        raise refuse(Reason.MALFORMED, f"{TIME}: {e}") from e
    if not _NONCE.match(nonce):
        raise refuse(Reason.MALFORMED, f"{NONCE} is not 32 hex digits")
    if not _SIGNATURE.match(signature_text):
        raise refuse(Reason.MALFORMED, f"{SIGNATURE} is not a signature")
    try:
        key.verify(
            base64.b64decode(signature_text, validate=True),
            message(method, path, body, public, time_text, nonce),
        )
    except (InvalidSignature, binascii.Error) as e:
        raise refuse(Reason.FORGED, "bad signature") from e

    now = now.astimezone(UTC)
    if time > now + sign.SKEW:
        raise refuse(Reason.NOT_YET, f"signed at {time_text}")
    if time < now - sign.SKEW:
        raise refuse(Reason.EXPIRED, f"signed at {time_text}")
    if not seen.add(f"{fingerprint}:{nonce}", time + sign.SKEW, now):
        raise refuse(Reason.REPLAY, "this nonce was used already")
    return Caller(key, public, fingerprint, time, nonce)
