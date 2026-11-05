"""Authenticated end-to-end messenger envelopes; no storage or network access."""

from __future__ import annotations

import base64
import hashlib
import os
import unicodedata
from datetime import UTC, datetime
from typing import Annotated, Literal

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, ValidationError

from desman.net import sign


class SealError(Exception):
    """Invalid messenger key, envelope, or signed evidence."""


def encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def decode(text: str, length: int | None = None) -> bytes:
    try:
        raw = base64.b64decode(text, validate=True)
    except ValueError:
        raise SealError("invalid base64") from None
    if length is not None and len(raw) != length:
        raise SealError("invalid byte length")
    return raw


def _bytes(text: str, length: int) -> str:
    try:
        decode(text, length)
    except SealError as e:
        raise ValueError(str(e)) from e
    return text


Raw32 = Annotated[str, AfterValidator(lambda s: _bytes(s, 32))]
Signature = Annotated[str, AfterValidator(lambda s: _bytes(s, 64))]


def message_text(text: str) -> str:
    """Reject terminal controls and bidi overrides in human correspondence."""
    if any(
        (unicodedata.category(c) == "Cc" and c not in "\n\t")
        or "\u202a" <= c <= "\u202e"
        or "\u2066" <= c <= "\u2069"
        for c in text
    ):
        raise ValueError("control character in message")
    return text


Text = Annotated[str, Field(min_length=1, max_length=2000), AfterValidator(message_text)]


class Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MessengerKey(Frozen):
    node: sign.Fingerprint
    x25519: Raw32
    signature: Signature


def _derive(raw: bytes, info: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(raw)


def messenger_private(key: sign.NodeKey) -> X25519PrivateKey:
    return X25519PrivateKey.from_private_bytes(
        _derive(key._private.private_bytes_raw(), b"desman messenger x25519 v1")
    )


def _key_body(node: str, public: str) -> bytes:
    return sign.canonical({"kind": "messenger_key", "node": node, "x25519": public})


def publish(key: sign.NodeKey) -> MessengerKey:
    public = encode(messenger_private(key).public_key().public_bytes_raw())
    return MessengerKey(
        node=key.fingerprint,
        x25519=public,
        signature=encode(key._private.sign(_key_body(key.fingerprint, public))),
    )


def verify(mk: MessengerKey, node_public: Ed25519PublicKey) -> None:
    if sign.fingerprint(node_public) != mk.node:
        raise SealError("wrong node")
    try:
        node_public.verify(decode(mk.signature, 64), _key_body(mk.node, mk.x25519))
    except (InvalidSignature, ValueError):
        raise SealError("invalid messenger key signature") from None


class Letter(Frozen):
    v: Literal[1] = 1
    sender: sign.Fingerprint
    recipient: sign.Fingerprint
    text: Text
    nonce: str = Field(pattern=r"^[0-9a-f]{32}$")
    written_at: Annotated[AwareDatetime, AfterValidator(lambda dt: dt.astimezone(UTC))]


class SignedLetter(Frozen):
    letter: Letter
    signature: Signature


class Wrap(Frozen):
    ephemeral: Raw32
    key: Annotated[str, AfterValidator(lambda s: _bytes(s, 48))]


def _ciphertext(text: str) -> str:
    try:
        raw = decode(text)
    except SealError as e:
        raise ValueError(str(e)) from e
    if not 16 <= len(raw) <= 16384:
        raise ValueError("invalid ciphertext size")
    return text


class Sealed(Frozen):
    v: Literal[1] = 1
    nonce: Annotated[str, AfterValidator(lambda s: _bytes(s, 12))]
    ciphertext: Annotated[str, Field(max_length=21848), AfterValidator(_ciphertext)]
    to: Wrap
    own: Wrap


def _wrap(content: bytes, recipient: bytes, ciphertext: bytes) -> Wrap:
    ephemeral = X25519PrivateKey.generate()
    public = ephemeral.public_key().public_bytes_raw()
    shared = ephemeral.exchange(X25519PublicKey.from_public_bytes(recipient))
    wk = _derive(shared, b"desman wrap v1" + public + recipient)
    return Wrap(
        ephemeral=encode(public),
        key=encode(
            ChaCha20Poly1305(wk).encrypt(bytes(12), content, hashlib.sha256(ciphertext).digest())
        ),
    )


def seal(
    key: sign.NodeKey, recipient_node: str, recipient_x25519: str, text: str, now: datetime
) -> Sealed:
    try:
        letter = Letter(
            sender=key.fingerprint,
            recipient=recipient_node,
            text=text,
            nonce=os.urandom(16).hex(),
            written_at=now,
        )
        signed = SignedLetter(
            letter=letter,
            signature=encode(key._private.sign(sign.canonical(letter.model_dump(mode="json")))),
        )
        content, nonce = os.urandom(32), os.urandom(12)
        ciphertext = ChaCha20Poly1305(content).encrypt(
            nonce, sign.canonical(signed.model_dump(mode="json")), b"desman message v1"
        )
        return Sealed(
            nonce=encode(nonce),
            ciphertext=encode(ciphertext),
            to=_wrap(content, decode(recipient_x25519, 32), ciphertext),
            own=_wrap(content, messenger_private(key).public_key().public_bytes_raw(), ciphertext),
        )
    except (ValueError, ValidationError) as e:
        raise SealError("cannot seal message") from e


def content_key(key: sign.NodeKey, sealed: Sealed) -> bytes:
    private = messenger_private(key)
    public = private.public_key().public_bytes_raw()
    for wrap, role in ((sealed.to, "recipient"), (sealed.own, "sender")):
        try:
            ep = decode(wrap.ephemeral, 32)
            wk = _derive(
                private.exchange(X25519PublicKey.from_public_bytes(ep)),
                b"desman wrap v1" + ep + public,
            )
            content = ChaCha20Poly1305(wk).decrypt(
                bytes(12), decode(wrap.key, 48), hashlib.sha256(decode(sealed.ciphertext)).digest()
            )
            signed = open_with(content, sealed)
            if getattr(signed.letter, role) != key.fingerprint:
                continue
            return content
        except (InvalidTag, ValueError, SealError):
            continue
    raise SealError("message key cannot be opened")


def open_with(content_key: bytes, sealed: Sealed) -> SignedLetter:
    try:
        raw = ChaCha20Poly1305(content_key).decrypt(
            decode(sealed.nonce, 12), decode(sealed.ciphertext), b"desman message v1"
        )
        return SignedLetter.model_validate_json(raw)
    except (InvalidTag, ValueError, ValidationError):
        raise SealError("message cannot be decrypted or parsed") from None


def check(
    signed: SignedLetter, sender_public: Ed25519PublicKey, sender: str, recipient: str
) -> Letter:
    letter = signed.letter
    if (
        letter.sender != sender
        or sender != sign.fingerprint(sender_public)
        or letter.recipient != recipient
    ):
        raise SealError("wrong message participants")
    try:
        sender_public.verify(
            decode(signed.signature, 64), sign.canonical(letter.model_dump(mode="json"))
        )
    except (InvalidSignature, ValueError):
        raise SealError("invalid letter signature") from None
    return letter
