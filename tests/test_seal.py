"""Messenger cryptography: identity, integrity, dual wraps and bounded payloads."""

from __future__ import annotations

from datetime import timedelta

import pytest
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from pydantic import ValidationError

from conftest import FakeClock
from desman.net import seal, sign


def test_deterministic_published_key() -> None:
    key = sign.NodeKey.generate()
    assert seal.publish(key) == seal.publish(key)
    assert (
        seal.messenger_private(key).private_bytes_raw()
        == seal.messenger_private(key).private_bytes_raw()
    )
    seal.verify(seal.publish(key), key.public)
    with pytest.raises(seal.SealError):
        seal.verify(seal.publish(key), sign.NodeKey.generate().public)
    forged = seal.publish(key).model_copy(update={"signature": seal.encode(bytes(64))})
    with pytest.raises(seal.SealError):
        seal.verify(forged, key.public)


@pytest.mark.parametrize("text", ["hello\n\t", "😀" * 2000, "\u200d" * 2000])
def test_roundtrip_and_bounds(clock: FakeClock, text: str) -> None:
    sender, recipient = sign.NodeKey.generate(), sign.NodeKey.generate()
    envelope = seal.seal(
        sender, recipient.fingerprint, seal.publish(recipient).x25519, text, clock()
    )
    assert len(seal.decode(envelope.ciphertext)) <= 16384
    assert seal.Sealed.model_validate_json(envelope.model_dump_json()) == envelope
    for key in (sender, recipient):
        signed = seal.open_with(seal.content_key(key, envelope), envelope)
        letter = seal.check(signed, sender.public, sender.fingerprint, recipient.fingerprint)
        assert letter.text == text and letter.written_at == clock()
    with pytest.raises(seal.SealError):
        seal.content_key(sign.NodeKey.generate(), envelope)


@pytest.mark.parametrize("part", ["ciphertext", "nonce", "to", "own", "swap"])
def test_tamper(clock: FakeClock, part: str) -> None:
    sender, recipient = sign.NodeKey.generate(), sign.NodeKey.generate()
    envelope = seal.seal(
        sender, recipient.fingerprint, seal.publish(recipient).x25519, "private", clock()
    )
    if part == "swap":
        broken = envelope.model_copy(update={"to": envelope.own, "own": envelope.to})
    elif part in {"to", "own"}:
        wrap = getattr(envelope, part)
        raw = seal.decode(wrap.key)
        broken = envelope.model_copy(
            update={
                part: wrap.model_copy(update={"key": seal.encode(bytes([raw[0] ^ 1]) + raw[1:])})
            }
        )
    else:
        raw = seal.decode(getattr(envelope, part))
        broken = envelope.model_copy(update={part: seal.encode(bytes([raw[0] ^ 1]) + raw[1:])})
    readers = [recipient] if part == "to" else [sender] if part == "own" else [sender, recipient]
    for key in readers:
        with pytest.raises(seal.SealError):
            seal.content_key(key, broken)


def test_letter_authentication(clock: FakeClock) -> None:
    sender, recipient = sign.NodeKey.generate(), sign.NodeKey.generate()
    envelope = seal.seal(
        sender, recipient.fingerprint, seal.publish(recipient).x25519, "private", clock()
    )
    content = seal.content_key(recipient, envelope)
    signed = seal.open_with(content, envelope)
    for altered in [
        signed.model_copy(update={"signature": seal.encode(bytes(64))}),
        signed.model_copy(
            update={"letter": signed.letter.model_copy(update={"recipient": sender.fingerprint})}
        ),
        signed.model_copy(update={"letter": signed.letter.model_copy(update={"text": "forged"})}),
        signed.model_copy(
            update={
                "letter": signed.letter.model_copy(
                    update={"written_at": clock() + timedelta(seconds=1)}
                )
            }
        ),
    ]:
        with pytest.raises(seal.SealError):
            seal.check(altered, sender.public, sender.fingerprint, recipient.fingerprint)
    with pytest.raises(seal.SealError):
        seal.check(signed, recipient.public, sender.fingerprint, recipient.fingerprint)
    with pytest.raises(seal.SealError):
        seal.open_with(bytes(32), envelope)
    # Authenticated but malformed plaintext must also be refused.
    raw = ChaCha20Poly1305(content).encrypt(
        seal.decode(envelope.nonce), b"{}", b"desman message v1"
    )
    with pytest.raises(seal.SealError):
        seal.open_with(content, envelope.model_copy(update={"ciphertext": seal.encode(raw)}))


@pytest.mark.parametrize(
    "field,value",
    [("nonce", "AA=="), ("ciphertext", "!"), ("ciphertext", seal.encode(bytes(16385)))],
)
def test_wire_bounds(clock: FakeClock, field: str, value: str) -> None:
    key = sign.NodeKey.generate()
    envelope = seal.seal(key, key.fingerprint, seal.publish(key).x25519, "ok", clock())
    with pytest.raises(ValidationError):
        seal.Sealed.model_validate(envelope.model_dump() | {field: value})
