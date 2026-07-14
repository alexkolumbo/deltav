from deltav.crypto import KeyPair, address_from_public, canonical_json, verify_signature


def test_roundtrip_seed():
    kp = KeyPair.generate()
    restored = KeyPair.from_seed_hex(kp.seed_hex)
    assert restored.address == kp.address
    assert restored.public_hex == kp.public_hex


def test_address_format():
    kp = KeyPair.generate()
    assert kp.address.startswith("dv1")
    assert len(kp.address) == 43
    assert address_from_public(kp.public_hex) == kp.address


def test_sign_verify():
    kp = KeyPair.generate()
    msg = canonical_json({"hello": "world", "n": 1})
    sig = kp.sign(msg)
    assert verify_signature(kp.public_hex, msg, sig)
    assert not verify_signature(kp.public_hex, msg + b"x", sig)
    other = KeyPair.generate()
    assert not verify_signature(other.public_hex, msg, sig)


def test_canonical_json_deterministic():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
