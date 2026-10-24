"""Typed vault items: encoding, references, TOTP, URL matching, and the vault/API seams."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kiro_crew.dashboard.handlers.secrets import setup_secrets_routes
from kiro_crew.mcp_gateway.secret_uri import resolve_secret_uris
from kiro_crew.secrets import items
from kiro_crew.secrets.vault import SecretVault, entry_item_type

# RFC 6238 Appendix B test vector seed (ASCII "12345678901234567890" in base32).
_RFC_SEED = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"
_LOGIN = {
    "username": "zezhexu",
    "password": "pw-1",
    "totp": _RFC_SEED,
    "urls": [{"href": "https://github.com", "match": "domain"}],
    "notes": "",
}


# ── items: validation and encoding ──


def test_secret_item_encodes_to_its_bare_value() -> None:
    assert items.encode_item("secret", {"value": " sk-1 "}) == " sk-1 "


def test_login_encodes_to_canonical_json_and_decodes_back() -> None:
    plaintext = items.encode_item("login", {**_LOGIN, "extra": "dropped"})
    fields = items.decode_item("login", plaintext)
    assert fields == _LOGIN
    assert "extra" not in plaintext


@pytest.mark.parametrize(
    ("item_type", "fields", "code"),
    [
        ("login", {"username": "u"}, "missing_field"),
        ("login", {"password": "p", "totp": "not base32!!"}, "invalid_totp"),
        ("login", {"password": "p", "urls": ["ftp://x"]}, "invalid_url"),
        (
            "login",
            {"password": "p", "urls": [{"href": "https://a", "match": "regex"}]},
            "invalid_match",
        ),
        ("login", {"password": 1}, "invalid_field_type"),
        ("note", {"text": "   "}, "missing_field"),
        ("secret", {"value": "x" * (items.MAX_FIELD_CHARS + 1)}, "field_too_long"),
        ("card", {"number": "1"}, "unknown_type"),
    ],
)
def test_validate_item_refuses_with_a_stable_code(item_type: str, fields: dict, code: str) -> None:
    with pytest.raises(items.ItemError) as excinfo:
        items.validate_item(item_type, fields)
    assert excinfo.value.code == code


def test_login_url_given_as_a_bare_string_gets_the_default_match() -> None:
    fields = items.validate_item("login", {"password": "p", "urls": ["https://Example.com/login"]})
    assert fields["urls"] == [{"href": "https://Example.com/login", "match": "domain"}]


def test_public_view_shows_username_and_urls_but_never_a_credential() -> None:
    view = items.public_view("login", items.encode_item("login", _LOGIN))
    assert view == {"username": "zezhexu", "urls": _LOGIN["urls"], "has_totp": True}
    assert items.public_view("secret", "sk-1") == {}
    assert items.public_view("note", items.encode_item("note", {"text": "t"})) == {}


# ── items: references ──


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("GITHUB", ("GITHUB", None)),
        ("GITHUB#username", ("GITHUB", "username")),
        ("GITHUB#", ("GITHUB", "")),
        ("A#b#c", ("A", "b#c")),
    ],
)
def test_parse_reference_splits_on_the_first_separator(ref: str, expected: tuple) -> None:
    assert items.parse_reference(ref) == expected


def test_bare_reference_means_the_default_field() -> None:
    plaintext = items.encode_item("login", _LOGIN)
    assert items.resolve_field("login", plaintext, None) == "pw-1"
    assert items.resolve_field("login", plaintext, "username") == "zezhexu"
    assert items.resolve_field("login", plaintext, "url") == "https://github.com"
    assert items.resolve_field("secret", "sk-1", None) == "sk-1"
    assert items.resolve_field("note", items.encode_item("note", {"text": "t"}), None) == "t"


def test_unknown_field_fails_closed() -> None:
    with pytest.raises(items.ItemError) as excinfo:
        items.resolve_field("secret", "sk-1", "username")
    assert excinfo.value.code == "unknown_field"


# ── items: TOTP ──


@pytest.mark.parametrize(
    ("now", "code"),
    [(59, "94287082"), (1111111109, "07081804"), (20000000000, "65353130")],
)
def test_totp_matches_rfc_6238_sha1_vectors(now: int, code: str) -> None:
    assert items.totp_code(f"otpauth://totp/x?secret={_RFC_SEED}&digits=8", now=now) == code


def test_totp_reference_resolves_to_the_current_code_not_the_seed() -> None:
    plaintext = items.encode_item("login", _LOGIN)
    assert items.resolve_field("login", plaintext, "totp", now=59) == "287082"
    assert items.totp_seconds_left(_RFC_SEED, now=59) == 1


def test_totp_reference_on_a_login_without_a_seed_fails_closed() -> None:
    plaintext = items.encode_item("login", {**_LOGIN, "totp": ""})
    with pytest.raises(items.ItemError) as excinfo:
        items.resolve_field("login", plaintext, "totp")
    assert excinfo.value.code == "no_totp"


# ── items: URL matching ──


@pytest.mark.parametrize(
    ("match", "page", "expected"),
    [
        ("domain", "https://github.com/login", True),
        ("domain", "https://gist.github.com/x", True),
        ("domain", "https://notgithub.com/", False),
        ("domain", "https://github.com.evil.example/", False),
        ("host", "https://github.com/login", True),
        ("host", "https://gist.github.com/", False),
        ("exact", "https://github.com/", True),
        ("exact", "https://github.com/login", False),
        ("never", "https://github.com/", False),
        ("domain", "file:///etc/passwd", False),
    ],
)
def test_url_matches(match: str, page: str, expected: bool) -> None:
    assert items.url_matches("https://github.com", match, page) is expected


# ── vault seam ──


def test_vault_round_trips_a_login_and_binds_its_type_into_the_ciphertext(tmp_path: Path) -> None:
    vault = SecretVault(tmp_path)
    revision = asyncio.run(vault.set_item_with_revision("GITHUB", "login", _LOGIN))
    got = vault.get_item("GITHUB")
    assert got is not None
    item_type, plaintext, got_revision = got
    assert (item_type, got_revision) == ("login", revision)
    assert items.decode_item("login", plaintext.reveal()) == _LOGIN
    # The plain reader answers the default field, so an existing consumer keeps working.
    assert vault.get("GITHUB").reveal() == "pw-1"  # type: ignore[union-attr]
    assert vault.get_many(["GITHUB"])["GITHUB"].reveal() == "pw-1"  # type: ignore[union-attr]
    # Relabelling the entry's type in the store does not read it as another type.
    store = tmp_path / ".vault" / "secrets.enc"
    envelope = json.loads(store.read_text())
    envelope["entries"]["GITHUB"]["type"] = "note"
    store.write_text(json.dumps(envelope))
    with pytest.raises(Exception):
        SecretVault(tmp_path).get_item("GITHUB")


def test_a_plain_secret_entry_keeps_its_two_field_shape(tmp_path: Path) -> None:
    vault = SecretVault(tmp_path)
    asyncio.run(vault.set_item_with_revision("API", "secret", {"value": "sk-1"}))
    asyncio.run(vault.set_with_revision("API2", "sk-2"))
    entries = json.loads((tmp_path / ".vault" / "secrets.enc").read_text())["entries"]
    assert set(entries["API"]) == set(entries["API2"]) == {"nonce", "ct"}
    assert entry_item_type(entries["API"]) == "secret"
    rows = {row["name"]: row for row in vault.list_items()}
    assert rows["API"]["type"] == "secret" and "updated_at" not in rows["API"]


def test_list_items_reads_type_and_updated_at_without_decrypting(tmp_path: Path) -> None:
    vault = SecretVault(tmp_path)
    asyncio.run(vault.set_item_with_revision("N", "note", {"text": "hello"}))
    (row,) = vault.list_items()
    assert row["name"] == "N" and row["type"] == "note" and row["updated_at"]
    assert vault.get_items_many(["N", "missing"])["missing"] is None


# ── secret_uri seam ──


def test_resolve_secret_uris_picks_fields_and_defaults(tmp_path: Path) -> None:
    vault = SecretVault(tmp_path)
    asyncio.run(vault.set_item_with_revision("GITHUB", "login", _LOGIN))
    vault.set_sync("PLAIN", "sk-1")
    env = {
        "GH_USER": "secret://GITHUB#username",
        "GH_PASS": "secret://GITHUB",
        "API": "secret://PLAIN",
        "KEEP": "literal",
    }
    resolved, keys = resolve_secret_uris(env, tmp_path)
    assert resolved == {"GH_USER": "zezhexu", "GH_PASS": "pw-1", "API": "sk-1", "KEEP": "literal"}
    assert keys == {"GH_USER", "GH_PASS", "API"}


def test_resolve_secret_uris_refuses_an_unknown_field_naming_only_the_key(tmp_path: Path) -> None:
    vault = SecretVault(tmp_path)
    vault.set_sync("PLAIN", "sk-1")
    with pytest.raises(ValueError) as excinfo:
        resolve_secret_uris({"API": "secret://PLAIN#username"}, tmp_path)
    message = str(excinfo.value)
    assert "'API'" in message and "unknown_field" in message
    assert "PLAIN" not in message and "sk-1" not in message


# ── dashboard API seam ──


class _FakeState:
    owner_id = ""


def _app() -> web.Application:
    @web.middleware
    async def _identity(request, handler):
        request["user"] = "local-app"
        request["app"] = ""
        return await handler(request)

    app = web.Application(middlewares=[_identity])
    app["state"] = _FakeState()
    setup_secrets_routes(app)
    return app


@pytest.mark.asyncio
async def test_post_typed_item_then_list_shows_public_fields_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import kiro_crew.dashboard.handlers.secrets as handlers

    monkeypatch.setattr(handlers, "config_dir", lambda: str(tmp_path))
    async with TestClient(TestServer(_app())) as client:
        await _exercise_typed_round_trip(client)


async def _exercise_typed_round_trip(client: TestClient) -> None:
    resp = await client.post(
        "/api/secrets", json={"name": "GITHUB", "type": "login", "fields": _LOGIN}
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body["type"] == "login" and body["revision"]

    resp = await client.post("/api/secrets", json={"name": "PLAIN", "value": "sk-1"})
    assert resp.status == 200

    resp = await client.get("/api/secrets")
    listing = await resp.json()
    assert listing["names"] == ["GITHUB", "PLAIN"]
    by_name = {item["name"]: item for item in listing["items"]}
    assert by_name["GITHUB"]["type"] == "login"
    assert by_name["GITHUB"]["fields"] == {
        "username": "zezhexu",
        "urls": _LOGIN["urls"],
        "has_totp": True,
    }
    assert by_name["PLAIN"]["type"] == "secret" and by_name["PLAIN"].get("fields") is None
    text = json.dumps(listing)
    assert "pw-1" not in text and _RFC_SEED not in text and "sk-1" not in text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"name": "X", "type": "login", "value": "p"}, "missing_fields"),
        ({"name": "X", "type": "login", "fields": {"username": "u"}}, "missing_field"),
        ({"name": "X", "type": "card", "fields": {}}, "invalid_type"),
        ({"name": "X", "type": "note", "fields": "text"}, "invalid_fields"),
        ({"name": "A#B", "value": "v"}, "invalid_name"),
    ],
)
async def test_post_refuses_a_bad_typed_body_with_a_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: dict, code: str
) -> None:
    import kiro_crew.dashboard.handlers.secrets as handlers

    monkeypatch.setattr(handlers, "config_dir", lambda: str(tmp_path))
    async with TestClient(TestServer(_app())) as client:
        resp = await client.post("/api/secrets", json=body)
        assert resp.status == 400
        assert (await resp.json())["code"] == code
    assert not (tmp_path / ".vault" / "secrets.enc").exists()
