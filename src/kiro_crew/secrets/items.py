"""Typed vault items: what one encrypted entry holds, and how a reference picks a field.

A vault entry is one ciphertext under one name. This module decides what the
PLAINTEXT inside it looks like, per item type, and nothing about how it is
stored or who may read it (that is :mod:`kiro_crew.secrets.vault` and the
sandbox).

Three types:

``secret``
    The plaintext IS the value, with no wrapper. Every entry stored before item
    types existed is this type, byte for byte, so nothing migrates.
``login``
    A website credential: ``username``, ``password``, optional ``totp``
    (an ``otpauth://`` URI or a bare base32 seed), ``urls`` (each with a
    ``match`` mode deciding which pages it may fill) and ``notes``. Stored as
    one JSON object. The URL and username sit INSIDE the ciphertext: a store
    that leaked would say which sites a person has accounts on otherwise.
``note``
    Free text under ``text``.

A reference is ``secret://NAME`` or ``secret://NAME#FIELD``. With no field it
resolves to the type's DEFAULT field (``password`` for a login, ``text`` for a
note, the value for a secret), so a consumer written for a plain secret keeps
working when the entry becomes a login. ``#totp`` resolves to the CURRENT code,
never the seed. An unknown field fails closed: a reference never degrades to a
literal.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import struct
import time
from collections.abc import Mapping
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlsplit

ITEM_TYPE_SECRET = "secret"
ITEM_TYPE_LOGIN = "login"
ITEM_TYPE_NOTE = "note"
ITEM_TYPES: frozenset[str] = frozenset({ITEM_TYPE_SECRET, ITEM_TYPE_LOGIN, ITEM_TYPE_NOTE})

#: Which pages a login's URL lets the fill path act on. ``domain``: the stored
#: host and its subdomains. ``host``: that exact host and port. ``exact``: the
#: whole URL. ``never``: the URL is a label, not a permit.
MATCH_DOMAIN = "domain"
MATCH_HOST = "host"
MATCH_EXACT = "exact"
MATCH_NEVER = "never"
MATCH_MODES: tuple[str, ...] = (MATCH_DOMAIN, MATCH_HOST, MATCH_EXACT, MATCH_NEVER)

#: Separates the item name from the field in a reference.
REFERENCE_FIELD_SEPARATOR = "#"

FIELD_VALUE = "value"
FIELD_USERNAME = "username"
FIELD_PASSWORD = "password"
FIELD_TOTP = "totp"
FIELD_NOTES = "notes"
FIELD_URL = "url"
FIELD_TEXT = "text"

#: The fields a reference may name, per type, and the one a bare reference means.
REFERENCE_FIELDS: dict[str, frozenset[str]] = {
    ITEM_TYPE_SECRET: frozenset({FIELD_VALUE}),
    ITEM_TYPE_LOGIN: frozenset(
        {FIELD_USERNAME, FIELD_PASSWORD, FIELD_TOTP, FIELD_NOTES, FIELD_URL}
    ),
    ITEM_TYPE_NOTE: frozenset({FIELD_TEXT}),
}
DEFAULT_FIELD: dict[str, str] = {
    ITEM_TYPE_SECRET: FIELD_VALUE,
    ITEM_TYPE_LOGIN: FIELD_PASSWORD,
    ITEM_TYPE_NOTE: FIELD_TEXT,
}

#: Bounds on what one item may hold. A vault entry is a credential, not a
#: document, and the whole store is read into memory on every lookup.
MAX_FIELD_CHARS = 16_384
MAX_LOGIN_URLS = 16

_TOTP_ALGORITHMS = {"SHA1": hashlib.sha1, "SHA256": hashlib.sha256, "SHA512": hashlib.sha512}
_TOTP_DEFAULT_DIGITS = 6
_TOTP_DEFAULT_PERIOD = 30
_TOTP_MIN_DIGITS = 6
_TOTP_MAX_DIGITS = 10


class ItemError(ValueError):
    """An item that cannot be stored or a field that cannot be resolved.

    ``code`` is a stable machine-readable reason for an API response; the
    message is for a person. Neither ever carries a field VALUE.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# ── References ──


def parse_reference(ref: str) -> tuple[str, Optional[str]]:
    """Split the part after ``secret://`` into ``(name, field)``.

    The field is what follows the FIRST ``#``; ``None`` when there is none. The
    name is returned as written: whether it is storable is the caller's check,
    the same one a plain reference gets.
    """
    name, sep, field = ref.partition(REFERENCE_FIELD_SEPARATOR)
    return (name, field) if sep else (ref, None)


def resolve_field(
    item_type: str, plaintext: str, field: Optional[str], *, now: float | None = None
) -> str:
    """The value a reference to *field* of this item means.

    *plaintext* is the decrypted entry. A ``None`` field is the type's default.
    ``totp`` returns the code valid at *now* (``time.time()`` when omitted).
    Raises :class:`ItemError` for a field the type does not have, so a reference
    fails closed rather than resolving to something else.
    """
    _check_type(item_type)
    wanted = DEFAULT_FIELD[item_type] if field is None else field
    if wanted not in REFERENCE_FIELDS[item_type]:
        raise ItemError(
            "unknown_field",
            f"a {item_type} item has no field {wanted!r}; "
            f"one of {sorted(REFERENCE_FIELDS[item_type])}",
        )
    if item_type == ITEM_TYPE_SECRET:
        return plaintext
    fields = decode_item(item_type, plaintext)
    if item_type == ITEM_TYPE_NOTE:
        return str(fields.get(FIELD_TEXT, ""))
    if wanted == FIELD_TOTP:
        seed = fields.get(FIELD_TOTP) or ""
        if not seed:
            raise ItemError("no_totp", "this login has no one-time password seed")
        return totp_code(seed, now=now)
    if wanted == FIELD_URL:
        urls = fields.get("urls") or []
        return str(urls[0]["href"]) if urls else ""
    return str(fields.get(wanted, ""))


# ── Encoding ──


def _check_type(item_type: str) -> None:
    if item_type not in ITEM_TYPES:
        raise ItemError(
            "unknown_type", f"unknown item type {item_type!r}; one of {sorted(ITEM_TYPES)}"
        )


def _text(fields: Mapping[str, Any], key: str, *, required: bool = False) -> str:
    value = fields.get(key, "")
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ItemError("invalid_field_type", f"field {key!r} must be a string")
    if required and not value.strip():
        raise ItemError("missing_field", f"field {key!r} is required")
    if len(value) > MAX_FIELD_CHARS:
        raise ItemError(
            "field_too_long", f"field {key!r} is longer than {MAX_FIELD_CHARS} characters"
        )
    return value


def _urls(raw: Any) -> list[dict[str, str]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ItemError("invalid_field_type", "field 'urls' must be a list")
    if len(raw) > MAX_LOGIN_URLS:
        raise ItemError("too_many_urls", f"a login may carry at most {MAX_LOGIN_URLS} URLs")
    out: list[dict[str, str]] = []
    for entry in raw:
        if isinstance(entry, str):
            entry = {"href": entry}
        if not isinstance(entry, Mapping):
            raise ItemError("invalid_field_type", "each URL must be an object with 'href'")
        href = _text(entry, "href", required=True).strip()
        parts = urlsplit(href)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ItemError("invalid_url", "each URL must be an http(s) URL with a host")
        match = entry.get("match", MATCH_DOMAIN) or MATCH_DOMAIN
        if match not in MATCH_MODES:
            raise ItemError("invalid_match", f"URL match must be one of {list(MATCH_MODES)}")
        out.append({"href": href, "match": match})
    return out


def validate_item(item_type: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    """The canonical field dict for *item_type*, or :class:`ItemError`.

    Unknown keys are dropped, every kept value is a string of bounded length,
    and a ``totp`` seed must decode. A ``secret`` needs a non-blank ``value``;
    a ``login`` needs a non-blank ``password``; a ``note`` needs ``text``.
    Values are kept as given (a credential may carry meaningful whitespace):
    only the emptiness check strips.
    """
    _check_type(item_type)
    if not isinstance(fields, Mapping):
        raise ItemError("invalid_fields", "fields must be an object")
    if item_type == ITEM_TYPE_SECRET:
        return {FIELD_VALUE: _text(fields, FIELD_VALUE, required=True)}
    if item_type == ITEM_TYPE_NOTE:
        return {FIELD_TEXT: _text(fields, FIELD_TEXT, required=True)}
    totp = _text(fields, FIELD_TOTP).strip()
    if totp:
        _parse_totp(totp)
    return {
        FIELD_USERNAME: _text(fields, FIELD_USERNAME),
        FIELD_PASSWORD: _text(fields, FIELD_PASSWORD, required=True),
        FIELD_TOTP: totp,
        "urls": _urls(fields.get("urls")),
        FIELD_NOTES: _text(fields, FIELD_NOTES),
    }


def encode_item(item_type: str, fields: Mapping[str, Any]) -> str:
    """The plaintext to encrypt for a validated item (see :func:`validate_item`)."""
    canonical = validate_item(item_type, fields)
    if item_type == ITEM_TYPE_SECRET:
        return canonical[FIELD_VALUE]
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def decode_item(item_type: str, plaintext: str) -> dict[str, Any]:
    """The field dict a decrypted entry of *item_type* holds.

    A ``secret`` answers ``{"value": plaintext}``. A typed entry whose plaintext
    is not a JSON object is corrupt and raises :class:`ItemError`: the type is
    authenticated with the ciphertext, so this means the store was damaged,
    not that an older format is in use.
    """
    _check_type(item_type)
    if item_type == ITEM_TYPE_SECRET:
        return {FIELD_VALUE: plaintext}
    try:
        fields = json.loads(plaintext)
    except ValueError:
        fields = None
    if not isinstance(fields, dict):
        raise ItemError("corrupt_item", f"a {item_type} item's content is not an object")
    return fields


def public_view(item_type: str, plaintext: str) -> dict[str, Any]:
    """The part of an item a listing may show without revealing a credential.

    A login shows its username, URLs and whether it has a one-time password; a
    secret or a note shows nothing (the whole content is the credential).
    """
    if item_type != ITEM_TYPE_LOGIN:
        return {}
    fields = decode_item(item_type, plaintext)
    return {
        FIELD_USERNAME: str(fields.get(FIELD_USERNAME, "")),
        "urls": [dict(u) for u in fields.get("urls") or [] if isinstance(u, Mapping)],
        "has_totp": bool(fields.get(FIELD_TOTP)),
    }


# ── TOTP (RFC 6238) ──


def _parse_totp(seed: str) -> tuple[bytes, int, int, Any]:
    """``(key, digits, period, digest)`` from an ``otpauth://`` URI or a bare base32 seed."""
    digits, period, algorithm, secret = _TOTP_DEFAULT_DIGITS, _TOTP_DEFAULT_PERIOD, "SHA1", seed
    if seed.lower().startswith("otpauth://"):
        parts = urlsplit(seed)
        if parts.netloc.lower() != "totp":
            raise ItemError("invalid_totp", "only otpauth://totp/ URIs are supported")
        query = parse_qs(parts.query)
        secret = unquote(query.get("secret", [""])[0])
        algorithm = query.get("algorithm", ["SHA1"])[0].upper()
        try:
            digits = int(query.get("digits", [str(_TOTP_DEFAULT_DIGITS)])[0])
            period = int(query.get("period", [str(_TOTP_DEFAULT_PERIOD)])[0])
        except ValueError:
            raise ItemError("invalid_totp", "otpauth digits and period must be integers") from None
    digest = _TOTP_ALGORITHMS.get(algorithm)
    if digest is None:
        raise ItemError(
            "invalid_totp", f"unsupported TOTP algorithm; one of {sorted(_TOTP_ALGORITHMS)}"
        )
    if not (_TOTP_MIN_DIGITS <= digits <= _TOTP_MAX_DIGITS) or period <= 0:
        raise ItemError("invalid_totp", "TOTP digits must be 6-10 and period positive")
    normalized = secret.replace(" ", "").replace("-", "").upper()
    normalized += "=" * (-len(normalized) % 8)
    try:
        key = base64.b32decode(normalized, casefold=True)
    except (ValueError, TypeError):
        raise ItemError("invalid_totp", "the TOTP seed is not base32") from None
    if not key:
        raise ItemError("invalid_totp", "the TOTP seed is empty")
    return key, digits, period, digest


def totp_code(seed: str, *, now: float | None = None) -> str:
    """The one-time code *seed* yields at *now* (``time.time()`` when omitted)."""
    key, digits, period, digest = _parse_totp(seed)
    counter = int((time.time() if now is None else now) // period)
    mac = hmac.new(key, struct.pack(">Q", counter), digest).digest()
    offset = mac[-1] & 0x0F
    code = (struct.unpack(">I", mac[offset : offset + 4])[0] & 0x7FFFFFFF) % (10**digits)
    return str(code).zfill(digits)


def totp_seconds_left(seed: str, *, now: float | None = None) -> int:
    """Seconds until the code from :func:`totp_code` changes."""
    _, _, period, _ = _parse_totp(seed)
    moment = time.time() if now is None else now
    return period - int(moment % period)


# ── URL matching ──


def url_matches(href: str, match: str, page_url: str) -> bool:
    """Whether a login URL with *match* mode permits acting on *page_url*.

    ``domain``: same host, or *page_url*'s host is a subdomain of it (a stored
    ``github.com`` covers ``gist.github.com``; a stored ``gist.github.com`` does
    not cover ``github.com``). Hosts compare case-insensitively; the scheme must
    be http(s) on both sides. ``host``: host and port equal. ``exact``: the URL
    equal up to a trailing slash. ``never``: false.
    """
    if match == MATCH_NEVER:
        return False
    stored, page = urlsplit(href), urlsplit(page_url)
    if page.scheme not in ("http", "https") or not page.hostname or not stored.hostname:
        return False
    if match == MATCH_EXACT:
        return href.rstrip("/") == page_url.rstrip("/")
    if match == MATCH_HOST:
        return stored.netloc.lower() == page.netloc.lower()
    stored_host, page_host = stored.hostname.lower(), page.hostname.lower()
    return page_host == stored_host or page_host.endswith("." + stored_host)
