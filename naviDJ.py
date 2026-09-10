"""
AI-powered playlist generator for Subsonic/Navidrome music servers.

- Uses OpenAI or Ollama LLMs to generate playlists based on a vibe prompt.
- Selects relevant artists/albums/genres from library using LLM.
- Filters library using metadata-based weighted scoring.
- Generates final playlist via LLM from filtered candidates.

Usage:
    python naviDJ.py [--playlist_name NAME] [--prompt PROMPT] [--min_songs N] [--llm_mode openai|ollama]

Arguments:
    --playlist_name   Name of the playlist to create or update (default: naviDJ)
    --prompt          Vibe prompt for the playlist (required if not interactive)
    --min_songs       Target/exact number of songs in the playlist (default: 35)
    --llm_mode        LLM backend to use: openai or ollama (default: openai)

If arguments are omitted, the script will prompt for them interactively.
"""

import os
import math
import requests
import xml.etree.ElementTree as ET
import json
import random
import re
import difflib
import time

from tqdm import tqdm
from openai import (
    BadRequestError,
    OpenAI,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)
import argparse
from typing import List, Dict  # optional, only for type hints
from collections import Counter
import configparser
import logging
from embeddings import EmbeddingManager


# --------------------------------------------------
# CONFIG & LLM CLIENT SETUP
# --------------------------------------------------

# Load secrets from secrets.txt
secrets = configparser.ConfigParser()
secrets.read(os.path.join(os.path.dirname(__file__), "secrets.txt"))

DEFAULT_OPENAI_KEY = secrets.get("openai", "openai_key", fallback=None)
DEFAULT_OLLAMA_BASE = secrets.get("ollama", "ollama_base", fallback=None)
DEFAULT_OLLAMA_API_KEY = secrets.get("ollama", "api_key", fallback=None) or "ollama"
DEFAULT_CUSTOM_API_KEY = secrets.get("custom", "api_key", fallback=None)
DEFAULT_CUSTOM_BASE_URL = secrets.get("custom", "base_url", fallback=None)

DEFAULT_LLM_MODE = secrets.get("llm", "mode", fallback="openai").lower()
DEFAULT_LLM_MODEL = secrets.get("llm", "model", fallback=None)
DEFAULT_CHUNK_SIZE = int(secrets.get("llm", "chunk_size", fallback="500"))
DEFAULT_THINKING_ENABLED = secrets.get("llm", "thinking_enabled", fallback="on").lower() == "on"


SUBSONIC_BASE_URL = secrets.get("subsonic", "BASE_URL", fallback=None)
SUBSONIC_AUTH_PARAMS = {
    "u": secrets.get("subsonic", "USER", fallback=None),
    "p": secrets.get("subsonic", "PASSWORD", fallback=None),
    "v": secrets.get("subsonic", "API_VERSION", fallback="1.16.1"),
    "c": secrets.get("subsonic", "CLIENT", fallback="naviDJ"),
}

# Will be overwritten in `configure_llm` but need a placeholder so _llm_chat can be defined early.
LLM_MODE: str = DEFAULT_LLM_MODE  # 'openai' | 'ollama'
LLM_MODEL: str = DEFAULT_LLM_MODEL or ""  # auto-filled later
THINKING_ENABLED: bool = DEFAULT_THINKING_ENABLED
client: OpenAI | None = None  # global client instance
EMBEDDING_MODEL: str | None = None
embedding_manager: EmbeddingManager | None = None

# --------------------------------------------------
# HELPER FOR CLEANING LLM OUTPUT
# --------------------------------------------------

_THINK_CLOSED_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)

def _remove_think_tags(text: str) -> str:
    """
    Strip <think>...</think> blocks and any content before the final </think>.
    Handles: closed blocks, unclosed blocks (model hit token limit mid-think),
    malformed whitespace in tags, and multiple consecutive think blocks.
    """
    # Step 1: Strip all complete <think>...</think> blocks (non-greedy)
    text = _THINK_CLOSED_RE.sub("", text)

    # Step 2: If a </think> closing tag remains anywhere, discard everything before
    # and including it — catches cases where opening <think> was already stripped
    # but closing tag remained, or multiple blocks left a trailing closer
    if "</think>" in text.lower():
        idx = text.lower().rfind("</think>")
        text = text[idx + len("</think>"):]

    # Step 3: Strip unclosed <think> block — model hit token limit mid-think.
    # Everything from an unclosed <think> to end of string is reasoning noise.
    unclosed = re.search(r"<think>[\s\S]*$", text, re.IGNORECASE)
    if unclosed:
        text = text[:unclosed.start()]

    return text.strip()


def _split_artist_string(artist_string: str) -> list[str]:
    """
    Split a complex artist string into individual artist names.
    Handles various delimiters: ",", ";", "•", "&", "feat.", "featuring", "ft.", etc.
    Returns a list of cleaned individual artist names.
    """
    if not artist_string:
        return []
    # Split on comma, semicolon, bullet point, or specific words surrounded by whitespace
    pattern = r",|;|•|\s+&\s+|\s+feat\.?\s+|\s+featuring\s+|\s+ft\.?\s+"
    parts = re.split(pattern, artist_string, flags=re.IGNORECASE)

    artists = []
    for part in parts:
        artist = part.strip()
        if artist and artist not in artists:  # Avoid duplicates
            artists.append(artist)

    return artists


# --------------------------------------------------
# VARIANT DEDUP HELPERS
# --------------------------------------------------
# Collapse duplicate variants of the same recording (remasters, editions,
# bonus-track tags, feat. tags, duplicate rips across albums). Live versions
# are treated as DISTINCT recordings (is_live is part of the dedup key), so a
# live cut never collapses onto its studio counterpart, but two live variants
# of the same song do collapse together.

# A trailing "( ... )" / "[ ... ]" group or "- ..." dash segment is only
# stripped when its *entire* inner text is one of these recognised suffix
# markers. This is deliberately conservative so real titles like
# "(Don't Fear) The Reaper" are never touched.
_SUFFIX_MARKER_RE = re.compile(
    r"^\s*(?:"
    r"\d{0,4}\s*re-?master(?:ed)?(?:\s+\d{2,4})?"        # Remaster/Remastered, 2019 Remaster, Remastered 2019
    r"|(?:deluxe|anniversary|expanded|special|collector'?s|legacy|super\s+deluxe)(?:\s+edition)?"
    r"|deluxe\s+version"
    r"|single\s+version"
    r"|album\s+version"
    r"|(?:radio|extended)\s+(?:edit|version|mix)"
    r"|mono(?:\s+version)?"
    r"|stereo(?:\s+version)?"
    r"|bonus\s+track"
    r"|re-?recorded(?:\s+version)?"
    r"|(?:feat|ft|featuring)\.?\s+.+"                     # trailing feat. / ft. / featuring ...
    r"|live\b.*"                                          # any live marker text
    r")\s*$",
    re.IGNORECASE,
)

_TRAILING_PAREN_RE = re.compile(r"\s*[\(\[]([^\(\)\[\]]*)[\)\]]\s*$")
_TRAILING_DASH_RE = re.compile(r"\s*-\s+([^-]*?)\s*$")


def _is_remaster(title: str) -> bool:
    """True if the title carries a remaster marker (used for variant preference)."""
    return bool(re.search(r"re-?master", title or "", re.IGNORECASE))


def _is_live(title: str) -> bool:
    """Detect live-performance markers: '(Live', '[Live', '- Live', 'Live at ...'."""
    t = title or ""
    if re.search(r"[\(\[]\s*live\b", t, re.IGNORECASE):
        return True
    if re.search(r"-\s*live\b", t, re.IGNORECASE):
        return True
    if re.search(r"\blive\s+at\b", t, re.IGNORECASE):
        return True
    return False


def _strip_edition_suffixes(title: str) -> str:
    """
    Iteratively strip recognised trailing edition/version/live/feat suffixes
    (parenthetical, bracketed, or dash-led). Only recognised markers are
    removed; arbitrary parentheses are left intact.
    """
    t = title or ""
    prev = None
    while t != prev:
        prev = t
        m = _TRAILING_PAREN_RE.search(t)
        if m and _SUFFIX_MARKER_RE.match(m.group(1)):
            t = t[: m.start()].strip()
            continue
        m = _TRAILING_DASH_RE.search(t)
        if m and _SUFFIX_MARKER_RE.match(m.group(1)):
            t = t[: m.start()].strip()
            continue
    return t


def _normalize_title(title: str) -> str:
    """
    Normalise a track title for variant matching: strip recognised suffixes,
    lowercase, drop punctuation, and collapse whitespace.
    """
    t = _strip_edition_suffixes(title or "")
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t)  # punctuation-insensitive
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _variant_dedup_key(song: dict) -> tuple:
    """Dedup key: (normalized_title, primary_artist_lower, is_live)."""
    title = song.get("title") or ""
    artists = _split_artist_string(song.get("artist") or "")
    primary = artists[0].lower() if artists else ""
    return (_normalize_title(title), primary, _is_live(title))


def _variant_preferred(a: dict, b: dict) -> bool:
    """
    Return True if variant *a* should be preferred over *b* when collapsing a
    group. Preference: starred > remastered > higher playCount > arbitrary.
    """
    a_star, b_star = bool(a.get("starred")), bool(b.get("starred"))
    if a_star != b_star:
        return a_star
    a_rm, b_rm = _is_remaster(a.get("title") or ""), _is_remaster(b.get("title") or "")
    if a_rm != b_rm:
        return a_rm
    a_pc, b_pc = a.get("playCount") or 0, b.get("playCount") or 0
    if a_pc != b_pc:
        return a_pc > b_pc
    return False


def _dedup_variants(songs: list[dict]) -> list[dict]:
    """
    Collapse variant duplicates in *songs*, keeping the preferred representative
    of each group. First-appearance order is preserved (no reordering).
    """
    best: dict[tuple, dict] = {}
    order: list[tuple] = []
    for s in songs:
        k = _variant_dedup_key(s)
        if k not in best:
            best[k] = s
            order.append(k)
        elif _variant_preferred(s, best[k]):
            best[k] = s
    return [best[k] for k in order]


def calculate_metadata_score(song: dict) -> float:
    """Calculate score from play count, skip count, and recency."""
    score = 0.0

    # 1. Play count bonus: logarithmic, diminishing returns
    play_count = song.get("playCount", 0)
    if play_count > 0:
        score += math.log2(play_count + 1) * 0.05

    # 2. Skip count penalty: -0.1 for each 5 skips above 5
    skip_count = song.get("skipCount", 0)
    if skip_count > 5:
        score -= 0.1 * ((skip_count - 1) // 5)

    # 3. Recency penalty: -0.25 if played within last 48 hours
    last_play = song.get("lastPlayTime", 0)
    if last_play > 0:
        if last_play > (time.time() - 172800):  # < 48 hours = 172800 seconds
            score -= 0.25

    return score


def _era_bonus(release_year, era_range: tuple[int, int] | None) -> float:
    """
    Loose scoring bonus for a song's `releaseYear` against a detected prompt
    era. +1.0 when the year falls inside the range, +0.5 when it's within 5
    years outside either edge, 0.0 otherwise (including when no era was
    detected or the year is missing/unparseable). This is purely additive —
    it must never be used to filter or exclude a song.
    """
    if not era_range or not release_year:
        return 0.0
    try:
        year = int(release_year)
    except (TypeError, ValueError):
        return 0.0

    start, end = era_range
    if start <= year <= end:
        return 1.0
    if (start - 5) <= year < start or end < year <= (end + 5):
        return 0.5
    return 0.0


def configure_llm(mode: str = None, model: str = None) -> None:
    """Initialise the global `client`, `LLM_MODE`, and `LLM_MODEL` based on *mode* and *model*."""
    global LLM_MODE, LLM_MODEL, client, EMBEDDING_MODEL, embedding_manager

    # Use secrets.txt defaults if not provided
    mode = (mode or DEFAULT_LLM_MODE or "openai").lower()

    # Reload secrets to get latest config (important for frontend updates)
    secrets.read(os.path.join(os.path.dirname(__file__), "secrets.txt"))

    if mode not in {"openai", "ollama", "custom"}:
        raise ValueError(
            "Unsupported LLM_MODE. Choose 'openai', 'ollama', or 'custom'."
        )

    LLM_MODE = mode
    THINKING_ENABLED = secrets.get("llm", "thinking_enabled", fallback="on").lower() == "on"

    if mode == "openai":
        LLM_MODEL = model or DEFAULT_LLM_MODEL or "gpt-4o-mini"
        EMBEDDING_MODEL = secrets.get(
            "openai", "embedding_model", fallback="text-embedding-3-small"
        )
        client = OpenAI(api_key=DEFAULT_OPENAI_KEY)
        embedding_manager = EmbeddingManager(
            api_type="openai", model_name=EMBEDDING_MODEL, api_key=DEFAULT_OPENAI_KEY
        )
    elif mode == "ollama":
        LLM_MODEL = model or DEFAULT_LLM_MODEL or "gemma3n:latest"
        EMBEDDING_MODEL = secrets.get(
            "ollama", "embedding_model", fallback="nomic-embed-text:latest"
        )
        client = OpenAI(api_key=DEFAULT_OLLAMA_API_KEY, base_url=DEFAULT_OLLAMA_BASE)
        embedding_manager = EmbeddingManager(
            api_type="ollama",
            model_name=EMBEDDING_MODEL,
            base_url=DEFAULT_OLLAMA_BASE,
            api_key=DEFAULT_OLLAMA_API_KEY,
        )
    else:  # custom
        LLM_MODEL = model or DEFAULT_LLM_MODEL or "gpt-4o-mini"
        client = OpenAI(
            api_key=DEFAULT_CUSTOM_API_KEY, base_url=DEFAULT_CUSTOM_BASE_URL
        )
        embedding_manager = None


# --------------------------------------------------
# SUBSONIC HELPERS
# --------------------------------------------------


def _parse_subsonic_genres(raw_song: dict) -> list[str]:
    """
    Extract the full genre list for a raw Subsonic/OpenSubsonic song JSON
    object. Plain Subsonic only exposes a singular `genre` string;
    OpenSubsonic servers also expose a `genres` array of `{"name": ...}`
    objects. The legacy `genre` value is kept first (for backward
    compatibility) followed by any additional distinct names from the array.
    Defensive: returns [] when neither field is present.
    """
    genres: list[str] = []
    primary = raw_song.get("genre")
    if primary:
        genres.append(primary)
    for g in raw_song.get("genres") or []:
        name = g.get("name") if isinstance(g, dict) else g if isinstance(g, str) else None
        if name and name not in genres:
            genres.append(name)
    return genres


def _song_genres(song: dict) -> list[str]:
    """
    Full genre list for a song dict already produced by
    `fetch_all_subsonic_songs` (i.e. reads our internal `genres`/`genre`
    keys, not the raw API response). Falls back to the singular `genre`
    field when `genres` is absent, for defensiveness.
    """
    genres = song.get("genres")
    if genres:
        return genres
    g = song.get("genre")
    return [g] if g else []


def _song_embedding_text(song: dict) -> str:
    """
    Build the text used to embed a song for semantic search. Includes title,
    artist, and (when present) album, genres, and release year so the vector
    captures more than just "title by artist". Empty/missing fields are
    omitted cleanly rather than leaving literal blanks like "album: ".
    """
    title = song.get("title") or ""
    artist = song.get("artist") or ""
    parts = [f"{title} by {artist}".strip()]

    album = song.get("album")
    if album:
        parts.append(f"album: {album}")

    genres = _song_genres(song)
    if genres:
        parts.append(f"genres: {', '.join(genres)}")

    year = song.get("releaseYear")
    if year:
        parts.append(f"year: {year}")

    return " | ".join(parts)


def fetch_starred_ids() -> set[str]:
    resp = requests.get(
        f"{SUBSONIC_BASE_URL}/getStarred2.view",
        params={**SUBSONIC_AUTH_PARAMS, "f": "json"},
    )
    resp.raise_for_status()
    songs = resp.json()["subsonic-response"]["starred2"].get("song", [])
    return {s["id"] for s in songs}


def fetch_all_subsonic_songs() -> list[dict]:
    all_songs: list[dict] = []
    song_offset, song_count = 0, 500
    bar = tqdm(desc="Fetching songs", unit="song", dynamic_ncols=True, ascii=True)
    starred = fetch_starred_ids()  # fetched once per run, not once per page

    while True:
        resp = requests.get(
            f"{SUBSONIC_BASE_URL}/search3.view",
            params={
                **SUBSONIC_AUTH_PARAMS,
                "query": "",
                "f": "json",
                "songCount": song_count,
                "songOffset": song_offset,
            },
        )
        resp.raise_for_status()
        data = resp.json()["subsonic-response"].get("searchResult3", {})
        songs = data.get("song", [])

        for s in songs:
            genres = _parse_subsonic_genres(s)
            all_songs.append(
                {
                    "id": s.get("id"),
                    "title": s.get("title"),
                    "artist": s.get("artist"),
                    "genre": genres[0] if genres else None,
                    "genres": genres,
                    "album": s.get("album"),
                    "releaseYear": s.get("year"),
                    "starred": s.get("id") in starred,
                    "playCount": s.get("playCount", 0),
                    "skipCount": s.get("skipCount", 0),
                    "lastPlayTime": s.get("lastPlayTime", 0),
                }
            )
        bar.update(len(songs))
        if len(songs) < song_count:
            break
        song_offset += song_count
    bar.close()
    print()  # Ensure newline after progress bar
    return all_songs


# --------------------------------------------------
# LLM UTILITIES - artist and genre selection
# --------------------------------------------------

# Transient errors are worth retrying (rate limit, timeout, connection drop,
# 5xx from the backend); auth/bad-request/etc. are not and should fail fast.
_TRANSIENT_LLM_ERRORS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)
_LLM_RETRY_DELAYS = (2, 4, 8)  # seconds; exponential backoff between retries


def _llm_chat(messages: list[dict], _retries: int = 1, max_tokens: int | None = None) -> str:
    """Universal chat helper that works for both OpenAI & Ollama and always returns clean JSON-only content."""

    # Sanitize: every message content field must be a plain string.
    # Models that emit <think> blocks or return None content will cause
    # "Content path must be a string" crashes in the OpenAI client otherwise.
    safe_messages = []
    for msg in messages:
        m = dict(msg)  # shallow copy
        c = m.get("content")
        if c is None:
            m["content"] = ""
        elif not isinstance(c, str):
            m["content"] = json.dumps(c) if isinstance(c, (dict, list)) else str(c)
        safe_messages.append(m)

    create_kwargs = dict(
        model=LLM_MODEL,
        messages=safe_messages,
        stream=False,
        response_format={"type": "json_object"},
    )
    if max_tokens:
        create_kwargs["max_tokens"] = max_tokens
    if LLM_MODE != "openai":
        # chat_id required by Open WebUI 0.9.x for external API clients (ignored by native Ollama)
        create_kwargs["extra_body"] = {
            "chat_id": "navidj-api",
            "chat_template_kwargs": {"enable_thinking": THINKING_ENABLED},
        }

    def _call(kwargs: dict):
        """
        Call the chat completion endpoint, retrying transient errors (rate
        limit, timeout, connection failure, 5xx) with exponential backoff
        (2s/4s/8s). Non-transient errors (auth, bad request, etc.) propagate
        on the first failure so the caller's existing fallback/workaround
        handling still applies without delay.
        """
        last_err = None
        total_attempts = len(_LLM_RETRY_DELAYS) + 1
        for attempt, delay in enumerate((0,) + _LLM_RETRY_DELAYS, start=1):
            if delay:
                time.sleep(delay)
            try:
                return client.chat.completions.create(**kwargs)
            except _TRANSIENT_LLM_ERRORS as e:
                last_err = e
                if attempt < total_attempts:
                    logging.warning(
                        f"[LLM] Transient error ({type(e).__name__}) on attempt "
                        f"{attempt}/{total_attempts}: {e}. Retrying..."
                    )
        logging.error(
            f"[LLM] Giving up after {total_attempts} attempts due to transient error: {last_err}"
        )
        raise last_err

    try:
        resp = _call(create_kwargs)
    except BadRequestError as e:
        err_text = str(e).lower()
        extra_body = create_kwargs.get("extra_body") or {}

        if "startswith" in err_text and not extra_body.get("chat_id"):
            logging.warning("[LLM] Retrying with chat_id (Open WebUI workaround)")
            retry_kwargs = dict(create_kwargs)
            retry_extra = dict(extra_body)
            retry_extra["chat_id"] = "navidj-api"
            retry_kwargs["extra_body"] = retry_extra
            resp = _call(retry_kwargs)
        elif LLM_MODE != "openai" and "response_format" in create_kwargs:
            logging.warning(f"[LLM] BadRequestError, retrying without response_format: {e}")
            retry_kwargs = dict(create_kwargs)
            retry_kwargs.pop("response_format", None)
            resp = _call(retry_kwargs)
        else:
            logging.error(f"[LLM] BadRequestError: {e}")
            raise
    except Exception as e:
        if LLM_MODE != "openai" and "response_format" in create_kwargs:
            logging.warning(f"[LLM] Request failed ({e}); retrying without response_format")
            retry_kwargs = dict(create_kwargs)
            retry_kwargs.pop("response_format", None)
            resp = _call(retry_kwargs)
        else:
            logging.error(f"[LLM] Request failed: {e}")
            raise

    if resp is None or not resp.choices:
        raise RuntimeError(
            f"LLM returned an empty response (model={LLM_MODEL}, mode={LLM_MODE}). "
            "Check that the model is loaded and the backend is reachable."
        )

    content = resp.choices[0].message.content or ""

    # reasoning_content is the extracted think block — only use it if content is empty
    reasoning = getattr(resp.choices[0].message, "reasoning_content", None) or ""
    if not content.strip() and reasoning:
        logging.debug(
            f"[LLM] content empty but reasoning_content present (len={len(reasoning)}); model may have only generated a think block"
        )
        # content stays empty, will trigger retry below

    # Debug: show raw content so we can see what the model actually returned
    logging.debug(f"[LLM RAW] len={len(content)} | first 300 chars: {content[:300]!r}")
    if not content.strip():
        logging.debug(
            f"[LLM] content empty but reasoning_content present (len={len(reasoning)}); model may have only generated a think block"
        )

    # --- Step 1: strip ALL think blocks FIRST before any further processing ---
    content = _remove_think_tags(content)

    # Retry once if the model returned empty content
    if not content.strip() and _retries > 0:
        print(f"[WARN] LLM returned empty content (model={LLM_MODEL}), retrying...")
        time.sleep(1)
        return _llm_chat(messages, _retries=_retries - 1, max_tokens=max_tokens)

    # --- Step 2: extract JSON object {...} buried in surrounding prose ---
    if "{" in content and "}" in content:
        start = content.find("{")
        end = content.rfind("}") + 1
        content = content[start:end]
    # --- Step 3: or extract JSON array [...] if no object found ---
    elif "[" in content and "]" in content:
        start = content.find("[")
        end = content.rfind("]") + 1
        content = content[start:end]

    return content.strip()


def _strip_fences(text: str) -> str:
    text = _remove_think_tags(text)
    if text.startswith("```"):
        _, rest = text.split("\n", 1)
        if rest.rstrip().endswith("```"):
            rest = rest.rstrip()[:-3]
        return rest.strip()
    return text


def _clean_json(text: str) -> str:
    """Attempt to fix common JSON errors from LLMs (trailing commas, etc)."""
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r",\s*([\]}])", r"\1", text)
    return text


def select_focus_metadata_single_call(
    prompt: str, all_artists: list[str], all_genres: list[str], all_albums: list[str]
) -> dict[str, list[str]]:
    """
    Single LLM call to select relevant artists, genres, and albums together.
    """
    system_msg = {
        "role": "system",
        "content": (
            "You are a music metadata curator. Select metadata for a vibe prompt.\n"
            "Rules:\n"
            "- Select ONLY items that exist exactly in the provided lists.\n"
            "- STRICTLY return ONLY valid JSON, no extra text, no markdown formatting:\n"
            '{"artists": [...], "genres": [...], "albums": [...]}\n'
            "- artists: exactly 5, genres: exactly 10, albums: exactly 5.\n"
            "- No preamble, no postamble, no explanation. Just the JSON."
        ),
    }

    user_msg = {
        "role": "user",
        "content": (
            f'Vibe prompt: "{prompt}"\n\n'
            f"Available artists ({len(all_artists)}): {all_artists}\n"
            f"Available genres ({len(all_genres)}): {all_genres}\n"
            f"Available albums ({len(all_albums)}): {all_albums}\n"
        ),
    }

    raw = _llm_chat([system_msg, user_msg], max_tokens=1200)
    raw = _clean_json(_strip_fences(raw))

    def _parse_metadata(text):
        parsed = json.loads(text)
        return {
            "artists": [x for x in parsed.get("artists", []) if x in all_artists],
            "genres": [x for x in parsed.get("genres", []) if x in all_genres],
            "albums": [x for x in parsed.get("albums", []) if x in all_albums],
        }

    try:
        return _parse_metadata(raw)
    except Exception as primary_err:
        # Backup: try to find any JSON object in the original raw response
        fallback_match = re.search(r"\{[\s\S]*\}", raw)
        if fallback_match:
            try:
                return _parse_metadata(_clean_json(fallback_match.group(0)))
            except Exception:
                pass
        print(
            f"[WARN] Metadata selection parse failed ({primary_err}); using top defaults."
        )
        return {
            "artists": all_artists[:10],
            "genres": all_genres[:15],
            "albums": all_albums[:5],
        }

def filter_library_by_metadata(
    explicit_artists: list[str],
    explicit_genres: list[str],
    explicit_albums: list[str],
    context_artists: list[str],
    context_genres: list[str],
    context_albums: list[str],
    selected_artists: list[str],
    selected_genres: list[str],
    selected_albums: list[str],
    all_songs: list[dict],
    context_song_ids: set[str] = None,
    semantic_song_ids: set[str] = None,
    era_range: tuple[int, int] = None,
) -> list[dict]:
    """
    Filter library based on combined focus items with hierarchical, additive weighting.
    Note that context playlist songs are used only as a vibe/metadata signal and are not scored for direct inclusion.

    `era_range`, when provided, is an inclusive (start_year, end_year) tuple
    detected from the prompt; it only ever adds a small bonus (see
    `_era_bonus`) and never filters or excludes songs.
    """
    # Create sets for efficient lookup
    exp_a = set(explicit_artists)
    exp_g = set(explicit_genres)
    exp_al = set(explicit_albums)

    ctx_a = set(context_artists)
    ctx_g = set(context_genres)
    ctx_al = set(context_albums)
    ctx_ids = context_song_ids or set()

    sel_a = set(selected_artists)
    sel_g = set(selected_genres)
    sel_al = set(selected_albums)

    sem_ids = semantic_song_ids or set()

    filtered = []
    for s in all_songs:
        song_artists = _split_artist_string(s.get("artist", ""))
        song_genres = _song_genres(s)
        song_album = s.get("album")
        song_id = s.get("id")

        score = 0.0

        # New Additive Weighting System:

        # 1. Explicit Matches (artist +4.5, genre/album +3.0 each)
        if any(a in exp_a for a in song_artists):
            score += 4.5
        if any(g in exp_g for g in song_genres):
            score += 3.0
        if song_album in exp_al:
            score += 4.5  # P0: explicit album must outrank context-in-playlist (+4.0)

        # 2. Context Playlist Matches
        # Explicitly in context playlist (+4.0)
        if song_id in ctx_ids:
            score += 4.0
        # Metadata matches if part of context
        if any(a in ctx_a for a in song_artists):
            score += 2.0
        if any(g in ctx_g for g in song_genres):
            score += 2.0
        if song_album in ctx_al:
            score += 2.0

        # 3. LLM Chosen Focus Matches
        if any(a in sel_a for a in song_artists):
            score += 1.5
        if any(g in sel_g for g in song_genres):
            score += 1.5
        if song_album in sel_al:
            score += 1.5
        # 4. Starred/Favorited Boost (+1.5)
        if s.get("starred"):
            score += 1.5

        # 5. Semantic Vibe Matches (+1.5)
        if song_id in sem_ids:
            score += 1.5

        # 6. Metadata-Based Score (play count, skip count, recency)
        score += calculate_metadata_score(s)

        # 7. Loose era bonus (never a filter; zero effect when no era detected)
        score += _era_bonus(s.get("releaseYear"), era_range)

        if score > 0:
            s_copy = s.copy()
            s_copy["_relevance_score"] = score
            filtered.append(s_copy)

    # Sort by relevance
    filtered.sort(key=lambda x: x.get("_relevance_score", 0), reverse=True)

    # Stratified tiered candidate pool
    tier1 = [s for s in filtered if s.get("_relevance_score", 0) >= 5.0][:250]
    tier2 = [s for s in filtered if 2.0 <= s.get("_relevance_score", 0) < 5.0][:200]
    tier3 = [s for s in filtered if 0 < s.get("_relevance_score", 0) < 2.0]
    random.shuffle(tier3)
    tier3 = tier3[:50]

    pool = tier1 + tier2 + tier3
    random.shuffle(pool)
    # P5: tier distribution is a first-class decision log (which songs the
    # picker can even see, and in what order, depends on these tiers).
    print(f"Tier distribution: tier1={len(tier1)} tier2={len(tier2)} tier3={len(tier3)} (pool={len(pool)})")

    if len(pool) < len(filtered):
        print(f"Stratified pool: {len(tier1)} high + {len(tier2)} medium + {len(tier3)} discovery = {len(pool)} songs.")

    return pool


def _trim_to_relevance(
    playlist: list[dict], filtered_songs: list[dict], min_songs: int
) -> list[dict]:
    """Keep the highest-relevance *min_songs* picks (no-op if already <= that).

    min_songs means exactly N, so overshoot is trimmed. Picks carry only
    {id, title}; relevance scores live on *filtered_songs*, so look them up by
    id (missing score -> 0, never crash).
    """
    if len(playlist) <= min_songs:
        return playlist
    score_by_id = {s["id"]: s.get("_relevance_score", 0) for s in filtered_songs}
    playlist.sort(key=lambda item: score_by_id.get(item["id"], 0), reverse=True)
    return playlist[:min_songs]


def generate_playlist_single_call(
    prompt: str,
    filtered_songs: list[dict],
    min_songs: int,
    explicit_artists: list[str] = None,
    explicit_albums: list[str] = None,
    explicit_genres: list[str] = None,
    mode: str = "default",
) -> list[dict]:
    """
    Generate playlist with a single LLM call.

    Uses compact numbered-line input and number-only JSON output to minimise
    LLM output tokens.  A Python-side numbering table maps the returned pick
    numbers back to full song dicts (id, title) so the LLM never has to
    reproduce IDs or exact titles.
    """
    # -- 1. Build local numbering table (never sent to LLM) ----------------
    num_to_song = {i + 1: s for i, s in enumerate(filtered_songs)}

    # -- 2. Build compact candidate lines for the LLM ---------------------
    lines = []
    for n, s in num_to_song.items():
        star = "*" if s.get("starred") else ""
        title   = s.get("title")   or "Unknown"
        artist  = s.get("artist")  or "Unknown"
        genre   = ", ".join(_song_genres(s)) or "Unknown"
        album   = s.get("album")   or "Unknown"
        year    = s.get("releaseYear") or "Unknown"
        lines.append(f"{n}{star}|{title}|{artist}|{genre}|{album}|{year}")
    candidate_text = "\n".join(lines)

    # -- 3. System prompt --------------------------------------------------
    # P2: the picker previously only ever saw explicit artists; albums and
    # genres affected pool weighting but were invisible to the LLM, so it had
    # no way to connect a vibe phrase to the matching rows in the list.
    explicit_bits = []
    if explicit_artists:
        explicit_bits.append(f"artists: {', '.join(explicit_artists)}")
    if explicit_albums:
        explicit_bits.append(f"albums: {', '.join(explicit_albums)}")
    if explicit_genres:
        explicit_bits.append(f"genres: {', '.join(explicit_genres)}")
    if mode == "free":
        explicit_line = (
            f"- The user specifically requested: {'; '.join(explicit_bits)}. "
            "Strongly prefer songs matching this request and do NOT enforce "
            "artist diversity against them.\n"
        )
    elif mode == "anchor" and explicit_bits:
        explicit_line = (
            f"- The user specifically requested: {'; '.join(explicit_bits)}. "
            "Include a meaningful number of songs matching this request, but "
            "the rest of the playlist should follow the stated vibe; keep the "
            "requested artists well under half of the playlist.\n"
        )
    elif explicit_bits:
        explicit_line = (
            f"- The user specifically requested: {'; '.join(explicit_bits)}. "
            "Strongly prefer songs matching this request, but still keep "
            "artist diversity.\n"
        )
    else:
        explicit_line = "- Ensure artist diversity; avoid over-representing any single artist.\n"

    system_msg = {
        "role": "system",
        "content": (
            "You are a playlist-builder AI.\n"
            "You will receive a numbered list of candidate songs. "
            "A number followed by * means the user has starred/favorited that song.\n"
            "Rules:\n"
            f"- Select exactly {min_songs} songs.\n"
            "- Pick ONLY numbers that appear in the provided list.\n"
            "- Maximize artist and album diversity.\n"
            "- Slightly prefer starred songs if they fit.\n"
            f"{explicit_line}"
            "- STRICTLY return ONLY a JSON object with a single key \"picks\" whose value "
            "is an array of the selected candidate numbers (integers). Example:\n"
            '  {"picks": [3, 12, 44, 1, 27]}\n'
            "- Numbers only. No titles, no extra keys, no explanation, no markdown fences."
        ),
    }

    # -- 4. User message ----------------------------------------------------
    user_msg = {
        "role": "user",
        "content": f"Vibe: {prompt}\n\nCandidates:\n{candidate_text}",
    }

    call_max_tokens = None if LLM_MODE != "openai" else min_songs * 24 + 100
    raw = _strip_fences(_llm_chat([system_msg, user_msg], max_tokens=call_max_tokens))

    # -- 5. Post-process: map picks back to song dicts ---------------------
    def _resolve_picks(picks):
        """Convert a list of raw pick values to playlist dicts, skipping invalid entries."""
        playlist = []
        seen_ids = set()
        for p in picks:
            try:
                num = int(p)
            except (TypeError, ValueError):
                continue
            if num in num_to_song and num_to_song[num]["id"] not in seen_ids:
                song = num_to_song[num]
                playlist.append({"id": song["id"], "title": song["title"]})
                seen_ids.add(song["id"])
        return playlist

    picks = []
    try:
        parsed = json.loads(raw)
        picks = (
            parsed.get("picks", [])
            if isinstance(parsed, dict)
            else (parsed if isinstance(parsed, list) else [])
        )
    except Exception:
        print(f"[DEBUG] raw response (first 500 chars): {raw[:500]!r}")
        # --- Backup: regex-extract all integers from the raw response ---
        fallback_raw = _remove_think_tags(raw)
        extracted = re.findall(r"\b(\d+)\b", fallback_raw)
        if extracted:
            print(
                f"[WARN] JSON parse failed for playlist; extracted {len(extracted)} integers via regex fallback."
            )
            picks = extracted
        else:
            print(
                f"[ERROR] Playlist generation failed: could not parse response. Raw: {raw[:150]}"
            )

    playlist = _resolve_picks(picks)

    # Pad if needed
    if len(playlist) < min_songs:
        playlist = ensure_min_songs(
            playlist,
            filtered_songs,
            min_songs,
            explicit_artists=explicit_artists,
            mode=mode,
        )

    # Trim if the LLM overshot: min_songs means exactly N, keep highest-relevance.
    playlist = _trim_to_relevance(playlist, filtered_songs, min_songs)

    return playlist


def generate_playlist_chunked(
    prompt: str,
    filtered_songs: list[dict],
    min_songs: int,
    chunk_size: int = 200,
    explicit_artists: list[str] = None,
    explicit_albums: list[str] = None,
    explicit_genres: list[str] = None,
    mode: str = "default",
) -> list[dict]:
    """
    Generate playlist using chunked processing for large candidate pools.

    Args:
        prompt: User's vibe prompt
        filtered_songs: Pre-filtered candidate songs
        min_songs: Minimum number of songs to include
        chunk_size: Number of songs to process per chunk (default: 200)
        explicit_artists: Artists explicitly mentioned in prompt

    Returns:
        List of playlist items: [{"id": "...", "title": "..."}, ...]
    """
    if len(filtered_songs) <= chunk_size:
        # If candidates fit in one chunk, use single-call generation
        return generate_playlist_single_call(
            prompt=prompt,
            filtered_songs=filtered_songs,
            min_songs=min_songs,
            explicit_artists=explicit_artists,
            explicit_albums=explicit_albums,
            explicit_genres=explicit_genres,
            mode=mode,
        )

    # Split into chunks, maintaining score order (already sorted)
    num_chunks = math.ceil(len(filtered_songs) / chunk_size)
    print(
        f"Processing {len(filtered_songs)} songs in {num_chunks} chunks of {chunk_size}"
    )

    all_selections = []
    songs_per_chunk = math.ceil(min_songs / num_chunks)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(filtered_songs))
        chunk = filtered_songs[start_idx:end_idx]
        random.shuffle(chunk)

        print(f"Chunk {i + 1}/{num_chunks}: {len(chunk)} songs")

        # Request proportional number of songs from this chunk
        chunk_target = min(songs_per_chunk, len(chunk))

        try:
            chunk_playlist = generate_playlist_single_call(
                prompt=prompt,
                filtered_songs=chunk,
                min_songs=chunk_target,
                explicit_artists=explicit_artists,
                explicit_albums=explicit_albums,
                explicit_genres=explicit_genres,
                mode=mode,
            )
            all_selections.extend(chunk_playlist)
        except Exception as e:
            print(f"[WARN] Chunk {i + 1} failed: {e}")
            # Fallback: take the highest-relevance songs from this chunk
            # (chunk was shuffled above), with minor tie-breaking randomness.
            fallback_pool = sorted(
                chunk,
                key=lambda s: (s.get("_relevance_score", 0), random.random()),
                reverse=True,
            )
            fallback = [
                {"id": s["id"], "title": s["title"]}
                for s in fallback_pool[:chunk_target]
            ]
            all_selections.extend(fallback)

    # Deduplicate and trim to requested size
    seen_ids = set()
    unique_selections = []
    for item in all_selections:
        if item["id"] not in seen_ids:
            seen_ids.add(item["id"])
            unique_selections.append(item)

    # If we don't have enough, pad with top-scored unused songs
    if len(unique_selections) < min_songs:
        unique_selections = ensure_min_songs(
            unique_selections,
            filtered_songs,
            min_songs,
            explicit_artists=explicit_artists,
            mode=mode,
        )

    # Trim to exactly min_songs by relevance (not positional), matching the
    # single-call path. Only bites when a small --chunk_size overshoots.
    return _trim_to_relevance(unique_selections, filtered_songs, min_songs)


# --------------------------------------------------
# PLAYLIST PUSH/UPDATE HELPERS
# --------------------------------------------------


def _update_playlist_on_server(
    name: str, song_ids: list[str], description: str
) -> bool:
    pl_resp = requests.get(
        f"{SUBSONIC_BASE_URL}/getPlaylists", params=SUBSONIC_AUTH_PARAMS
    )
    pl_resp.raise_for_status()
    root = ET.fromstring(pl_resp.content)
    ns = root.tag.split("}")[0] + "}"
    plid = next(
        (
            pl.get("id")
            for pl in root.findall(f".//{ns}playlist")
            if pl.get("name") == name
        ),
        None,
    )

    if plid:
        upd = requests.get(
            f"{SUBSONIC_BASE_URL}/createPlaylist",
            params={**SUBSONIC_AUTH_PARAMS, "playlistId": plid, "songId": song_ids},
        )
        if upd.status_code != 200:
            print("ERROR: Failed to update existing playlist tracks.")
            return False
    else:
        upd = requests.get(
            f"{SUBSONIC_BASE_URL}/createPlaylist",
            params={**SUBSONIC_AUTH_PARAMS, "name": name, "songId": song_ids},
        )
        if upd.status_code != 200:
            print("ERROR: Failed to create playlist.")
            return False
        root = ET.fromstring(upd.content)
        plid = root.find(f".//{ns}playlist").get("id")

    desc_upd = requests.get(
        f"{SUBSONIC_BASE_URL}/updatePlaylist",
        params={**SUBSONIC_AUTH_PARAMS, "playlistId": plid, "comment": description},
    )
    if desc_upd.status_code != 200:
        print("ERROR: Tracks updated but could not set description.")

    return True


# --------------------------------------------------
# SUBSONIC BROWSING HELPERS
# --------------------------------------------------
def fetch_all_artists() -> list[str]:
    """Return every artist name known to the server (1 per entry)."""
    r = requests.get(
        f"{SUBSONIC_BASE_URL}/getArtists.view",
        params={**SUBSONIC_AUTH_PARAMS, "f": "json"},
        timeout=60,
    )
    r.raise_for_status()
    idx = r.json()["subsonic-response"]["artists"]["index"]

    # Collect all artist names and split compound artists
    all_artists = []
    for letter in idx:
        for art in letter.get("artist", []):
            if art.get("name"):
                individual_artists = _split_artist_string(art["name"])
                all_artists.extend(individual_artists)

    # Remove duplicates and sort
    return sorted(list(dict.fromkeys(all_artists)))


def fetch_all_genres() -> list[str]:
    """Return every genre name known to the server."""
    r = requests.get(
        f"{SUBSONIC_BASE_URL}/getGenres.view",
        params={**SUBSONIC_AUTH_PARAMS, "f": "json"},
        timeout=60,
    )
    r.raise_for_status()
    genres = r.json()["subsonic-response"]["genres"]["genre"]
    return sorted(g["value"] for g in genres if g.get("value"))


def fetch_all_playlists(exclude_name: str = None) -> list[dict]:
    """Return all playlists from the server with their details, excluding specified playlists."""
    r = requests.get(
        f"{SUBSONIC_BASE_URL}/getPlaylists",
        params={**SUBSONIC_AUTH_PARAMS, "f": "json"},
        timeout=60,
    )
    r.raise_for_status()
    playlists = r.json()["subsonic-response"]["playlists"]["playlist"]
    all_playlists: list[dict] = []
    for pl in playlists:
        pl_id, pl_name = pl["id"], pl["name"]

        # Skip Daily Mix playlists (case-insensitive)
        if pl_name.lower().startswith("daily mix"):
            continue

        # Skip the target playlist to prevent circular references
        if exclude_name and pl_name.lower() == exclude_name.lower():
            continue

        pl_songs = fetch_playlist_songs(pl_id)
        all_playlists.append({"id": pl_id, "name": pl_name, "songs": pl_songs})
    return all_playlists


def fetch_playlist_songs(playlist_id: str) -> list[dict]:
    r = requests.get(
        f"{SUBSONIC_BASE_URL}/getPlaylist",
        params={**SUBSONIC_AUTH_PARAMS, "id": playlist_id, "f": "json"},
        timeout=60,
    )
    r.raise_for_status()
    pl = r.json()["subsonic-response"]["playlist"]
    tracks = pl.get("entry") or pl.get("song") or []
    return [{"id": t["id"], "title": t["title"], "artist": t["artist"]} for t in tracks]


# --------------------------------------------------
# CONTEXT PLAYLIST SELECTION
# --------------------------------------------------

def select_context_playlist_songs(
    prompt: str,
    existing_playlists: list[dict],
    all_songs: list[dict],
    embedding_manager=None,
) -> list[dict]:
    """
    Select ONE playlist for context. If embedding_manager is provided, uses semantic similarity against
    playlist composition string. Otherwise asks the LLM to pick ONE playlist by name.
    """
    if not existing_playlists:
        return []

    if embedding_manager is not None:
        descriptions = []
        for pl in existing_playlists:
            name = pl.get("name", "")
            description = pl.get("comment", "")
            artist_counts = Counter()
            genre_counts = Counter()
            for song in pl.get("songs", []):
                if song.get("artist"):
                    for a in _split_artist_string(song["artist"]):
                        artist_counts[a] += 1
                if song.get("genre"):
                    genre_counts[song["genre"]] += 1
            top_artists = [a for a, _ in artist_counts.most_common(8)]
            top_genres = [g for g, _ in genre_counts.most_common(4)]
            desc_str = f"{name} | {description} | {', '.join(top_artists)} | {', '.join(top_genres)}"
            descriptions.append(desc_str)
            
        prompt_emb = embedding_manager.get_embedding(prompt)
        pl_embs = embedding_manager.get_embeddings_batch(descriptions)
        
        best_score = -2.0
        best_idx = -1
        for i, emb in enumerate(pl_embs):
            if emb is not None and prompt_emb is not None:
                score = embedding_manager._cosine_similarity(prompt_emb, emb)
                if score > best_score:
                    best_score = score
                    best_idx = i
                    
        if best_score < 0.15 or best_idx == -1:
            print("No context playlist deemed semantically relevant enough (score < 0.15).")
            return []
            
        best_pl = existing_playlists[best_idx]
        print(f"Using context playlist '{best_pl['name']}' (semantic score: {best_score:.3f}).")
        
        id_map = {s["id"]: s for s in all_songs}
        return [id_map[s["id"]] for s in best_pl["songs"] if s["id"] in id_map]

    playlist_names = [pl["name"] for pl in existing_playlists]
    playlists_json = json.dumps(playlist_names).replace("```", "`\u200c`\u200c`")
    sys_msg = {
        "role": "system",
        "content": (
            "You are a playlist-builder AI.\n"
            "Pick the single most relevant playlist for the given vibe.\n"
            "STRICTLY return ONLY valid JSON: {\"playlist_name\": \"Name\"} or {} if none fit.\n"
            "No preamble, no postamble, no markdown formatting."
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Vibe prompt: {prompt}\n\nAvailable playlist names: {playlists_json}"
        ),
    }
    raw = _strip_fences(_llm_chat([sys_msg, user_msg], max_tokens=400))
    playlist_name = ""
    try:
        parsed = json.loads(raw)
        playlist_name = parsed.get("playlist_name", "").strip()
    except Exception as primary_err:
        # Backup: look for a quoted string after 'playlist_name'
        m = re.search(r'playlist_name["\s:]+(["\'])(.+?)\1', raw)
        if m:
            playlist_name = m.group(2).strip()
            print(f"[WARN] Context playlist JSON parse failed; extracted name via regex: '{playlist_name}'")
        else:
            print(f"[WARN] Context playlist JSON parse failed ({primary_err}). Raw: {raw[:120]}")
    if not playlist_name:
        return []

    target = playlist_name.lower()
    pl = next((pl for pl in existing_playlists if pl["name"].lower() == target), None)
    if not pl:
        print(f"Context playlist '{playlist_name}' not found among existing playlists.")
        return []
    print(f"Using context playlist '{playlist_name}' for focus selection.")
    id_map = {s["id"]: s for s in all_songs}
    songs = [id_map[s["id"]] for s in pl["songs"] if s["id"] in id_map]
    return songs


def _context_overlaps_explicit(
    context_songs: list[dict],
    explicit_artists: list[str] = None,
    explicit_albums: list[str] = None,
) -> bool:
    """True if any explicitly requested artist or album appears among the
    context songs (P0 entity-aware context gate).

    Splits multi-artist strings the same way scoring does
    (`_split_artist_string`) so a "A & B" credit counts as both A and B;
    albums compare case-insensitively on the exact album string.
    """
    wanted_artists = {a.lower() for a in (explicit_artists or [])}
    wanted_albums = {a.lower() for a in (explicit_albums or [])}
    for s in context_songs:
        if wanted_albums and (s.get("album") or "").lower() in wanted_albums:
            return True
        if wanted_artists and any(
            a.lower() in wanted_artists
            for a in _split_artist_string(s.get("artist", ""))
        ):
            return True
    return False


# --------------------------------------------------
# PLAYLIST LENGTH ENFORCEMENT
# --------------------------------------------------


def ensure_min_songs(
    playlist: list[dict],
    candidates: list[dict],
    min_songs: int,
    max_songs: int = 50,
    explicit_artists: list[str] = None,
    mode: str = "default",
) -> list[dict]:
    """
    Pad *playlist* up to *min_songs* using the highest-relevance unused
    candidates (descending `_relevance_score`, with only minor tie-breaking
    randomness). Padding respects the variant-dedup rules and a mode-aware
    per-artist cap (P3: free = no cap, anchor = relaxed cap for named
    artists, default = adaptive cap for everyone).
    """
    if len(playlist) >= min_songs:
        return playlist[:max_songs]
    needed = min(min_songs - len(playlist), max_songs - len(playlist))

    id_lookup = {s["id"]: s for s in candidates}
    existing_ids = {p["id"] for p in playlist}

    # Seed dedup keys and per-artist counts from the current playlist.
    existing_keys: set[tuple] = set()
    artist_counts: Counter = Counter()
    for p in playlist:
        full = id_lookup.get(p["id"])
        if full:
            existing_keys.add(_variant_dedup_key(full))
            for n in [
                x.lower() for x in _split_artist_string(full.get("artist") or "")
            ] or [""]:
                artist_counts[n] += 1

    if mode == "free":
        cap = None
    elif mode == "anchor" and explicit_artists:
        cap = max(2, math.ceil(_ANCHOR_CAP_FRAC * min_songs))
    else:
        cap = max(2, math.ceil(_ADAPTIVE_CAP_FRAC * min_songs))

    remaining = [s for s in candidates if s["id"] not in existing_ids]
    # Descending relevance, minor tie-breaking randomness only.
    remaining.sort(
        key=lambda s: (s.get("_relevance_score", 0), random.random()), reverse=True
    )

    added = 0
    for s in remaining:
        if added >= needed:
            break
        k = _variant_dedup_key(s)
        if k in existing_keys:
            continue
        names = [n.lower() for n in _split_artist_string(s.get("artist") or "")]
        if cap is not None and any(artist_counts[n] >= cap for n in names or [""]):
            continue
        playlist.append({"id": s["id"], "title": s.get("title")})
        existing_keys.add(k)
        for n in names or [""]:
            artist_counts[n] += 1
        added += 1

    print(
        f"Added {added} relevance-ranked songs from filtered options to reach minimum length of {min_songs}."
    )
    return playlist[:max_songs]


def _dedup_playlist_variants(
    playlist: list[dict], id_lookup: dict[str, dict]
) -> list[dict]:
    """
    Final safety pass: collapse any variant duplicates in an assembled playlist.
    Entries are enriched from *id_lookup* (id -> full song dict) so the variant
    key has artist/starred/playCount data; entries not found are kept as-is.
    """
    enriched = [id_lookup.get(item["id"], item) for item in playlist]
    deduped = _dedup_variants(enriched)
    return [{"id": s["id"], "title": s.get("title")} for s in deduped]


_ADAPTIVE_CAP_FRAC = 0.15  # per-artist cap, default mode (P3)
_ANCHOR_CAP_FRAC = 0.45    # relaxed per-artist cap for named artists, anchor mode (P3)
_ALBUM_CAP_FRAC = 0.30     # per-album cap (P6)


def _enforce_artist_cap(
    playlist: list[dict],
    candidates: list[dict],
    target_size: int,
    explicit_artists: list[str] = None,
    mode: str = "default",
) -> list[dict]:
    """
    Mode-aware post-selection per-artist cap (P3, replaces the old binary
    'explicit artists => never cap' rule).

    - mode "free": entity-dominant prompt that names artists ("Michael
      Jackson", "just Radiohead and Portishead") - no cap, as before.
    - mode "anchor": named (anchor) artists get the relaxed anchor cap; every
      other artist keeps the adaptive base cap.
    - mode "default": adaptive cap for everyone.

    Every credited artist in a multi-artist string counts (not just the
    primary), matching how run history reports artist percentages. Over-cap
    picks (keeping the highest-relevance ones per artist) are replaced by the
    highest-relevance unused candidates, respecting the cap and dedup rules.
    """
    if mode == "free":
        return playlist

    base_cap = max(2, math.ceil(_ADAPTIVE_CAP_FRAC * target_size))
    anchor_cap = max(2, math.ceil(_ANCHOR_CAP_FRAC * target_size))
    anchor_set = {a.lower() for a in (explicit_artists or [])}

    def cap_for(artist_lc: str) -> int:
        if mode == "anchor" and artist_lc in anchor_set:
            return anchor_cap
        return base_cap

    id_lookup = {s["id"]: s for s in candidates}

    def relscore(item: dict) -> float:
        return id_lookup.get(item.get("id"), {}).get("_relevance_score", 0)

    def artist_names(song_or_item: dict) -> list[str]:
        full = id_lookup.get(song_or_item.get("id"), song_or_item)
        return [n.lower() for n in _split_artist_string(full.get("artist") or "")]

    # Decide which picks to keep per artist: highest relevance up to that
    # artist's cap. A multi-credit song counts against every artist named.
    from collections import defaultdict

    by_artist: dict[str, list[dict]] = defaultdict(list)
    for item in playlist:
        for n in artist_names(item) or [""]:
            by_artist[n].append(item)

    keep_ids: set[str] = set()
    for artist, items in by_artist.items():
        items_sorted = sorted(items, key=relscore, reverse=True)
        for it in items_sorted[: cap_for(artist)]:
            keep_ids.add(it["id"])

    kept: list[dict] = []
    kept_ids: set[str] = set()
    artist_counts: Counter = Counter()
    existing_keys: set[tuple] = set()
    removed = 0
    for item in playlist:  # preserve original order for the retained picks
        if item["id"] in keep_ids:
            kept.append(item)
            kept_ids.add(item["id"])
            for n in artist_names(item) or [""]:
                artist_counts[n] += 1
            full = id_lookup.get(item["id"])
            if full:
                existing_keys.add(_variant_dedup_key(full))
        else:
            removed += 1

    # Hard-cap pass: keep_ids is a UNION of per-artist top-N, so one artist's
    # keep budget can rescue a track that over-credits another artist (a
    # "Logic • Pusha T" pick kept for Pusha T still counts against Logic's
    # anchor cap). Enforce each artist's cap as a hard ceiling on total
    # credited tracks by evicting the lowest-relevance over-cap picks.
    def _credit_counts(items: list[dict]) -> Counter:
        cc: Counter = Counter()
        for it in items:
            for n in artist_names(it) or [""]:
                cc[n] += 1
        return cc

    evicted = 0
    while True:
        counts = _credit_counts(kept)
        over = {a for a, c in counts.items() if c > cap_for(a)}
        if not over:
            break
        victims = [
            it for it in kept if over.intersection(artist_names(it) or [""])
        ]
        if not victims:
            break
        victim = min(victims, key=lambda it: relscore(it))
        kept.remove(victim)
        kept_ids.discard(victim["id"])
        full = id_lookup.get(victim["id"])
        if full:
            existing_keys.discard(_variant_dedup_key(full))
        evicted += 1
    artist_counts = _credit_counts(kept)

    if not (removed or evicted):
        return kept

    # Backfill removed slots with the best-scoring alternatives from other
    # artists, respecting the cap and dedup rules.
    target_len = len(playlist)
    pool = sorted(
        candidates,
        key=lambda s: (s.get("_relevance_score", 0), random.random()),
        reverse=True,
    )
    for s in pool:
        if len(kept) >= target_len:
            break
        if s["id"] in kept_ids:
            continue
        names = artist_names(s)
        if any(artist_counts[n] >= cap_for(n) for n in names or [""]):
            continue
        k = _variant_dedup_key(s)
        if k in existing_keys:
            continue
        kept.append({"id": s["id"], "title": s.get("title")})
        kept_ids.add(s["id"])
        for n in names or [""]:
            artist_counts[n] += 1
        existing_keys.add(k)

    print(
        f"Artist cap (mode={mode}, base={base_cap}, anchor={anchor_cap}) "
        f"enforced: replaced {removed + evicted} over-cap pick(s)."
    )
    return kept


def _enforce_album_cap(
    playlist: list[dict],
    candidates: list[dict],
    target_size: int,
    explicit_albums: list[str] = None,
) -> list[dict]:
    """
    Per-album cap (P6): at most max(2, ceil(_ALBUM_CAP_FRAC * target_size))
    tracks from a single album, unless that album was explicitly named in the
    prompt (a deliberate whole-album request is exempt - that's the point of
    naming the album). Over-cap picks (keeping the highest-relevance ones per
    album) are replaced by the highest-relevance unused candidates from other
    non-exempt albums, respecting the variant-dedup rules.
    """
    from collections import defaultdict

    cap = max(2, math.ceil(_ALBUM_CAP_FRAC * target_size))
    exempt = {a.lower() for a in (explicit_albums or [])}
    id_lookup = {s["id"]: s for s in candidates}

    def album_of(item: dict) -> str:
        full = id_lookup.get(item.get("id"), item)
        return (full.get("album") or "").lower()

    def relscore(item: dict) -> float:
        return id_lookup.get(item.get("id"), {}).get("_relevance_score", 0)

    by_album: dict[str, list[dict]] = defaultdict(list)
    for item in playlist:
        by_album[album_of(item)].append(item)

    keep_ids: set[str] = set()
    for album, items in by_album.items():
        if album in exempt or len(items) <= cap:
            keep_ids.update(it["id"] for it in items)
            continue
        items_sorted = sorted(items, key=relscore, reverse=True)
        keep_ids.update(it["id"] for it in items_sorted[:cap])

    kept: list[dict] = []
    kept_ids: set[str] = set()
    album_counts: Counter = Counter()
    existing_keys: set[tuple] = set()
    removed = 0
    for item in playlist:  # preserve original order for the retained picks
        if item["id"] in keep_ids:
            kept.append(item)
            kept_ids.add(item["id"])
            album_counts[album_of(item)] += 1
            full = id_lookup.get(item["id"])
            if full:
                existing_keys.add(_variant_dedup_key(full))
        else:
            removed += 1

    if removed == 0:
        return kept

    # Backfill removed slots with the best-scoring alternatives from other
    # non-exempt albums, respecting the cap and dedup rules.
    target_len = len(playlist)
    pool = sorted(
        candidates,
        key=lambda s: (s.get("_relevance_score", 0), random.random()),
        reverse=True,
    )
    for s in pool:
        if len(kept) >= target_len:
            break
        if s["id"] in kept_ids:
            continue
        al = (s.get("album") or "").lower()
        if al in exempt or album_counts[al] >= cap:
            continue
        k = _variant_dedup_key(s)
        if k in existing_keys:
            continue
        kept.append({"id": s["id"], "title": s.get("title")})
        kept_ids.add(s["id"])
        album_counts[al] += 1
        existing_keys.add(k)

    print(
        f"Album cap ({cap}/album, exempt={sorted(exempt) or 'none'}) enforced: "
        f"replaced {removed} over-cap pick(s)."
    )
    return kept


def _guarantee_explicit(
    playlist: list[dict],
    candidates: list[dict],
    explicit_artists: list[str] = None,
    explicit_albums: list[str] = None,
    explicit_tracks: list[dict] = None,
) -> list[dict]:
    """
    P3 guaranteed inclusion. Runs AFTER the caps so the user's actual request
    can't be evicted by them:
    - every explicitly named track appears at least once,
    - every explicitly named album has >=3 tracks in the mix,
    - every explicitly named artist has >=3 tracks in the mix.
    Missing entity tracks are swapped in for the lowest-relevance non-protected
    picks (protected = matches any explicit entity), variant-dedup respected.
    """
    explicit_artists = explicit_artists or []
    explicit_albums = explicit_albums or []
    explicit_tracks = explicit_tracks or []
    if not (explicit_artists or explicit_albums or explicit_tracks):
        return playlist

    id_lookup = {s["id"]: s for s in candidates}
    explicit_album_set = {a.lower() for a in explicit_albums}
    explicit_artist_set = {a.lower() for a in explicit_artists}
    explicit_title_set = {
        (t.get("title") or "").lower() for t in explicit_tracks if t.get("title")
    }

    def is_protected(song: dict) -> bool:
        if not song:
            return False
        if (song.get("album") or "").lower() in explicit_album_set:
            return True
        if (song.get("title") or "").lower() in explicit_title_set:
            return True
        return any(
            a.lower() in explicit_artist_set
            for a in _split_artist_string(song.get("artist") or "")
        )

    have_titles: set[str] = set()
    album_counts: Counter = Counter()
    artist_counts: Counter = Counter()
    for item in playlist:
        full = id_lookup.get(item["id"])
        if not full:
            continue
        have_titles.add((full.get("title") or "").lower())
        album_counts[(full.get("album") or "").lower()] += 1
        for n in _split_artist_string(full.get("artist") or ""):
            artist_counts[n.lower()] += 1

    # Build the set of songs still needed to satisfy the explicit request.
    wanted: list[dict] = []
    for t in explicit_tracks:
        if (t.get("title") or "").lower() not in have_titles:
            wanted.append(t)
    for al in explicit_albums:
        need = max(0, 3 - album_counts.get(al.lower(), 0))
        if need:
            cands = sorted(
                (s for s in candidates if (s.get("album") or "").lower() == al.lower()),
                key=lambda s: s.get("_relevance_score", 0),
                reverse=True,
            )
            wanted.extend(cands[:need])
    for ar in explicit_artists:
        arl = ar.lower()
        need = max(0, 3 - artist_counts.get(arl, 0))
        if need:
            cands = sorted(
                (
                    s
                    for s in candidates
                    if any(
                        n.lower() == arl
                        for n in _split_artist_string(s.get("artist") or "")
                    )
                ),
                key=lambda s: s.get("_relevance_score", 0),
                reverse=True,
            )
            wanted.extend(cands[:need])

    if not wanted:
        return playlist

    existing_ids = {item["id"] for item in playlist}
    existing_keys: set[tuple] = set()
    for item in playlist:
        full = id_lookup.get(item["id"])
        if full:
            existing_keys.add(_variant_dedup_key(full))

    # Evict lowest-relevance unprotected picks first; protected picks stay.
    evictable = [
        item for item in playlist if not is_protected(id_lookup.get(item["id"]))
    ]
    evictable.sort(
        key=lambda item: id_lookup.get(item["id"], {}).get("_relevance_score", 0)
    )

    added = 0
    for cand in wanted:
        if not evictable:
            break
        if cand["id"] in existing_ids:
            continue
        k = _variant_dedup_key(cand)
        if k in existing_keys:
            continue
        victim = evictable.pop(0)
        playlist = [item for item in playlist if item["id"] != victim["id"]]
        playlist.append({"id": cand["id"], "title": cand.get("title")})
        existing_ids.add(cand["id"])
        existing_keys.add(k)
        added += 1

    if added:
        print(f"Guaranteed inclusion: swapped in {added} explicit request track(s).")
    return playlist


# --------------------------------------------------
# PLAYLIST ENTRY SANITISER
# --------------------------------------------------


def _sanitize_playlist(
    entries: List[dict], candidates: List[dict]
) -> List[dict]:
    """
    Ensure each playlist entry has an 'id'. If an entry only has a 'title'
    (and optionally 'artist'), try to resolve the matching song in *candidates*
    via a case-insensitive title-and-artist match. Drop any rows we can't
    resolve. This is backend-agnostic and therefore safe for both Ollama and
    OpenAI modes.
    """
    id_by_pair = {
        (s["title"].lower(), (s.get("artist") or "").lower()): s["id"]
        for s in candidates
        if s.get("id") and s.get("title")
    }
    cleaned: List[dict] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        if "id" in e and any(s["id"] == e["id"] for s in candidates):
            cleaned.append(e)
            continue
        key = (e.get("title", "").lower(), e.get("artist", "").lower())
        resolved = id_by_pair.get(key)
        if resolved:
            cleaned.append({"id": resolved, "title": e.get("title")})
    return cleaned


# --------------------------------------------------
# PROMPT ENTITY EXTRACTION
# --------------------------------------------------

STOPWORDS = {
    "and",
    "but",
    "mix",
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "to",
    "from",
    "by",
    "with",
    "at",
    "as",
    "is",
    "it",
    "or",
    "vs",
    "feat",
    "featuring",
}


def extract_prompt_entities(
    prompt: str, all_artists: list[str], all_genres: list[str], all_albums: list[str]
) -> dict:
    """
    Extract artists, genres, and albums mentioned in the prompt using smart partial matching.
    Handles partial names like 'gambino' -> 'Childish Gambino'.
    Returns a dict with keys: 'artists', 'genres', 'albums'.
    """
    entities = {"artists": [], "genres": [], "albums": []}

    # Clean prompt: remove stopwords and punctuation, split into words
    prompt_lc = prompt.lower()
    prompt_words = set(re.findall(r"\b\w+\b", prompt_lc)) - STOPWORDS
    # Helper to check matches in prompt
    def check_matches(items, key):
        for item in items:
            item_lc = item.lower()
            item_words = set(re.findall(r"\b\w+\b", item_lc)) - STOPWORDS
            # Exact match (full item name in prompt). Word-bounded so a short
            # title can't match as a fragment of unrelated words ("on" in
            # "one more song" must not resolve an album literally titled "On").
            if re.search(rf"\b{re.escape(item_lc)}\b", prompt_lc):
                entities[key].append(item)
            # Partial match (ALL significant words from item must be in prompt)
            elif len(item_words) >= 2 and item_words.issubset(prompt_words):
                entities[key].append(item)

    # Artists: check both exact and partial matches
    check_matches(all_artists, "artists")

    # Fuzzy fallback: only if the exact/partial pass found no artist at all.
    # Catches typo'd prompts ("micheal jackson" -> "Michael Jackson") so the
    # downstream artist-preference system still engages instead of silently
    # doing nothing. Accept an artist only if EVERY significant word has a
    # close prompt match (mirrors the all-words partial-match rule above).
    # ponytail: multi-word artists only, to avoid single-word fuzzy false
    # positives ("prince"); add a per-artist alias table if this is too coarse.
    if not entities["artists"]:
        for item in all_artists:
            item_words = set(re.findall(r"\b\w+\b", item.lower())) - STOPWORDS
            if len(item_words) < 2:
                continue
            if all(
                difflib.get_close_matches(w, prompt_words, n=1, cutoff=0.8)
                for w in item_words
            ):
                entities["artists"].append(item)
                print(f"Fuzzy-matched artist: '{item}' (approximate prompt match)")

    # P1 initials expansion: "mj" -> "Michael Jackson". The exact/partial/fuzzy
    # passes can never connect initials to full names, so letter tokens of 1-3
    # chars are tested against each artist's initials (first letter of every
    # significant word). Accepted only when the initials are UNIQUE in the
    # library, so ambiguous tokens never resolve. Matched tokens are recorded
    # so mode classification (P3) doesn't count them as vibe content.
    letter_tokens = {w for w in prompt_words if 1 <= len(w) <= 3 and w.isalpha()}
    if letter_tokens:
        initials_map: dict[str, set[str]] = {}
        for item in all_artists:
            init_words = [
                w for w in re.findall(r"\b\w+\b", item.lower()) if w not in STOPWORDS
            ]
            init = "".join(w[0] for w in init_words)
            if 2 <= len(init) <= 4:
                initials_map.setdefault(init, set()).add(item)
        for tok in sorted(letter_tokens):
            matches = initials_map.get(tok)
            if matches and len(matches) == 1:
                artist = next(iter(matches))
                if artist not in entities["artists"]:
                    entities["artists"].append(artist)
                    entities.setdefault("initials_tokens", set()).add(tok)
                    print(f"Initials-matched artist: '{tok}' -> {artist}")

    # Albums: check both exact and partial matches
    check_matches(all_albums, "albums")

    # Genres: exact match only (genres are typically single words or short phrases)
    for genre in all_genres:
        if genre.lower() in prompt_lc:
            entities["genres"].append(genre)

    return entities


# --------------------------------------------------
# PROMPT TRACK EXTRACTION + MODE CLASSIFICATION (P3)
# --------------------------------------------------

_DOMINANCE_IGNORE = {
    # Functional/quantifier/positional words that carry no vibe content.
    # Kept separate from STOPWORDS on purpose: STOPWORDS feeds entity word-set
    # matching (words like "just" can be part of an artist/album name), while
    # this set only trims prompt content for the dominance test.
    "just", "only", "all", "start", "with",
    "song", "songs", "track", "tracks", "playlist",
    "give", "make", "create", "build", "put", "me",
}


def extract_prompt_tracks(prompt: str, all_songs: list[dict]) -> list[dict]:
    """Tracks whose full title appears word-bounded in the prompt.

    Only titles with >=2 significant words are eligible (single-word titles
    like "Love" or "Fire" are far too ambiguous to pin from a prompt phrase).
    Returns one representative song dict per matched title.
    """
    prompt_lc = prompt.lower()
    out: list[dict] = []
    seen: set[str] = set()
    for s in all_songs:
        title = (s.get("title") or "").strip()
        if not title:
            continue
        title_lc = title.lower()
        if title_lc in seen:
            continue
        if len([w for w in re.findall(r"\b\w+\b", title_lc) if w not in STOPWORDS]) < 2:
            continue
        if re.search(rf"\b{re.escape(title_lc)}\b", prompt_lc):
            seen.add(title_lc)
            out.append(s)
    return out


def classify_prompt_mode(prompt: str, entities: dict) -> str:
    """Classify the prompt into a selection mode (P3; replaces the old binary
    'explicit artists => no cap' rule, which backfired on anchor prompts).

    - "free": prompt is entity-dominant (no vibe content words beyond the
      named entities + their initials) AND names at least one artist.
      Deliberate 1-2-artist mixes ("Michael Jackson", "just Radiohead and
      Portishead") keep the old no-cap behavior.
    - "anchor": prompt names explicit entities but ALSO contains vibe/scene
      content ("...upbeat work mix"). Named entities are guaranteed present
      (_guarantee_explicit) and anchor artists are capped (_ANCHOR_CAP_FRAC);
      the rest follows the vibe. Album-only entity-dominant prompts ("off the
      wall mix") land here too: the album is guaranteed but the adaptive
      artist cap stays on (a named album is not a named artist).
    - "default": nothing explicit; adaptive caps as before.
    """
    prompt_words = set(re.findall(r"\b\w+\b", prompt.lower()))
    entity_words: set[str] = set()
    for kind in ("artists", "albums", "tracks"):
        for item in entities.get(kind, []):
            if kind == "tracks" and isinstance(item, dict):
                item = item.get("title", "")
            entity_words |= set(re.findall(r"\b\w+\b", str(item).lower()))
    entity_words |= set(entities.get("initials_tokens", ()))
    content = prompt_words - STOPWORDS - _DOMINANCE_IGNORE - entity_words
    entity_dominant = not content
    has_entities = bool(
        entities.get("artists") or entities.get("albums") or entities.get("tracks")
    )
    if has_entities and entity_dominant and entities.get("artists"):
        return "free"
    if has_entities:
        return "anchor"
    return "default"


# --------------------------------------------------
# PROMPT ERA DETECTION (loose scoring bonus only)
# --------------------------------------------------

_ERA_RANGE_RE = re.compile(
    r"\b(19\d{2}|20\d{2})\s*(?:-|to|–|—)\s*(19\d{2}|20\d{2})\b", re.IGNORECASE
)
_DECADE_DIGIT_RE = re.compile(r"\b(\d{2,4})s\b", re.IGNORECASE)
_DECADE_WORDS = {
    "forties": 1940,
    "fifties": 1950,
    "sixties": 1960,
    "seventies": 1970,
    "eighties": 1980,
    "nineties": 1990,
}
_DECADE_WORD_RE = re.compile(r"\b(" + "|".join(_DECADE_WORDS) + r")\b", re.IGNORECASE)
_EXPLICIT_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _decade_digits_to_range(digits: str) -> tuple[int, int] | None:
    """'1980' -> (1980, 1989); '80' -> (1980, 1989); '20' -> (2020, 2029).

    Only decade-shaped numbers (ending in 0) are treated as a decade;
    anything else (e.g. a stray "1975s" typo) is not a decade and returns
    None so the caller can fall through to other detection strategies.
    """
    n = int(digits)
    if n % 10 != 0:
        return None
    if len(digits) == 4:
        start = n
    elif len(digits) == 2:
        # Two-digit shorthand: low numbers ("00s", "10s", "20s") read as the
        # 2000s; higher numbers ("50s".."90s") read as the 1900s.
        start = (2000 if n < 30 else 1900) + n
    else:
        return None
    return (start, start + 9)


def detect_prompt_era(prompt: str) -> tuple[int, int] | None:
    """
    Detect an implied era (inclusive start/end year) from a vibe prompt.
    Handles explicit year ranges ("1975-1980", "1975 to 1980"), decade
    shorthand ("80s", "1980s"), decade words ("eighties"), and a bare
    explicit year ("1975") as a fallback. Returns None when no era is
    implied. Callers must only ever use the result as a loose scoring bonus
    (see `_era_bonus`), never as a filter.
    """
    if not prompt:
        return None
    text = prompt.lower()

    m = _ERA_RANGE_RE.search(text)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        return (min(y1, y2), max(y1, y2))

    m = _DECADE_DIGIT_RE.search(text)
    if m:
        rng = _decade_digits_to_range(m.group(1))
        if rng:
            return rng

    m = _DECADE_WORD_RE.search(text)
    if m:
        start = _DECADE_WORDS[m.group(1).lower()]
        return (start, start + 9)

    m = _EXPLICIT_YEAR_RE.search(text)
    if m:
        year = int(m.group(1))
        return (year, year)

    return None


# --------------------------------------------------
# MAIN (updated flow with context playlist)
# --------------------------------------------------


def _main_impl(args):
    playlist_name = args.playlist_name
    prompt = args.prompt or input("Enter a prompt for the playlist vibe: ").strip()
    if not prompt:
        print("Prompt required - exiting.")
        return

    print("=== NaviDJ - AI Playlist Generator ===")
    print("=" * 60)
    print(f"Using LLM backend: {LLM_MODE} (model: {LLM_MODEL})")

    # ========== STAGE 0: LIBRARY FETCH ==========
    print("STAGE: Fetching Library")
    start_t = time.time()
    all_songs = fetch_all_subsonic_songs()
    if not all_songs:
        print("No songs found on the server.")
        return
    print(
        f"Library fetch complete: {len(all_songs)} songs ({time.time() - start_t:.1f}s)"
    )

    # ========== STAGE 1: METADATA GATHERING ==========
    print("STAGE: Gathering Metadata")
    start_t = time.time()
    all_artists = fetch_all_artists()
    all_genres = fetch_all_genres()
    all_albums = [
        a
        for a in {s.get("album") for s in all_songs if s.get("album")}
        if isinstance(a, str)
    ]

    # Randomize to avoid bias
    random.shuffle(all_artists)
    random.shuffle(all_genres)
    random.shuffle(all_albums)
    print(
        f"Gathered metadata: {len(all_artists)} artists, {len(all_genres)} genres, {len(all_albums)} albums ({time.time() - start_t:.1f}s)"
    )


    # ========== SEMANTIC PRE-FILTERING (OPTIONAL) ==========
    print("STAGE: Semantic Pre-filtering")
    start_t = time.time()
    sem_artists = all_artists
    sem_genres = all_genres
    sem_albums = all_albums
    semantic_song_ids = set()

    if embedding_manager:
        print(f"Using embedding model: {EMBEDDING_MODEL}")
        embedding_manager.check_library_size(len(all_songs))

        # 1. Semantic Metadata Pre-selection (Top 40 each)
        print("Performing semantic metadata pre-selection...")
        sem_artists = embedding_manager.find_similar(prompt, all_artists, top_k=40)
        sem_genres = embedding_manager.find_similar(prompt, all_genres, top_k=40)
        album_artist_map = {}
        for s in all_songs:
            al = s.get("album")
            ar = s.get("artist")
            if al and isinstance(al, str) and al not in album_artist_map:
                album_artist_map[al] = ar

        album_texts = [
            f"{al} by {album_artist_map[al]}" if album_artist_map.get(al) else al
            for al in all_albums
        ]
        sem_album_indices = embedding_manager.find_similar_indices(
            prompt, album_texts, top_k=40
        )
        sem_albums = [all_albums[i] for i in sem_album_indices]
        # 2. Semantic Song Pre-selection (Top 200)
        print("Finding semantically similar songs...")
        song_texts = [_song_embedding_text(s) for s in all_songs]
        sem_song_indices = embedding_manager.find_similar_indices(
            prompt, song_texts, top_k=200
        )
        semantic_song_ids = {
            all_songs[idx]["id"] for idx in sem_song_indices if idx < len(all_songs)
        }

        print(f"Semantic pre-filtering complete ({time.time() - start_t:.1f}s)")
    else:
        print("Skipping semantic pre-filtering (no embedding manager initialized).")

    # ========== CONTEXT ANALYSIS ==========
    print("STAGE: Context Analysis")
    start_t = time.time()

    # Extract explicit mentions from prompt FIRST, so context selection is aware
    # of any explicit request (used to discard an off-target context playlist
    # just below). Only depends on prompt + catalog lists.
    prompt_entities = extract_prompt_entities(
        prompt, all_artists, all_genres, all_albums
    )
    explicit_artists = prompt_entities["artists"]
    explicit_genres = prompt_entities["genres"]
    explicit_albums = prompt_entities["albums"]
    explicit_tracks = extract_prompt_tracks(prompt, all_songs)
    prompt_entities["tracks"] = explicit_tracks
    mode = classify_prompt_mode(prompt, prompt_entities)
    era_range = detect_prompt_era(prompt)
    # P5: mode + full entity set are decision-critical and previously unlogged.
    print(
        f"Prompt mode: {mode} | artists={explicit_artists or '-'} | "
        f"albums={explicit_albums or '-'} | genres={explicit_genres or '-'} | "
        f"tracks={[t.get('title') for t in explicit_tracks] or '-'}"
    )

    existing_playlists = fetch_all_playlists(exclude_name=playlist_name)
    context_songs = select_context_playlist_songs(
        prompt, existing_playlists, all_songs, embedding_manager=embedding_manager
    )

    # P0 context gate: if the prompt names explicit artist(s) or album(s) but
    # the chosen context playlist contains none of them, discard it. Its
    # context bonuses (+4.0 in-playlist, +2.0 metadata) can otherwise outrank
    # explicit matches and pull in off-target tracks (e.g. "michael jackson
    # mix" picking an indie-rock list; test_dj_override poisoning 5 straight
    # runs on Sep 9). The 0.15 semantic floor alone never fired (observed
    # scores 0.443-0.684), so this entity check is the effective gate.
    if (
        (explicit_artists or explicit_albums)
        and context_songs
        and not _context_overlaps_explicit(
            context_songs, explicit_artists, explicit_albums
        )
    ):
        print(
            f"Discarding context playlist: no songs by requested artist(s) "
            f"{', '.join(explicit_artists) or '-'} or from requested album(s) "
            f"{', '.join(explicit_albums) or '-'}. The 0.15 semantic floor "
            f"passed; this entity check rejected it (P5 log)."
        )
        context_songs = []

    # Extract metadata from context (Top N most frequent)
    context_artists = []
    context_genres = []
    context_albums = []
    if context_songs:
        art_counts = Counter()
        gen_counts = Counter()
        alb_counts = Counter()
        for s in context_songs:
            if s.get("artist"):
                for a in _split_artist_string(s["artist"]):
                    art_counts[a] += 1
            for g in _song_genres(s):
                gen_counts[g] += 1
            if s.get("album"):
                alb_counts[s["album"]] += 1

        context_artists = [a for a, _ in art_counts.most_common(10)]
        context_genres = [g for g, _ in gen_counts.most_common(5)]
        context_albums = [al for al, _ in alb_counts.most_common(5)]

    print(f"Context analysis complete ({time.time() - start_t:.1f}s)")

    if explicit_artists:
        print(f"Explicit artists identified: {', '.join(explicit_artists)}")
    if explicit_genres:
        print(f"Explicit genres identified: {', '.join(explicit_genres)}")
    if explicit_albums:
        print(f"Explicit albums identified: {', '.join(explicit_albums)}")
    if explicit_tracks:
        print(
            "Explicit tracks identified: "
            + ", ".join(t.get("title") or "?" for t in explicit_tracks)
        )
    if era_range:
        print(f"Implied era detected: {era_range[0]}-{era_range[1]} (loose scoring bonus only)")

    # ========== STAGE 1: METADATA SELECTION ==========
    print("STAGE: Selecting Focus Metadata")
    start_t = time.time()
    if embedding_manager is not None:
        print("Skipping LLM metadata selection; using semantically derived candidates.")

        # Determine focus artists (explicitly requested + top 5 semantic)
        focus_artists = list(dict.fromkeys(explicit_artists + sem_artists[:5]))

        # Inject albums from these focus artists
        focus_albums = []
        for al, ar in album_artist_map.items():
            if ar:
                ar_split = _split_artist_string(ar)
                if any(fa in ar_split for fa in focus_artists):
                    focus_albums.append(al)

        combined_albums = list(dict.fromkeys(focus_albums + sem_albums))

        selected_metadata = {
            "artists": list(dict.fromkeys(explicit_artists + sem_artists))[:15],
            "genres": sem_genres[:15],
            "albums": combined_albums[:15],
        }
    else:
        print("Using LLM for metadata selection...")
        selected_metadata = select_focus_metadata_single_call(
            prompt=prompt,
            all_artists=sem_artists,
            all_genres=sem_genres,
            all_albums=sem_albums,
        )
    duration = time.time() - start_t

    # Display selected metadata (for debugging)
    all_artists_combined = list(
        dict.fromkeys(explicit_artists + context_artists + selected_metadata["artists"])
    )
    all_genres_combined = list(
        dict.fromkeys(explicit_genres + context_genres + selected_metadata["genres"])
    )
    all_albums_combined = list(
        dict.fromkeys(context_albums + selected_metadata["albums"])
    )

    print(f"Chosen Artists: {', '.join(all_artists_combined)}")
    print(f"Chosen Genres: {', '.join(all_genres_combined)}")
    print(f"Chosen Albums: {', '.join(all_albums_combined)}")
    print(f"Metadata selection complete ({duration:.1f}s)")

    # ========== STAGE 2: WEIGHTED METADATA FILTER ==========
    print("STAGE: Filtering Candidates")
    start_t = time.time()
    context_song_ids = {s["id"] for s in context_songs} if context_songs else set()

    candidate_pool = filter_library_by_metadata(
        explicit_artists=explicit_artists,
        explicit_genres=explicit_genres,
        explicit_albums=explicit_albums,
        context_artists=context_artists,
        context_genres=context_genres,
        context_albums=context_albums,
        selected_artists=selected_metadata["artists"],
        selected_genres=selected_metadata["genres"],
        selected_albums=selected_metadata["albums"],
        all_songs=all_songs,
        context_song_ids=context_song_ids,
        semantic_song_ids=semantic_song_ids,
        era_range=era_range,
    )
    # Collapse variant duplicates (remasters/editions/dupe rips) before the LLM
    # sees the pool, so it can't pick several variants of the same recording.
    pre_dedup = len(candidate_pool)
    candidate_pool = _dedup_variants(candidate_pool)
    if len(candidate_pool) < pre_dedup:
        print(
            f"Variant dedup: collapsed {pre_dedup - len(candidate_pool)} duplicate "
            f"variant(s) -> {len(candidate_pool)} candidates."
        )

    print(
        f"Final candidate pool: {len(candidate_pool)} songs ({time.time() - start_t:.1f}s)"
    )

    if not candidate_pool:
        print("\nNo songs match the selected criteria. Try a different prompt.")
        return

    # ========== STAGE 3: PLAYLIST GENERATION ==========
    print("STAGE: Generating Playlist")
    start_t = time.time()
    print(f"Generating playlist...")
    playlist_items = generate_playlist_chunked(
        prompt=prompt,
        filtered_songs=candidate_pool,
        min_songs=args.min_songs,
        chunk_size=args.chunk_size,
        explicit_artists=explicit_artists,
        explicit_albums=explicit_albums,
        explicit_genres=explicit_genres,
        mode=mode,
    )
    print(
        f"Final playlist generated: {len(playlist_items)} tracks ({time.time() - start_t:.1f}s)"
    )

    # Sanitize playlist (resolve any missing IDs)
    playlist_items = _sanitize_playlist(playlist_items, candidate_pool)

    if not playlist_items:
        print("\nFailed to generate playlist.")
        return

    # ========== STAGE 3b: POST-SELECTION QUALITY PASSES ==========
    print("STAGE: Finalizing Playlist")
    id_lookup = {s["id"]: s for s in candidate_pool}

    # Final variant-dedup safety pass (in case padding reintroduced a variant).
    before = len(playlist_items)
    playlist_items = _dedup_playlist_variants(playlist_items, id_lookup)
    if len(playlist_items) < before:
        print(f"Final dedup pass: removed {before - len(playlist_items)} variant duplicate(s).")

    # Per-artist cap, mode-aware (P3: free / anchor / default).
    playlist_items = _enforce_artist_cap(
        playlist_items,
        candidate_pool,
        args.min_songs,
        explicit_artists,
        mode=mode,
    )

    # Per-album cap (P6): no album may dominate unless explicitly named.
    playlist_items = _enforce_album_cap(
        playlist_items, candidate_pool, args.min_songs, explicit_albums
    )

    # P3 guaranteed inclusion: the user's actual request (named tracks/albums/
    # artists) must survive the caps, so this runs AFTER them.
    playlist_items = _guarantee_explicit(
        playlist_items,
        candidate_pool,
        explicit_artists=explicit_artists,
        explicit_albums=explicit_albums,
        explicit_tracks=explicit_tracks,
    )

    # Top up if the quality passes dropped us below the requested minimum.
    if len(playlist_items) < args.min_songs:
        playlist_items = ensure_min_songs(
            playlist_items,
            candidate_pool,
            args.min_songs,
            explicit_artists=explicit_artists,
            mode=mode,
        )

    # ========== STRUCTURED OUTPUT FOR FRONTEND ==========
    # One machine-readable line the web UI can pick out of the raw log to
    # render the final tracklist. Full song data (artist/album/genres/year)
    # is pulled from id_lookup, which already maps candidate ids -> full
    # song dicts. Output-only; does not affect selection.
    playlist_tracks = []
    for _item in playlist_items:
        _full = id_lookup.get(_item.get("id"), {})
        playlist_tracks.append(
            {
                "title": _full.get("title") or _item.get("title"),
                "artist": _full.get("artist"),
                "album": _full.get("album"),
                "genres": _song_genres(_full) if _full else [],
                "year": _full.get("releaseYear"),
            }
        )
    print(
        "PLAYLIST_JSON: "
        + json.dumps({"playlist_name": playlist_name, "tracks": playlist_tracks})
    )

    # ========== STAGE 4: UPLOAD TO SERVER ==========
    print("STAGE: Uploading to Server")
    start_t = time.time()
    song_ids = [t["id"] for t in playlist_items]
    success = _update_playlist_on_server(playlist_name, song_ids, prompt)

    if success:
        print(
            f"Playlist '{playlist_name}' successfully updated on server ({time.time() - start_t:.1f}s)"
        )
    else:
        print("ERROR: Failed to update playlist on server.")

    print("STAGE: Complete")
    print("\nComplete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a playlist based on a vibe prompt."
    )
    parser.add_argument(
        "--playlist_name",
        type=str,
        default="naviDJ",
        help="Name of the playlist to create or update.",
    )
    parser.add_argument("--prompt", type=str, help="Vibe prompt for the playlist.")
    parser.add_argument(
        "--min_songs",
        type=int,
        default=35,
        help="Target/exact number of songs in the playlist.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of songs per LLM chunk (adjust based on context size).",
    )
    parser.add_argument(
        "--llm_mode",
        type=str,
        choices=["openai", "ollama"],
        default=DEFAULT_LLM_MODE,
        help="Which LLM backend to use (overrides secrets.txt).",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help="Which LLM model to use (overrides secrets.txt).",
    )
    args = parser.parse_args()

    configure_llm(args.llm_mode, args.llm_model)
    _main_impl(args)
