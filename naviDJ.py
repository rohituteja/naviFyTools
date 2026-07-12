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
    --min_songs       Minimum number of songs in the playlist (default: 35)
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
import time

from tqdm import tqdm
from openai import BadRequestError, OpenAI
import argparse
from typing import List, Dict  # optional, only for type hints
from collections import Counter
import configparser
from rapidfuzz import fuzz
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
        starred = fetch_starred_ids()

        for s in songs:
            all_songs.append(
                {
                    "id": s.get("id"),
                    "title": s.get("title"),
                    "artist": s.get("artist"),
                    "genre": s.get("genre"),
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
        return client.chat.completions.create(**kwargs)

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
) -> list[dict]:
    """
    Filter library based on combined focus items with hierarchical, additive weighting.
    Note that context playlist songs are used only as a vibe/metadata signal and are not scored for direct inclusion.
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
        song_genre = s.get("genre")
        song_album = s.get("album")
        song_id = s.get("id")

        score = 0.0

        # New Additive Weighting System:

        # 1. Explicit Matches (+3.0 each)
        if any(a in exp_a for a in song_artists):
            score += 3.0
        if song_genre in exp_g:
            score += 3.0
        if song_album in exp_al:
            score += 3.0

        # 2. Context Playlist Matches
        # Explicitly in context playlist (+4.0)
        if song_id in ctx_ids:
            score += 4.0
        # Metadata matches if part of context
        if any(a in ctx_a for a in song_artists):
            score += 2.0
        if song_genre in ctx_g:
            score += 2.0
        if song_album in ctx_al:
            score += 2.0

        # 3. LLM Chosen Focus Matches
        if any(a in sel_a for a in song_artists):
            score += 1.5
        if song_genre in sel_g:
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

    if len(pool) < len(filtered):
        print(f"Stratified pool: {len(tier1)} high + {len(tier2)} medium + {len(tier3)} discovery = {len(pool)} songs.")

    return pool


def generate_playlist_single_call(
    prompt: str,
    filtered_songs: list[dict],
    min_songs: int,
    explicit_artists: list[str] = None,
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
        genre   = s.get("genre")   or "Unknown"
        album   = s.get("album")   or "Unknown"
        year    = s.get("releaseYear") or "Unknown"
        lines.append(f"{n}{star}|{title}|{artist}|{genre}|{album}|{year}")
    candidate_text = "\n".join(lines)

    # -- 3. System prompt --------------------------------------------------
    if explicit_artists:
        explicit_line = (
            f"- The user specifically requested these artists: {', '.join(explicit_artists)}. "
            "Strongly prefer songs by them and do NOT enforce artist diversity against them.\n"
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
            playlist, filtered_songs, min_songs, explicit_artists=explicit_artists
        )

    return playlist


def generate_playlist_chunked(
    prompt: str,
    filtered_songs: list[dict],
    min_songs: int,
    chunk_size: int = 200,
    explicit_artists: list[str] = None,
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
            unique_selections, filtered_songs, min_songs, explicit_artists=explicit_artists
        )

    return unique_selections[:min_songs]


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
# PROMPT ARTIST EXTRACTION
# --------------------------------------------------


def extract_prompt_artists(prompt: str, all_artists: list[str]) -> list[str]:
    """
    Detect artist names that appear as whole words (case-insensitive) in the user's prompt.
    Returns them in library order to preserve stability.
    """
    import re

    # Split prompt into words, ignore punctuation
    words = set(re.findall(r"\b\w+\b", prompt.lower()))
    result = []
    for artist in all_artists:
        artist_words = set(re.findall(r"\b\w+\b", artist.lower()))
        # If any prompt word is a whole word in the artist name
        if words & artist_words:
            result.append(artist)
    return result


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


# --------------------------------------------------
# PLAYLIST LENGTH ENFORCEMENT
# --------------------------------------------------


def ensure_min_songs(
    playlist: list[dict],
    candidates: list[dict],
    min_songs: int,
    max_songs: int = 50,
    explicit_artists: list[str] = None,
) -> list[dict]:
    """
    Pad *playlist* up to *min_songs* using the highest-relevance unused
    candidates (descending `_relevance_score`, with only minor tie-breaking
    randomness). Padding respects the variant-dedup rules and, when no explicit
    artists were requested, the adaptive per-artist cap.
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
            a = _split_artist_string(full.get("artist") or "")
            artist_counts[a[0].lower() if a else ""] += 1

    cap = None if explicit_artists else max(2, math.ceil(0.15 * min_songs))

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
        a = _split_artist_string(s.get("artist") or "")
        pa = a[0].lower() if a else ""
        if cap is not None and artist_counts[pa] >= cap:
            continue
        playlist.append({"id": s["id"], "title": s.get("title")})
        existing_keys.add(k)
        artist_counts[pa] += 1
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


def _enforce_artist_cap(
    playlist: list[dict],
    candidates: list[dict],
    target_size: int,
    explicit_artists: list[str] = None,
) -> list[dict]:
    """
    Adaptive post-selection per-artist cap. Only applies when *explicit_artists*
    is empty (deliberate 1-2 artist mixes are never capped). Cap =
    max(2, ceil(0.15 * target_size)). Songs over the cap for an artist (keeping
    the highest-relevance ones) are replaced by the highest-relevance unused
    candidates from other artists, respecting the dedup rules and the cap.
    """
    if explicit_artists:
        return playlist

    cap = max(2, math.ceil(0.15 * target_size))
    id_lookup = {s["id"]: s for s in candidates}

    def primary_artist(song_or_item: dict) -> str:
        full = id_lookup.get(song_or_item.get("id"), song_or_item)
        a = _split_artist_string(full.get("artist") or "")
        return a[0].lower() if a else ""

    def relscore(item: dict) -> float:
        return id_lookup.get(item.get("id"), {}).get("_relevance_score", 0)

    # Decide which picks to keep per artist: highest relevance up to the cap.
    from collections import defaultdict

    by_artist: dict[str, list[dict]] = defaultdict(list)
    for item in playlist:
        by_artist[primary_artist(item)].append(item)

    keep_ids: set[str] = set()
    for _artist, items in by_artist.items():
        items_sorted = sorted(items, key=relscore, reverse=True)
        for it in items_sorted[:cap]:
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
            artist_counts[primary_artist(item)] += 1
            full = id_lookup.get(item["id"])
            if full:
                existing_keys.add(_variant_dedup_key(full))
        else:
            removed += 1

    if removed == 0:
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
        pa = primary_artist(s)
        if artist_counts[pa] >= cap:
            continue
        k = _variant_dedup_key(s)
        if k in existing_keys:
            continue
        kept.append({"id": s["id"], "title": s.get("title")})
        kept_ids.add(s["id"])
        artist_counts[pa] += 1
        existing_keys.add(k)

    print(
        f"Artist cap ({cap}/artist) enforced: replaced {removed} over-cap pick(s)."
    )
    return kept


# --------------------------------------------------
# PLAYLIST ENTRY SANITISER
# --------------------------------------------------


def _sanitize_playlist(
    entries: List[dict], candidates: List[dict], fuzzy_threshold: int = 90
) -> List[dict]:
    """
    Ensure each playlist entry has an 'id'. If an entry only has a 'title'
    (and optionally 'artist'), try to resolve the matching song in *candidates*
    via a case-insensitive title-and-artist match. If that fails, use fuzzy matching
    on title and artist. Drop any rows we can't resolve. This is backend-agnostic and
    therefore safe for both Ollama and OpenAI modes.
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
            continue
        # Fuzzy match fallback using rapidfuzz
        best_score = 0
        best_id = None
        for s in candidates:
            title_score = fuzz.ratio(
                (e.get("title") or "").lower(), (s.get("title") or "").lower()
            )
            artist_score = fuzz.ratio(
                (e.get("artist") or "").lower(), (s.get("artist") or "").lower()
            )
            avg_score = (title_score + artist_score) // 2
            if avg_score > best_score and avg_score >= fuzzy_threshold:
                best_score = avg_score
                best_id = s["id"]
        if best_id:
            cleaned.append({"id": best_id, "title": e.get("title")})
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
            # Exact match (full item name in prompt)
            if item_lc in prompt_lc:
                entities[key].append(item)
            # Partial match (ALL significant words from item must be in prompt)
            elif len(item_words) >= 2 and item_words.issubset(prompt_words):
                entities[key].append(item)

    # Artists: check both exact and partial matches
    check_matches(all_artists, "artists")

    # Albums: check both exact and partial matches
    check_matches(all_albums, "albums")

    # Genres: exact match only (genres are typically single words or short phrases)
    for genre in all_genres:
        if genre.lower() in prompt_lc:
            entities["genres"].append(genre)

    return entities


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

    # ========== STAGE 0: LIBRARY FETCH ==========
    start_t = time.time()
    all_songs = fetch_all_subsonic_songs()
    if not all_songs:
        print("No songs found on the server.")
        return
    print(
        f"Library fetch complete: {len(all_songs)} songs ({time.time() - start_t:.1f}s)"
    )

    # ========== STAGE 1: METADATA GATHERING ==========
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
        song_texts = [
            f"{s.get('title', '')} by {s.get('artist', '')}" for s in all_songs
        ]
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
    start_t = time.time()
    existing_playlists = fetch_all_playlists(exclude_name=playlist_name)
    context_songs = select_context_playlist_songs(
        prompt, existing_playlists, all_songs, embedding_manager=embedding_manager
    )

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
            if s.get("genre"):
                gen_counts[s["genre"]] += 1
            if s.get("album"):
                alb_counts[s["album"]] += 1

        context_artists = [a for a, _ in art_counts.most_common(10)]
        context_genres = [g for g, _ in gen_counts.most_common(5)]
        context_albums = [al for al, _ in alb_counts.most_common(5)]
    # Extract explicit mentions from prompt
    prompt_entities = extract_prompt_entities(
        prompt, all_artists, all_genres, all_albums
    )
    explicit_artists = prompt_entities["artists"]
    explicit_genres = prompt_entities["genres"]
    explicit_albums = prompt_entities["albums"]

    print(f"Context analysis complete ({time.time() - start_t:.1f}s)")

    if explicit_artists:
        print(f"Explicit artists identified: {', '.join(explicit_artists)}")
    if explicit_genres:
        print(f"Explicit genres identified: {', '.join(explicit_genres)}")

    # ========== STAGE 1: METADATA SELECTION ==========
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

    if candidate_pool:
        sample_scores = [
            (s["title"], s.get("_metadata_score", 0)) for s in candidate_pool[:3]
        ]

    if not candidate_pool:
        print("\nNo songs match the selected criteria. Try a different prompt.")
        return

    # ========== STAGE 3: PLAYLIST GENERATION ==========
    start_t = time.time()
    print(f"Generating playlist...")
    playlist_items = generate_playlist_chunked(
        prompt=prompt,
        filtered_songs=candidate_pool,
        min_songs=args.min_songs,
        chunk_size=args.chunk_size,
        explicit_artists=explicit_artists,
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
    id_lookup = {s["id"]: s for s in candidate_pool}

    # Final variant-dedup safety pass (in case padding reintroduced a variant).
    before = len(playlist_items)
    playlist_items = _dedup_playlist_variants(playlist_items, id_lookup)
    if len(playlist_items) < before:
        print(f"Final dedup pass: removed {before - len(playlist_items)} variant duplicate(s).")

    # Adaptive per-artist cap (only when no explicit artists were requested).
    playlist_items = _enforce_artist_cap(
        playlist_items, candidate_pool, args.min_songs, explicit_artists
    )

    # Top up if the quality passes dropped us below the requested minimum.
    if len(playlist_items) < args.min_songs:
        playlist_items = ensure_min_songs(
            playlist_items,
            candidate_pool,
            args.min_songs,
            explicit_artists=explicit_artists,
        )

    # ========== STAGE 4: UPLOAD TO SERVER ==========
    start_t = time.time()
    song_ids = [t["id"] for t in playlist_items]
    success = _update_playlist_on_server(playlist_name, song_ids, prompt)

    if success:
        print(
            f"Playlist '{playlist_name}' successfully updated on server ({time.time() - start_t:.1f}s)"
        )
    else:
        print("ERROR: Failed to update playlist on server.")

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
        help="Minimum number of songs in the playlist.",
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
