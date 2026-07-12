"""
MusicBrainz-only genre sync for local music libraries.

- Fetches genres from MusicBrainz for each track (artist/album tags).
- Writes/updates GENRE tags in audio files.

Usage:
    python portGenres.py [<library_dir>] [--dry-run]

Arguments:
    <library_dir>   Path to your music library (default: current dir or $MUSIC_LIBRARY)
    --dry-run       Only print intended changes, do not modify files
"""

import os
import re
import time
import json
import argparse
import logging
from typing import List, Set
import configparser

import requests
from rapidfuzz import fuzz
from tqdm import tqdm
from mutagen import File as MutagenFile
import unicodedata

# Load secrets from secrets.txt (if needed for future API keys)
secrets = configparser.ConfigParser()
secrets.read(os.path.join(os.path.dirname(__file__), "secrets.txt"))

# --------------- Paths & config ---------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
DEFAULT_LIBRARY_DIR = os.getenv("MUSIC_LIBRARY", PARENT_DIR)
MUSICBRAINZ_BASE = "https://musicbrainz.org/ws/2"
USER_AGENT = "naviFy-portGenres"
REQUEST_DELAY = 0.5  # seconds, per MusicBrainz etiquette
REQUEST_TIMEOUT = 15  # seconds

SUPPORTED_EXTENSIONS = (".flac", ".mp3", ".ogg", ".oga", ".opus", ".m4a", ".m4b", ".mp4")

# --------------- Logging ---------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("genre-sync")

# --------------- MusicBrainz lookup cache ---------------
# In-memory dict backed by a JSON file next to the script. Keyed by the
# lookup query (artist name / artist+album), including negative results
# (empty tag lists) so repeated runs skip the throttled MB request entirely.
MB_CACHE_PATH = os.path.join(SCRIPT_DIR, ".mb_cache.json")
MB_CACHE_SAVE_EVERY = 50  # flush to disk after this many new entries

_mb_cache: dict = {}
_mb_cache_dirty_count = 0


def load_mb_cache() -> None:
    """Load the on-disk MusicBrainz lookup cache into memory (best-effort)."""
    global _mb_cache
    try:
        with open(MB_CACHE_PATH, "r", encoding="utf-8") as f:
            _mb_cache = json.load(f)
        logger.info("Loaded MB cache: %d entries", len(_mb_cache))
    except FileNotFoundError:
        _mb_cache = {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read MB cache (%s) – starting fresh: %s", MB_CACHE_PATH, exc)
        _mb_cache = {}


def save_mb_cache() -> None:
    """Persist the in-memory MusicBrainz lookup cache to disk (best-effort)."""
    try:
        with open(MB_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_mb_cache, f)
    except OSError as exc:
        logger.warning("Failed to save MB cache to %s: %s", MB_CACHE_PATH, exc)


def _mb_cache_store(key: str, value: List[str]) -> None:
    """Record a cache entry (including negative/empty results) and flush periodically."""
    global _mb_cache_dirty_count
    _mb_cache[key] = value
    _mb_cache_dirty_count += 1
    if _mb_cache_dirty_count >= MB_CACHE_SAVE_EVERY:
        save_mb_cache()
        _mb_cache_dirty_count = 0

# --------------- Text helpers ---------------

def _strip_parens_feats(text: str) -> str:
    patterns = [
        r"\(feat\..*?\)", r"\(ft\..*?\)", r"-\s*feat\..*", r"-\s*ft\..*",
        r"ft\..*", r"feat\..*", r"\(.*?remaster.*?\)", r"- .*? Remaster",
        r"- Remastered.*?", r"- Single Version", r"- Live", r"- From",
        r"\[.*?\]", r"\(.*?\)"
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return re.sub(r"[-–—]\s*$", "", text).strip()


def clean_field(field: str) -> str:
    return _strip_parens_feats(field or "")


def normalize(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if (c.isalnum() or c.isspace())).lower().strip()


def format_existing_genres(raw) -> List[str]:
    tokens: List[str] = []
    if isinstance(raw, str):
        raw = [raw]
    for item in raw or []:
        for part in re.split(r"[,/]", item):
            part = part.strip().lower()
            if part and part not in tokens:
                tokens.append(part)
    return tokens

# --------------- MusicBrainz helpers ---------------

def _mb_get(endpoint: str, params: dict) -> dict:
    """GET wrapper with polite delay & JSON handling."""
    time.sleep(REQUEST_DELAY)
    resp = requests.get(
        f"{MUSICBRAINZ_BASE}/{endpoint}",
        params={**params, "fmt": "json"},
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    resp.raise_for_status()
    return resp.json()


def mb_artist_tags(artist_name: str) -> List[str]:
    cache_key = f"artist::{artist_name.lower()}"
    if cache_key in _mb_cache:
        return _mb_cache[cache_key]

    params = {"query": f'artist:"{artist_name}"', "limit": 1}
    try:
        data = _mb_get("artist/", params)
    except requests.RequestException as exc:
        # Network/HTTP failure - transient, must NOT be cached as a negative result.
        logger.error("MB artist lookup request failed for %s: %s", artist_name, exc)
        return []

    try:
        artist = data.get("artists", [None])[0]
        tags = artist.get("tags", []) if artist else []
        tags_sorted = sorted(tags, key=lambda t: t.get("count", 0), reverse=True)
        result = [t["name"].lower() for t in tags_sorted][:5]
    except Exception as exc:
        # Successful response, unexpected shape - treat as "no tags" and cache it.
        logger.debug("MB artist lookup response parsing failed for %s: %s", artist_name, exc)
        result = []

    _mb_cache_store(cache_key, result)
    return result


def mb_release_tags(artist_name: str, album_name: str) -> List[str]:
    if not album_name:
        return []

    cache_key = f"release::{artist_name.lower()}::{album_name.lower()}"
    if cache_key in _mb_cache:
        return _mb_cache[cache_key]

    # include artist to narrow search
    q = f'releasegroup:"{album_name}" AND artist:"{artist_name}"'
    params = {"query": q, "limit": 1, "inc": "tags"}
    try:
        data = _mb_get("release-group/", params)
    except requests.RequestException as exc:
        # Network/HTTP failure - transient, must NOT be cached as a negative result.
        logger.error("MB release lookup request failed for %s – %s: %s", artist_name, album_name, exc)
        return []

    try:
        rg = data.get("release-groups", [None])[0]
        tags = rg.get("tags", []) if rg else []
        tags_sorted = sorted(tags, key=lambda t: t.get("count", 0), reverse=True)
        result = [t["name"].lower() for t in tags_sorted][:5]
    except Exception as exc:
        # Successful response, unexpected shape - treat as "no tags" and cache it.
        logger.debug("MB release lookup response parsing failed for %s – %s: %s", artist_name, album_name, exc)
        result = []

    _mb_cache_store(cache_key, result)
    return result

# --------------- Genre aggregation ---------------

def fetch_official_genres() -> Set[str]:
    """Fetch the official genre list from MusicBrainz (used only to order canonical
    genres first in the written tag list - non-whitelist tags are still kept)."""
    url = MUSICBRAINZ_BASE + "/genre/all?fmt=txt"
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("MB official genre list request failed: %s", exc)
        raise
    return set(line.strip().lower() for line in response.text.splitlines() if line.strip())


def _sanity_filter(tags: List[str]) -> List[str]:
    """Light sanity filtering only - NOT a whitelist gate. Strips empties/whitespace
    and drops pure-noise tags (e.g. single characters left over from bad splits)."""
    cleaned = []
    for tag in tags:
        tag = (tag or "").strip().lower()
        if not tag or len(tag) <= 1:
            continue
        cleaned.append(tag)
    return cleaned


def genres_for_track(artist: str, album: str, official_genres: Set[str]) -> List[str]:
    """Aggregate MusicBrainz tags for a track. All tags are kept (moods/eras/scenes
    like "80s", "chill", "party" included) - only deduped and lightly sanity-filtered.
    Tags that are official MusicBrainz genres are ordered first so that anything
    reading only the first genre still sees a real genre; the rest (folksonomy tags)
    are appended after, preserving their relative MB-count-sorted order."""
    raw_tags = []
    raw_tags.extend(mb_release_tags(artist, album))
    raw_tags.extend(mb_artist_tags(artist))

    # Deduplicate while preserving order
    seen = set()
    unique_tags = []
    for tag in _sanity_filter(raw_tags):
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)

    canonical = [tag for tag in unique_tags if tag in official_genres]
    other = [tag for tag in unique_tags if tag not in official_genres]
    return canonical + other

# --------------- Tag writer ---------------

def update_audio_file(path: str, genres: List[str]):
    audio = MutagenFile(path, easy=True)
    if audio is None:
        logger.warning("%s unsupported – skipped", path)
        return
    if audio.tags is None:
        try:
            audio.add_tags()
        except Exception:
            logger.warning("Can't add tags → %s", path)
            return
    if not genres:
        return
    audio.tags["genre"] = [", ".join(genres)] if path.lower().endswith((".m4a", ".m4b", ".mp4")) else genres
    try:
        audio.save()
        logger.debug("Updated %s", os.path.basename(path))
    except Exception as exc:
        logger.error("Save failed for %s: %s", path, exc)

# --------------- Library processing ---------------

def gather_paths(root_dir: str) -> List[str]:
    return [os.path.join(dp, f) for dp, _, files in os.walk(root_dir) for f in files if f.lower().endswith(SUPPORTED_EXTENSIONS)]


def process_library(root: str, dry=False):
    load_mb_cache()
    try:
        official_genres = fetch_official_genres()
        missing_albums = set()
        for i, path in enumerate(tqdm(gather_paths(root), desc="Updating genres", unit="file", dynamic_ncols=True)):
            try:
                audio = MutagenFile(path, easy=True)
                if audio is None or not audio.tags:
                    continue
                title = clean_field(audio.tags.get("title", [os.path.splitext(os.path.basename(path))[0]])[0])
                artist = clean_field(audio.tags.get("artist", [""])[0])
                album = clean_field(audio.tags.get("album", [""])[0])
                if not artist:
                    continue
                genres = genres_for_track(artist, album, official_genres)
                existing_genres = format_existing_genres(audio.tags.get("genre", []))
                if not genres:
                    genres = existing_genres
                    if dry:
                        logger.info("DRY‑RUN fallback: No MB genres found. Retaining existing for %s: %s", path, ", ".join(existing_genres) or "<none>")
                    else:
                        update_audio_file(path, genres)
                    if album and artist:
                        missing_albums.add((artist, album))
                else:
                    if dry:
                        logger.info("DRY‑RUN update: %s would change from [%s] → [%s]", path, ", ".join(existing_genres) or "<none>", ", ".join(genres))
                    else:
                        update_audio_file(path, genres)
                if dry and i >= 19:
                    logger.info("Dry‑run limit reached. Exiting.")
                    break
            except Exception as exc:
                logger.error("Error %s: %s", path, exc)

        if missing_albums:
            logger.warning("\nMissing genre info for the following albums:")
            for artist, album in sorted(missing_albums):
                logger.warning("  - Artist: '%s', Album: '%s'", artist or "<unknown>", album or "<unknown>")
    finally:
        # Always flush the cache, even on early dry-run exit or an unexpected error,
        # so an interrupted run keeps its progress.
        save_mb_cache()

# --------------- CLI ---------------

def main():
    ap = argparse.ArgumentParser(description="Sync GENRE tags using MusicBrainz tags only (no Spotify)")
    ap.add_argument("library", nargs="?", default=DEFAULT_LIBRARY_DIR)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not os.path.isdir(args.library):
        ap.error(f"{args.library} is not a directory")
    process_library(args.library, dry=args.dry_run)
    logger.info("Finished.")

if __name__ == "__main__":
    main()