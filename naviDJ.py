"""
AI-powered playlist generator for Subsonic/Navidrome music servers.

- Uses OpenAI or Ollama LLMs to generate playlists based on a vibe prompt.
- Prioritizes artists/genres from prompt and context playlists.
- Pushes generated playlists to the server.

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
from tqdm import tqdm
from openai import OpenAI
import argparse
from typing import List, Dict  # optional, only for type hints
import configparser
from rapidfuzz import fuzz
import logging

# Import embedding manager (optional dependency)
try:
    from embeddings import EmbeddingManager
except ImportError:
    EmbeddingManager = None

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
DEFAULT_OLLAMA_EMBEDDING_MODEL = secrets.get("ollama", "embedding_model", fallback="nomic-embed-text")
DEFAULT_OPENAI_EMBEDDING_MODEL = secrets.get("openai", "embedding_model", fallback="text-embedding-3-small")

SUBSONIC_BASE_URL = secrets.get("subsonic", "BASE_URL", fallback=None)
SUBSONIC_AUTH_PARAMS = {
    "u": secrets.get("subsonic", "USER", fallback=None),
    "p": secrets.get("subsonic", "PASSWORD", fallback=None),
    "v": secrets.get("subsonic", "API_VERSION", fallback="1.16.1"),
    "c": secrets.get("subsonic", "CLIENT", fallback="naviDJ"),
}

# Will be overwritten in `configure_llm` but need a placeholder so _llm_chat can be defined early.
LLM_MODE: str = DEFAULT_LLM_MODE  # 'openai' | 'ollama'
LLM_MODEL: str = DEFAULT_LLM_MODEL or ""  # auto‑filled later
client: OpenAI | None = None  # global client instance
embedding_mgr = None  # global embedding manager instance (optional)

# --------------------------------------------------
# HELPER FOR CLEANING LLM OUTPUT
# --------------------------------------------------

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)

def _remove_think_tags(text: str) -> str:
    """Strip <think>...</think> blocks (Ollama 'thinking' traces) and trim whitespace."""
    return _THINK_RE.sub("", text).strip()

def _split_artist_string(artist_string: str) -> list[str]:
    """
    Split a complex artist string into individual artist names.
    Handles various delimiters: ",", "•", ";", "feat.", "featuring", etc.
    Returns a list of cleaned individual artist names.
    """
    if not artist_string:
        return []
    
    # Normalize common delimiters
    normalized = artist_string.replace("•", ",").replace(";", ",").replace("feat.", ",").replace("featuring", ",")
    
    # Split by comma and clean each part
    artists = []
    for part in normalized.split(","):
        artist = part.strip()
        if artist and artist not in artists:  # Avoid duplicates
            artists.append(artist)
    
    return artists

# --------------------------------------------------
# LLM CLIENT INITIALISATION
# --------------------------------------------------

def configure_embeddings(api_type: str, model_name: str, base_url: str = None, api_key: str = None) -> None:
    """
    Initialize the global embedding manager.
    
    Args:
        api_type: Type of API to use ("ollama" or "openai")
        model_name: Name of the embedding model
        base_url: Base URL for Ollama or OpenAI (optional for OpenAI, defaults to https://api.openai.com/v1)
        api_key: API key for OpenAI (required for OpenAI, not used for Ollama)
    """
    global embedding_mgr
    
    if EmbeddingManager is None:
        logging.warning("EmbeddingManager not available. Install required dependencies to use embeddings.")
        embedding_mgr = None
        return
    
    try:
        embedding_mgr = EmbeddingManager(
            api_type=api_type,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key
        )
        logging.info(f"Embedding manager initialized with {api_type} model: {model_name}")
    except Exception as e:
        logging.warning(f"Failed to initialize embedding manager: {e}. Continuing without embeddings.")
        embedding_mgr = None

def configure_llm(mode: str = None, model: str = None) -> None:
    """Initialise the global `client`, `LLM_MODE`, and `LLM_MODEL` based on *mode* and *model*."""
    global LLM_MODE, LLM_MODEL, client, embedding_mgr

    # Use secrets.txt defaults if not provided
    mode = (mode or DEFAULT_LLM_MODE or "openai").lower()
    model = model or DEFAULT_LLM_MODEL or "gpt-4o-mini"

    if mode not in {"openai", "ollama", "custom"}:
        raise ValueError("Unsupported LLM_MODE. Choose 'openai', 'ollama', or 'custom'.")

    LLM_MODE = mode

    if mode == "openai":
        LLM_MODEL = model or "gpt-4o-mini"
        client = OpenAI(api_key=DEFAULT_OPENAI_KEY)
        # Initialize embedding manager for OpenAI
        if DEFAULT_OPENAI_KEY:
            configure_embeddings("openai", DEFAULT_OPENAI_EMBEDDING_MODEL, api_key=DEFAULT_OPENAI_KEY)
    elif mode == "ollama":
        LLM_MODEL = model or "gemma3n:latest"  # e.g. 'llama3' or 'mistral-7b-instruct'
        # Ollama's shim at /v1 is OpenAI‑compatible
        client = OpenAI(api_key=DEFAULT_OLLAMA_API_KEY, base_url=DEFAULT_OLLAMA_BASE)
        # Initialize embedding manager for Ollama
        if DEFAULT_OLLAMA_BASE:
            configure_embeddings("ollama", DEFAULT_OLLAMA_EMBEDDING_MODEL, base_url=DEFAULT_OLLAMA_BASE, api_key=DEFAULT_OLLAMA_API_KEY)
    else:  # custom
        LLM_MODEL = model or "gpt-4o-mini"
        client = OpenAI(api_key=DEFAULT_CUSTOM_API_KEY, base_url=DEFAULT_CUSTOM_BASE_URL)
        embedding_mgr = None  # Embeddings not supported with custom API

# --------------------------------------------------
# SUBSONIC HELPERS
# --------------------------------------------------

def fetch_starred_ids() -> set[str]:
    resp = requests.get(
        f"{SUBSONIC_BASE_URL}/getStarred2.view", params={**SUBSONIC_AUTH_PARAMS, "f": "json"}
    )
    resp.raise_for_status()
    songs = resp.json()["subsonic-response"]["starred2"].get("song", [])
    return {s["id"] for s in songs}


def fetch_all_subsonic_songs() -> list[dict]:
    all_songs: list[dict] = []
    song_offset, song_count = 0, 500
    bar = tqdm(desc="Fetching songs", unit="song", dynamic_ncols=True)

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
                }
            )
        bar.update(len(songs))
        if len(songs) < song_count:
            break
        song_offset += song_count
    bar.close()
    return all_songs

# --------------------------------------------------
# LLM UTILITIES  –  artist & genre selection
# --------------------------------------------------

def _llm_chat(messages: list[dict]) -> str:
    """Universal chat helper that works for both OpenAI & Ollama and always returns clean JSON-only content."""

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            stream=False,
            response_format={"type": "json_object"}
        )
    except TypeError:
        # Fallback for older openai‑python that doesn't know `response_format`.
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            stream=False,
        )
        
        # Extract and log token usage
        if resp.usage:
            logging.info(f"LLM Token Usage - Prompt: {resp.usage.prompt_tokens}, Completion: {resp.usage.completion_tokens}, Total: {resp.usage.total_tokens}")
        else:
            logging.info("LLM Token Usage: Not available in response.")

 
    # Remove any Ollama <think>...</think> traces *before* returning content.
    content = resp.choices[0].message.content
    return _remove_think_tags(content)


def _strip_fences(text: str) -> str:
    text = _remove_think_tags(text)
    if text.startswith("```"):
        _, rest = text.split("\n", 1)
        if rest.rstrip().endswith("```"):
            rest = rest.rstrip()[:-3]
        return rest.strip()
    return text


def _llm_select_items(prompt: str, items: list[str], n: int, label: str) -> list[str]:
    """Internal function that performs LLM-based selection. Returns JSON-only; caller can safely json.loads() the result."""
    system_msg = {
        "role": "system",
        "content": (
            f"You are a music‑curation assistant. Given a vibe prompt and a list of {label}, "
            f"return exactly {n} {label} from the list that best fit the vibe, as a JSON array of strings.\n"
            f"You MUST ONLY select from the provided list. Do NOT invent or return any item not in the list.\n"
            f"If the prompt specifically mentions anything/anyone in the list, be SURE to include it, "
            f"but do not include ANY items that are not in the list. If it makes sense to terminate the "
            f"list before hitting {n} objects, do so.\n"
            f"If nothing matches, return an empty list.\n"
            f"Consider the whole list first and the vibe and then make your decisions.\n"
            f"Return **only** a JSON array of strings. No objects, keys, or comments."
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Vibe prompt: \"{prompt}\"\n\nAvailable {label} ({len(items)}): {items}\n\n"
            f"Respond with **ONLY** a JSON array of {n} strings (no extra keys, no prose)."
        ),
    }
    raw = _strip_fences(_llm_chat([system_msg, user_msg]))
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            # Only return items that are actually in the provided list
            valid = [x for x in arr if x in items]
            dropped = [x for x in arr if x not in items]
            if dropped:
                print(f"[WARN] LLM returned {len(dropped)} invalid {label}: {dropped}")
            return valid[:n]
    except Exception:
        pass
    return items[:n]


def select_top_items(prompt: str, items: list[str], n: int, label: str) -> list[str]:
    """
    Select top n items using LLM, with optional embedding-based pre-filtering.
    If embedding_mgr is available and items list has >200 items, pre-filter to top 200
    using embeddings before sending to LLM. Otherwise, use LLM directly on all items.
    """
    # Pre-filter with embeddings if available and list is large
    if embedding_mgr is not None and len(items) > 200:
        try:
            pre_filtered = embedding_mgr.find_similar(prompt, items, top_k=200)
            if pre_filtered:
                print(f"[INFO] Pre-filtered {len(items)} {label} to {len(pre_filtered)} using embeddings")
                return _llm_select_items(prompt, pre_filtered, n, label)
            else:
                # Fallback if embedding filtering failed
                logging.warning("Embedding pre-filtering returned empty, using all items")
        except Exception as e:
            logging.warning(f"Embedding pre-filtering failed: {e}. Using all items.")
    
    # Use LLM directly (either no embeddings available, or list is small)
    return _llm_select_items(prompt, items, n, label)

# --------------------------------------------------
# PLAYLIST GENERATION
# --------------------------------------------------

def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def generate_playlist(
    prompt: str,
    songs: list[dict],
    *,
    chunk_size: int,
    required_ratio: float,
    existing_playlists: list[dict] = [],
) -> list[dict]:
    """
    Generate playlist using LLM, with optional embedding-based pre-filtering.
    If embedding_mgr is available, pre-filter songs to ~60% of original size using embeddings
    before chunking and processing. Returns JSON-only; caller can safely json.loads() the result.
    """
    # Pre-filter with embeddings if available
    songs_to_use = songs
    if embedding_mgr is not None:
        try:
            # Create text representations: "song title by artist name - genre"
            song_texts = []
            for song in songs:
                title = song.get("title", "")
                artist = song.get("artist", "")
                genre = song.get("genre", "")
                text = f"{title} by {artist}"
                if genre:
                    text += f" - {genre}"
                song_texts.append(text)
            
            # Pre-filter to ~60% of original size, but capped at 400 items to keep LLM workload reasonable
            target_size = min(len(songs), 400)
            if len(songs) > target_size:
                similar_indices = embedding_mgr.find_similar_indices(prompt, song_texts, top_k=target_size)
                if similar_indices:
                    songs_to_use = [songs[i] for i in similar_indices]
                    print(f"[INFO] Pre-filtered {len(songs)} songs to {len(songs_to_use)} using embeddings")
                else:
                    logging.warning("Embedding pre-filtering returned empty, using all songs")
        except Exception as e:
            logging.warning(f"Embedding pre-filtering failed: {e}. Using all songs.")
    
    playlist = []
    bar = tqdm(total=len(songs_to_use), desc="Building playlist", unit="song", dynamic_ncols=True)

    for chunk in chunk_list(songs_to_use, chunk_size):
        chunk_json = json.dumps(chunk).replace("```", "`\u200c`\u200c`")

        # How many songs must this chunk contribute?
        min_needed = max(1, math.ceil(len(chunk) * required_ratio))

        # Randomly chosen diversity options to slightly bias the selection a little differently each time.
        diversity_options = ["Slightly prefer songs where 'starred' IS true, as long as they fit the vibe.",
                             "Slightly prefer songs where 'starred' IS NOT true, as long as they fit the vibe.",
                             "Slightly prefer songs from MORE popular artists, as long as they fit the vibe.",
                             "Slightly prefer songs from LESS popular artists, as long as they fit the vibe."]

        sys_msg = {
            "role": "system",
            "content": (
                "You are a playlist‑builder AI.\n"
                "Rules (apply across the FINAL playlist, not just this chunk):\n"
                f"• {random.choice(diversity_options)} This is not a requirement.\n"
                "• Ensure artist diversity – ensure a healthy mix of all artists you see, UNLESS specific artists have been requested.\n"
                "• Obey the guideline and output schema exactly.\n"
                "• Return exactly one JSON object like:\n"
                '{"playlist": [{{"id": "…", "title": "…"}}, …]}\n'
                f"• **Include AT LEAST {min_needed} song(s) from this chunk.**\n"
                "• No other keys, arrays, or commentary.\n"
                "• Do not wrap your response in triple-backticks.\n"
            ),
        }

        user_msg = {
            "role": "user",
            "content": (
                f"Vibe prompt: {prompt}\n\nSelect songs from the given options. Return the song id and titles of songs to include EXACTLY as provided in the list of available songs.\n"
                f"Available songs: {chunk_json}\n\nReply with JSON **only** in the format described above."
            ),
        }

        raw = _strip_fences(_llm_chat([sys_msg, user_msg]))
        try:
            parsed = json.loads(raw)
            picked = parsed.get("playlist", [])

            # If the model still under-delivered, pad with random extras from this chunk
            if len(picked) < min_needed:
                remaining = [
                    s for s in chunk
                    if s["id"] not in {p["id"] for p in picked}
                ]
                random.shuffle(remaining)
                picked.extend(
                    {"id": s["id"], "title": s["title"]}
                    for s in remaining[: (min_needed - len(picked))]
                )

            playlist.extend(picked)
        except Exception as e:
            print(f"WARNING: JSON parse error – {e}. Raw start: {raw[:120]}")
        bar.update(len(chunk))
    bar.close()

    return playlist

# --------------------------------------------------
# PLAYLIST PUSH/UPDATE HELPERS
# --------------------------------------------------

def _update_playlist_on_server(name: str, song_ids: list[str], description: str) -> bool:
    pl_resp = requests.get(f"{SUBSONIC_BASE_URL}/getPlaylists", params=SUBSONIC_AUTH_PARAMS)
    pl_resp.raise_for_status()
    root = ET.fromstring(pl_resp.content)
    ns = root.tag.split("}")[0] + "}"
    plid = next((pl.get("id") for pl in root.findall(f".//{ns}playlist") if pl.get("name") == name), None)

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
        timeout=60
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

def select_context_playlist_songs(prompt: str, existing_playlists: list[dict], all_songs: list[dict]) -> list[dict]:
    """Ask the LLM to pick ONE playlist by name using a strict JSON contract. Returns JSON-only; caller can safely json.loads() the result."""
    if not existing_playlists:
        return []
    playlist_names = [pl["name"] for pl in existing_playlists]
    playlists_json = json.dumps(playlist_names).replace("```", "`\u200c`\u200c`")
    sys_msg = {
        "role": "system",
        "content": (
            "You are a playlist-builder AI.\n"
            "Given a vibe prompt and a list of **playlist names only**, select **one** playlist that best matches the vibe "
            "or was explicitly requested.\n\n"
            "• Reply with a **single JSON object** exactly like: {\"playlist_name\": \"Chosen Playlist\"}\n"
            "• If none are relevant, reply with an **empty object**: {}\n"
            "• Do **not** return comments, arrays, or extra keys.\n"
            "• Your entire reply **must** be valid JSON. Do not wrap in ``` fences."
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Vibe prompt: {prompt}\n\n"
            f"Available playlist names: {playlists_json}"
        ),
    }
    raw = _strip_fences(_llm_chat([sys_msg, user_msg]))
    try:
        parsed = json.loads(raw)
        playlist_name = parsed.get("playlist_name", "").strip()
    except Exception as e:
        print(f"WARNING: JSON parse error – {e}. Raw start: {raw[:120]}")
        playlist_name = ""
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

def ensure_min_songs(playlist: list[dict], candidates: list[dict], min_songs: int, max_songs: int = 50) -> list[dict]:
    if len(playlist) >= min_songs:
        return playlist[:max_songs]
    needed = min(min_songs - len(playlist), max_songs - len(playlist))
    remaining = [s for s in candidates if s["id"] not in {p["id"] for p in playlist}]
    random.shuffle(remaining)
    playlist.extend({"id": s["id"], "title": s["title"]} for s in remaining[:needed])
    print(f"Added {len(remaining[:needed])} random songs from filtered options to reach minimum length of {min_songs}.")
    return playlist[:max_songs]

# --------------------------------------------------
# PLAYLIST ENTRY SANITISER
# --------------------------------------------------

def _sanitize_playlist(entries: List[dict], candidates: List[dict], fuzzy_threshold: int = 90) -> List[dict]:
    """
    Ensure each playlist entry has an 'id'. If an entry only has a 'title'
    (and optionally 'artist'), try to resolve the matching song in *candidates*
    via a case-insensitive title-and-artist match. If that fails, use fuzzy matching
    on title and artist. Drop any rows we can’t resolve. This is backend-agnostic and
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
            title_score = fuzz.ratio((e.get("title") or "").lower(), (s.get("title") or "").lower())
            artist_score = fuzz.ratio((e.get("artist") or "").lower(), (s.get("artist") or "").lower())
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

STOPWORDS = {'and', 'but', 'mix', 'the', 'a', 'an', 'of', 'in', 'on', 'for', 'to', 'from', 'by', 'with', 'at', 'as', 'is', 'it', 'or', 'vs', 'feat', 'featuring'}

def extract_prompt_entities(prompt: str, all_artists: list[str], all_genres: list[str]) -> dict:
    """
    Extract artists and genres mentioned in the prompt.
    Returns a dict with keys: 'artists', 'genres'.
    """
    entities = {'artists': [], 'genres': []}
    prompt_lc = prompt.lower()
    # Artists
    for artist in all_artists:
        if artist.lower() in prompt_lc:
            entities['artists'].append(artist)
    # Genres
    for genre in all_genres:
        if genre.lower() in prompt_lc:
            entities['genres'].append(genre)
    return entities

# New function to select relevant albums using LLM

def select_relevant_albums_llm(prompt: str, all_albums: list[str], n: int = 5) -> list[str]:
    """
    Use the LLM to select up to n relevant albums from all_albums for the given prompt.
    If embedding_mgr is available and album list has >50 items, pre-filter to 50 using embeddings.
    The LLM must return a JSON object: {"albums": [ ... ]} (never a list or dict with other keys).
    Only albums in all_albums are valid. If the response is not valid, return [].
    """
    # Pre-filter with embeddings if available and list is large
    albums_to_use = all_albums
    if embedding_mgr is not None and len(all_albums) > 50:
        try:
            pre_filtered = embedding_mgr.find_similar(prompt, all_albums, top_k=50)
            if pre_filtered:
                print(f"[INFO] Pre-filtered {len(all_albums)} albums to {len(pre_filtered)} using embeddings")
                albums_to_use = pre_filtered
            else:
                # Fallback if embedding filtering failed
                logging.warning("Embedding pre-filtering returned empty, using all albums")
        except Exception as e:
            logging.warning(f"Embedding pre-filtering failed: {e}. Using all albums.")
    
    system_msg = {
        "role": "system",
        "content": (
            f"You are a music‑curation assistant. Given a vibe prompt and a list of album names, "
            f"return up to {n} album names from the list that are most relevant to the prompt, as a JSON object.\n"
            f"You MUST ONLY select from the provided list. Do NOT invent or return any item not in the list.\n"
            f"If the prompt specifically mentions anything in the list, be SURE to include it.\n"
            f"If nothing matches or seems relevant, return an empty list.\n"
            f"Return **only** a JSON object in this format: {{\"albums\": [ ... ]}}. No other keys, no comments, no prose.\n"
            f"Do NOT return a list, mapping, or dictionary with other keys. Only a JSON object with an 'albums' array."
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Vibe prompt: \"{prompt}\"\n\nAvailable albums ({len(albums_to_use)}): {albums_to_use}\n\n"
            f"Respond with **ONLY** a JSON object: {{\"albums\": [ ... ]}} (no extra keys, no prose)."
        ),
    }
    raw = _strip_fences(_llm_chat([system_msg, user_msg]))
    try:
        parsed = json.loads(raw)
        if not (isinstance(parsed, dict) and "albums" in parsed and isinstance(parsed["albums"], list)):
            print(f"[WARN] LLM did not return a valid albums object: {raw}")
            return []
        valid = [x for x in parsed["albums"] if x in all_albums]
        dropped = [x for x in parsed["albums"] if x not in all_albums]
        if dropped:
            print(f"[WARN] LLM returned {len(dropped)} invalid albums: {dropped}")
        return valid[:n]
    except Exception:
        print(f"[WARN] JSON parse error: {raw}")
        return []

# --------------------------------------------------
# MAIN (updated flow with context playlist)
# --------------------------------------------------

def _main_impl(args):
    playlist_name = args.playlist_name
    prompt = args.prompt or input("Enter a prompt for the playlist vibe: ").strip()
    if not prompt:
        print("Prompt required – exiting.")
        return

    print("Fetching library…")
    all_songs = fetch_all_subsonic_songs()
    if not all_songs:
        print("No songs found on the server.")
        return
    print("\n")

    print("Choosing context...")
    existing_playlists = fetch_all_playlists(exclude_name=playlist_name)
    # Select context playlist songs
    context_songs = select_context_playlist_songs(prompt, existing_playlists, all_songs)

    # Fetch full library artists/genres/albums
    all_artists = fetch_all_artists()
    random.shuffle(all_artists)
    all_genres = fetch_all_genres()
    random.shuffle(all_genres)
    all_albums = [a for a in {s.get('album') for s in all_songs if s.get('album')} if isinstance(a, str)]
    random.shuffle(all_albums)
    
    # Check if library size has increased and invalidate embedding cache if needed
    if embedding_mgr is not None:
        library_size = len(all_artists) + len(all_genres) + len(all_albums)
        embedding_mgr.check_library_size(library_size)


    print("Analyzing request...")
    # Extract prompt entities (artists, genres)
    prompt_entities = extract_prompt_entities(prompt, all_artists, all_genres)
    explicit_prompt_artists = prompt_entities['artists']
    explicit_prompt_genres = prompt_entities['genres']

    print("Connsidering albums...")
    # Use LLM to select up to 5 relevant albums
    relevant_albums = select_relevant_albums_llm(prompt, all_albums, n=5)
    album_artists = []
    album_genres = []
    if relevant_albums:
        album_songs = [s for s in all_songs if s.get('album') in relevant_albums]
        context_song_ids = {s['id'] for s in context_songs} if context_songs else set()
        new_album_songs = [s for s in album_songs if s['id'] not in context_song_ids]
        if context_songs:
            context_songs = context_songs + new_album_songs
        else:
            context_songs = new_album_songs
        # Collect artists and genres from relevant albums
        for s in album_songs:
            if s.get('artist'):
                individual_artists = _split_artist_string(s['artist'])
                album_artists.extend(individual_artists)
            if s.get('genre'):
                album_genres.append(s['genre'])
        album_artists = list(dict.fromkeys(album_artists))  # Remove duplicates
        album_genres = list(dict.fromkeys(album_genres))
        print("Focus albums: ", ", ".join(relevant_albums))

    # Extract artists/genres from context playlist (weighted highest)
    context_artists = []
    context_genres = []
    if context_songs:
        # Use the helper function to properly split artist strings
        for s in context_songs:
            if s.get("artist"):
                individual_artists = _split_artist_string(s["artist"])
                context_artists.extend(individual_artists)
        context_artists = list(dict.fromkeys(context_artists))  # Remove duplicates
        for s in context_songs:
            if s.get("genre"):
                raw = s["genre"].replace(";", ",").replace("•", ",")
                for g in raw.split(","):
                    name = g.strip()
                    if name:
                        context_genres.append(name)
        context_genres = list(dict.fromkeys(context_genres))

    # Continue as before, but use album_filtered_songs for focus selection
    num_artists: int = 30
    num_genres:  int = 50

    print("Choosing top artists and genres...")
    init_focus_artists = select_top_items(prompt, all_artists, num_artists, "artists")
    init_focus_genres  = select_top_items(prompt, all_genres, num_genres, "genres")

    # Combine context, prompt, and LLM-selected for focus
    combined_artists = list(dict.fromkeys(
        explicit_prompt_artists + album_artists + context_artists + init_focus_artists
    ))
    combined_genres = list(dict.fromkeys(
        explicit_prompt_genres + album_genres + context_genres + init_focus_genres 
    ))

    focus_artists = select_top_items(prompt, combined_artists, num_artists, "artists")
    print("Focus artists:", ", ".join(focus_artists))
    
    focus_genres = select_top_items(prompt, combined_genres, num_genres, "genres")
    print("Focus genres: ", ", ".join(focus_genres))

    # Filter library based on focus
    print("\nFiltering library...")
    filtered = []
    for s in all_songs:
        # Split the song's artist field and check if any individual artist matches
        song_artists = _split_artist_string(s.get("artist", ""))
        artist_match = any(artist in focus_artists for artist in song_artists)
        genre_match = s.get("genre") in focus_genres
        if artist_match and genre_match:
            filtered.append(s)
    # Also include all songs from relevant albums, avoiding duplicates
    if relevant_albums:
        filtered_ids = {s['id'] for s in filtered}
        album_songs = [s for s in all_songs if s.get('album') in relevant_albums and s['id'] not in filtered_ids]
        filtered_ids.update(s['id'] for s in album_songs)
        filtered.extend(album_songs)
    if not filtered and not context_songs:
        print("No songs match the selected artists/genres – aborting.")
        return

    # Combine filtered with context songs, no duplicates
    if context_songs:
        filtered_ids = {s["id"] for s in filtered}
        combined_songs = filtered + [s for s in context_songs if s["id"] not in filtered_ids]
    else:
        combined_songs = filtered

    random.shuffle(combined_songs)
    print(f"Generating playlist ({len(combined_songs)} candidate songs)…")

    # We need at least args.min_songs tracks in total → derive a ratio
    required_ratio = args.min_songs / max(1, len(combined_songs))

    # Get chunk size from configuration, default to 500
    chunk_size = int(secrets.get("llm", "chunk_size", fallback=500))
    playlist_items = generate_playlist(
        prompt,
        combined_songs,
        chunk_size=chunk_size,
        required_ratio=required_ratio,
    )
    print(f"Generated {len(playlist_items)} tracks from {len(combined_songs)} candidates.")
    playlist_items = ensure_min_songs(playlist_items, combined_songs, args.min_songs)

    # Resolve any entries missing an 'id' before upload (with fuzzy matching)
    playlist_items = _sanitize_playlist(playlist_items, combined_songs)

    # After sanitization, ensure we still meet the minimum song count
    playlist_items = ensure_min_songs(playlist_items, combined_songs, args.min_songs)

    if not playlist_items:
        print("LLM returned an empty playlist.")
        return

    print(f"Got {len(playlist_items)} tracks. Pushing to server…")
    song_ids = [t["id"] for t in playlist_items]
    if _update_playlist_on_server(playlist_name, song_ids, prompt):
        print(f"SUCCESS: Playlist '{playlist_name}' updated!")
    else:
        print("ERROR: Failed to update playlist on server.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a playlist based on a vibe prompt.")
    parser.add_argument("--playlist_name", type=str, default="naviDJ", help="Name of the playlist to create or update.")
    parser.add_argument("--prompt", type=str, help="Vibe prompt for the playlist.")
    parser.add_argument("--min_songs", type=int, default=35, help="Minimum number of songs in the playlist.")
    parser.add_argument("--llm_mode", type=str, choices=["openai", "ollama"], default=DEFAULT_LLM_MODE, help="Which LLM backend to use (overrides secrets.txt).")
    parser.add_argument("--llm_model", type=str, default=DEFAULT_LLM_MODEL, help="Which LLM model to use (overrides secrets.txt).")
    args = parser.parse_args()

    configure_llm(args.llm_mode, args.llm_model)
    _main_impl(args)
