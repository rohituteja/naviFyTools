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
from openai import OpenAI
import argparse
from typing import List, Dict  # optional, only for type hints
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
client: OpenAI | None = None  # global client instance
EMBEDDING_MODEL: str | None = None
embedding_manager: EmbeddingManager | None = None

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
    Handles various delimiters: ",", ".", ";", "feat.", "featuring", etc.
    Returns a list of cleaned individual artist names.
    """
    if not artist_string:
        return []
    
    # Normalize common delimiters
    normalized = artist_string.replace(",", ",").replace(";", ",").replace("feat.", ",").replace("featuring", ",")
    
    # Split by comma and clean each part
    artists = []
    for part in normalized.split(","):
        artist = part.strip()
        if artist and artist not in artists:  # Avoid duplicates
            artists.append(artist)
    
    return artists

def configure_llm(mode: str = None, model: str = None) -> None:
    """Initialise the global `client`, `LLM_MODE`, and `LLM_MODEL` based on *mode* and *model*."""
    global LLM_MODE, LLM_MODEL, client, EMBEDDING_MODEL, embedding_manager

    # Use secrets.txt defaults if not provided
    mode = (mode or DEFAULT_LLM_MODE or "openai").lower()
    
    # Reload secrets to get latest config (important for frontend updates)
    secrets.read(os.path.join(os.path.dirname(__file__), "secrets.txt"))

    if mode not in {"openai", "ollama", "custom"}:
        raise ValueError("Unsupported LLM_MODE. Choose 'openai', 'ollama', or 'custom'.")

    LLM_MODE = mode

    if mode == "openai":
        LLM_MODEL = model or DEFAULT_LLM_MODEL or "gpt-4o-mini"
        EMBEDDING_MODEL = secrets.get("openai", "embedding_model", fallback="text-embedding-3-small")
        client = OpenAI(api_key=DEFAULT_OPENAI_KEY)
        embedding_manager = EmbeddingManager(
            api_type="openai",
            model_name=EMBEDDING_MODEL,
            api_key=DEFAULT_OPENAI_KEY
        )
    elif mode == "ollama":
        LLM_MODEL = model or DEFAULT_LLM_MODEL or "gemma3n:latest" 
        EMBEDDING_MODEL = secrets.get("ollama", "embedding_model", fallback="nomic-embed-text:latest")
        client = OpenAI(api_key=DEFAULT_OLLAMA_API_KEY, base_url=DEFAULT_OLLAMA_BASE)
        embedding_manager = EmbeddingManager(
            api_type="ollama",
            model_name=EMBEDDING_MODEL,
            base_url=DEFAULT_OLLAMA_BASE,
            api_key=DEFAULT_OLLAMA_API_KEY
        )
    else:  # custom
        LLM_MODEL = model or DEFAULT_LLM_MODEL or "gpt-4o-mini"
        client = OpenAI(api_key=DEFAULT_CUSTOM_API_KEY, base_url=DEFAULT_CUSTOM_BASE_URL)
        embedding_manager = None

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

def _llm_chat(messages: list[dict]) -> str:
    """Universal chat helper that works for both OpenAI & Ollama and always returns clean JSON-only content."""

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            stream=False,
            response_format={"type": "json_object"}
        )
    except Exception:
        # Fallback for older openai-python or backends that don't know `response_format`.
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            stream=False,
        )
        
    content = resp.choices[0].message.content
    content = _remove_think_tags(content)
    
    # Robustly find the JSON object if it's buried in text
    if "{" in content and "}" in content:
        start = content.find("{")
        end = content.rfind("}") + 1
        content = content[start:end]
        
    return content


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
    text = re.sub(r',\s*([\]}])', r'\1', text)
    return text


def select_focus_metadata_single_call(
    prompt: str,
    all_artists: list[str],
    all_genres: list[str],
    all_albums: list[str]
) -> dict[str, list[str]]:
    """
    Single LLM call to select relevant artists, genres, and albums together.
    """
    system_msg = {
        "role": "system",
        "content": (
            "You are a music metadata curator. Your task is to select focused metadata items for a given vibe prompt.\n"
            "You MUST select items EXACTLY as they appear in the provided lists. Do NOT modify capitalization, spelling, or formatting.\n"
            "Return ONLY items that exist in the library.\n\n"
            "Selection targets:\n"
            "- Artists: Exactly 5 artists that fit the vibe\n"
            "- Genres: Exactly 10 relevant genres\n"
            "- Albums: Exactly 5 representative albums\n\n"
            "Return ONLY valid JSON in this exact format:\n"
            '{\n'
            '  "artists": ["artist1", "artist2", ...],\n'
            '  "genres": ["genre1", "genre2", ...],\n'
            '  "albums": ["album1", "album2", ...]\n'
            '}\n\n'
            "Rules:\n"
            "- ONLY select from the provided lists. Do NOT invent new items.\n"
            "- Use the EXACT string representation.\n"
            "- If the prompt explicitly mentions an item, ALWAYS include it if it exists."
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
    
    raw = _llm_chat([system_msg, user_msg])
    raw = _clean_json(_strip_fences(raw))
    
    try:
        parsed = json.loads(raw)
        result = {
            "artists": [x for x in parsed.get("artists", []) if x in all_artists],
            "genres": [x for x in parsed.get("genres", []) if x in all_genres],
            "albums": [x for x in parsed.get("albums", []) if x in all_albums]
        }
        
        return result
    except Exception as e:
        print(f"[ERROR] Metadata selection failed: {e}")
        return {"artists": all_artists[:10], "genres": all_genres[:15], "albums": all_albums[:5]}

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
    semantic_song_ids: set[str] = None
) -> list[dict]:
    """
    Filter library based on combined focus items with hierarchical, additive weighting.
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
        if any(a in exp_a for a in song_artists): score += 3.0
        if song_genre in exp_g: score += 3.0
        if song_album in exp_al: score += 3.0
            
        # 2. Context Playlist Matches
        # Explicitly in context playlist (+4.0)
        if song_id in ctx_ids:
            score += 4.0
        # Metadata matches if part of context (+2.0 each)
        if any(a in ctx_a for a in song_artists): score += 2.0
        if song_genre in ctx_g: score += 2.0
        if song_album in ctx_al: score += 2.0
            
        # 3. LLM Chosen Focus Matches (+1.5 each)
        if any(a in sel_a for a in song_artists): score += 1.5
        if song_genre in sel_g: score += 1.5
        if song_album in sel_al: score += 1.5
            
        # 4. Starred/Favorited Boost (+1.0)
        if s.get("starred"):
            score += 1.0

        # 5. Semantic Vibe Matches (+0.5)
        if song_id in sem_ids:
            score += 0.5

        if score > 0:
            s_copy = s.copy()
            s_copy["_relevance_score"] = score
            filtered.append(s_copy)

    # Sort by relevance
    filtered.sort(key=lambda x: x.get("_relevance_score", 0), reverse=True)
    
    # Cap candidate pool at Top 1000
    if len(filtered) > 1000:
        filtered = filtered[:1000]
        print(f"Capped candidate pool to top 1000 songs.")
    
    return filtered



def generate_playlist_single_call(
    prompt: str,
    filtered_songs: list[dict],
    min_songs: int,
    explicit_artists: list[str] = None
) -> list[dict]:
    """
    Generate playlist with a single LLM call. Improved parsing and fallback.
    """
    explicit_artist_str = ", ".join(explicit_artists or [])
    
    # Prepare songs (simplified for LLM to avoid context bloat)
    songs_for_llm = [
        {"id": s["id"], "title": s["title"], "artist": s.get("artist", "Unknown")}
        for s in filtered_songs
    ]
    songs_json = json.dumps(songs_for_llm)
    
    diversity_options = [
        "Slightly prefer starred songs if they fit.",
        "Ensure artist diversity unless specific artists were requested.",
        "Create a cohesive flow between tracks."
    ]

    system_msg = {
        "role": "system",
        "content": (
            "You are a playlist-builder AI.\n"
            "Rules:\n"
            f"• {random.choice(diversity_options)}\n"
            "• Maintain high artist and album diversity. Do NOT cluster multiple tracks from the same album unless the prompt explicitly justifies an album-focused selection.\n"
            "• Ensure healthy artist/genre mix.\n"
            "• Return exactly ONE JSON object like:\n"
            '  {"playlist": [{"id": "...", "title": "..."}, ...]}\n'
            f"• Include exactly {min_songs} songs.\n"
            "• Use ONLY the IDs and titles provided below.\n"
            "• Do NOT wrap in triple-backticks. Respond with JSON ONLY."
        ),
    }
    
    user_msg = {
        "role": "user",
        "content": f"Vibe: {prompt}\n\nAvailable Songs:\n{songs_json}"
    }
    
    raw = _strip_fences(_llm_chat([system_msg, user_msg]))
    
    try:
        parsed = json.loads(raw)
        picked = parsed.get("playlist", [])
        if not picked and isinstance(parsed, list):
            picked = parsed

        # Resolve IDs (robust matching)
        id_to_song = {s["id"]: s for s in filtered_songs}
        playlist = []
        for item in picked:
            if isinstance(item, str): # Just an ID
                sid = item
            else: # Dictionary
                sid = item.get("id")
            
            if sid and sid in id_to_song:
                playlist.append({"id": sid, "title": id_to_song[sid]["title"]})
            else:
                # Try title match as fallback
                title = item.get("title", "").lower() if isinstance(item, dict) else ""
                if title:
                    for s in filtered_songs:
                        if s["title"].lower() == title:
                            playlist.append({"id": s["id"], "title": s["title"]})
                            break
        
        # Pad if needed
        if len(playlist) < min_songs:
            playlist = ensure_min_songs(playlist, filtered_songs, min_songs)
            
        return playlist
        
    except Exception as e:
        print(f"[ERROR] Playlist generation failed: {e}. Raw: {raw[:100]}")
        return [{"id": s["id"], "title": s["title"]} for s in filtered_songs[:min_songs]]


def generate_playlist_chunked(
    prompt: str,
    filtered_songs: list[dict],
    min_songs: int,
    chunk_size: int = 200,
    explicit_artists: list[str] = None
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
            explicit_artists=explicit_artists
        )
    
    # Split into chunks, maintaining score order (already sorted)
    num_chunks = math.ceil(len(filtered_songs) / chunk_size)
    print(f"Processing {len(filtered_songs)} songs in {num_chunks} chunks of {chunk_size}")
    
    all_selections = []
    songs_per_chunk = math.ceil(min_songs / num_chunks)
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(filtered_songs))
        chunk = filtered_songs[start_idx:end_idx]
        
        print(f"Chunk {i+1}/{num_chunks}: {len(chunk)} songs")
        
        # Request proportional number of songs from this chunk
        chunk_target = min(songs_per_chunk, len(chunk))
        
        try:
            chunk_playlist = generate_playlist_single_call(
                prompt=prompt,
                filtered_songs=chunk,
                min_songs=chunk_target,
                explicit_artists=explicit_artists
            )
            all_selections.extend(chunk_playlist)
        except Exception as e:
            print(f"[WARN] Chunk {i+1} failed: {e}")
            # Fallback: take top-scored songs from this chunk
            fallback = [{"id": s["id"], "title": s["title"]} for s in chunk[:chunk_target]]
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
        unique_selections = ensure_min_songs(unique_selections, filtered_songs, min_songs)
    
    return unique_selections[:min_songs]


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
            "Rules:\n"
            "  - Reply with a single JSON object exactly like: {\"playlist_name\": \"Chosen Playlist\"}\n"
            "  - If none are relevant, reply with an empty object: {}\n"
            "  - Do not return comments, arrays, or extra keys.\n"
            "  - Your entire reply must be valid JSON. Do not wrap in backticks."
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
        print(f"WARNING: JSON parse error - {e}. Raw start: {raw[:120]}")
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

def extract_prompt_entities(prompt: str, all_artists: list[str], all_genres: list[str], all_albums: list[str]) -> dict:
    """
    Extract artists, genres, and albums mentioned in the prompt using smart partial matching.
    Handles partial names like 'gambino' -> 'Childish Gambino'.
    Returns a dict with keys: 'artists', 'genres', 'albums'.
    """
    entities = {'artists': [], 'genres': [], 'albums': []}
    
    # Clean prompt: remove stopwords and punctuation, split into words
    prompt_lc = prompt.lower()
    stopwords = {'and', 'the', 'a', 'an', 'of', 'in', 'on', 'for', 'to', 'from', 'by', 'with', 'at', 'as', 'is', 'it', 'or'}
    prompt_words = set(re.findall(r'\b\w+\b', prompt_lc)) - stopwords
    
    # Helper to check matches in prompt
    def check_matches(items, key):
        for item in items:
            item_lc = item.lower()
            item_words = set(re.findall(r'\b\w+\b', item_lc)) - stopwords
            
            # Exact match (full item name in prompt)
            if item_lc in prompt_lc:
                entities[key].append(item)
            # Partial match (any significant word from item name in prompt)
            elif item_words and item_words & prompt_words:  # Intersection
                entities[key].append(item)

    # Artists: check both exact and partial matches
    check_matches(all_artists, 'artists')
    
    # Albums: check both exact and partial matches
    check_matches(all_albums, 'albums')
    
    # Genres: exact match only (genres are typically single words or short phrases)
    for genre in all_genres:
        if genre.lower() in prompt_lc:
            entities['genres'].append(genre)
    
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
    print("="*60)

    # ========== STAGE 0: LIBRARY FETCH ==========
    start_t = time.time()
    all_songs = fetch_all_subsonic_songs()
    if not all_songs:
        print("No songs found on the server.")
        return
    print(f"Library fetch complete: {len(all_songs)} songs ({time.time()-start_t:.1f}s)")

    # ========== STAGE 1: METADATA GATHERING ==========
    start_t = time.time()
    all_artists = fetch_all_artists()
    all_genres = fetch_all_genres()
    all_albums = [a for a in {s.get('album') for s in all_songs if s.get('album')} if isinstance(a, str)]
    
    # Randomize to avoid bias
    random.shuffle(all_artists)
    random.shuffle(all_genres)
    random.shuffle(all_albums)
    print(f"Gathered metadata: {len(all_artists)} artists, {len(all_genres)} genres, {len(all_albums)} albums ({time.time()-start_t:.1f}s)")

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
        sem_albums = embedding_manager.find_similar(prompt, all_albums, top_k=40)
        
        # 2. Semantic Song Pre-selection (Top 200)
        print("Finding semantically similar songs...")
        song_texts = [f"{s.get('title','')} by {s.get('artist','')}" for s in all_songs]
        sem_song_indices = embedding_manager.find_similar_indices(prompt, song_texts, top_k=200)
        semantic_song_ids = {all_songs[idx]["id"] for idx in sem_song_indices if idx < len(all_songs)}
        
        print(f"Semantic pre-filtering complete ({time.time()-start_t:.1f}s)")
    else:
        print("Skipping semantic pre-filtering (no embedding manager initialized).")

    # ========== CONTEXT ANALYSIS ==========
    start_t = time.time()
    existing_playlists = fetch_all_playlists(exclude_name=playlist_name)
    context_songs = select_context_playlist_songs(prompt, existing_playlists, all_songs)
    
    # Extract metadata from context
    context_artists = []
    context_genres = []
    context_albums = []
    if context_songs:
        for s in context_songs:
            if s.get("artist"):
                context_artists.extend(_split_artist_string(s["artist"]))
            if s.get("genre"):
                context_genres.append(s["genre"])
            if s.get("album"):
                context_albums.append(s["album"])
        context_artists = list(dict.fromkeys(context_artists))
        context_genres = list(dict.fromkeys(context_genres))
        context_albums = list(dict.fromkeys(context_albums))
    
    # Extract explicit mentions from prompt
    prompt_entities = extract_prompt_entities(prompt, all_artists, all_genres, all_albums)
    explicit_artists = prompt_entities['artists']
    explicit_genres = prompt_entities['genres']
    explicit_albums = prompt_entities['albums']
    
    print(f"Context analysis complete ({time.time()-start_t:.1f}s)")
    
    if explicit_artists:
        print(f"Explicit artists identified: {', '.join(explicit_artists)}")
    if explicit_genres:
        print(f"Explicit genres identified: {', '.join(explicit_genres)}")

    # ========== STAGE 1: METADATA SELECTION ==========
    start_t = time.time()
    selected_metadata = select_focus_metadata_single_call(
        prompt=prompt,
        all_artists=sem_artists,
        all_genres=sem_genres,
        all_albums=sem_albums
    )
    
    duration = time.time() - start_t
    
    # Display selected metadata (for debugging)
    all_artists_combined = list(dict.fromkeys(explicit_artists + context_artists + selected_metadata["artists"]))
    all_genres_combined = list(dict.fromkeys(explicit_genres + context_genres + selected_metadata["genres"]))
    all_albums_combined = list(dict.fromkeys(context_albums + selected_metadata["albums"]))
    
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
        semantic_song_ids=semantic_song_ids
    )
    print(f"Final candidate pool: {len(candidate_pool)} songs ({time.time()-start_t:.1f}s)")
    
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
        explicit_artists=explicit_artists
    )
    print(f"Final playlist generated: {len(playlist_items)} tracks ({time.time()-start_t:.1f}s)")

    # Sanitize playlist (resolve any missing IDs)
    playlist_items = _sanitize_playlist(playlist_items, candidate_pool)

    if not playlist_items:
        print("\nFailed to generate playlist.")
        return

    # ========== STAGE 4: UPLOAD TO SERVER ==========
    start_t = time.time()
    song_ids = [t["id"] for t in playlist_items]
    success = _update_playlist_on_server(playlist_name, song_ids, prompt)
    
    if success:
        print(f"Playlist '{playlist_name}' successfully updated on server ({time.time()-start_t:.1f}s)")
    else:
        print("ERROR: Failed to update playlist on server.")
    
    print("\nComplete!")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a playlist based on a vibe prompt.")
    parser.add_argument("--playlist_name", type=str, default="naviDJ", help="Name of the playlist to create or update.")
    parser.add_argument("--prompt", type=str, help="Vibe prompt for the playlist.")
    parser.add_argument("--min_songs", type=int, default=35, help="Minimum number of songs in the playlist.")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Number of songs per LLM chunk (adjust based on context size).")
    parser.add_argument("--llm_mode", type=str, choices=["openai", "ollama"], default=DEFAULT_LLM_MODE, help="Which LLM backend to use (overrides secrets.txt).")
    parser.add_argument("--llm_model", type=str, default=DEFAULT_LLM_MODEL, help="Which LLM model to use (overrides secrets.txt).")
    args = parser.parse_args()

    configure_llm(args.llm_mode, args.llm_model)
    _main_impl(args)
