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

# --------------------------------------------------
# CONFIG & LLM CLIENT SETUP
# --------------------------------------------------

# Load secrets from secrets.txt
secrets = configparser.ConfigParser()
secrets.read(os.path.join(os.path.dirname(__file__), "secrets.txt"))

DEFAULT_OPENAI_KEY = secrets.get("openai", "OPENAI_KEY", fallback=None)
DEFAULT_OLLAMA_BASE = secrets.get("ollama", "OLLAMA_BASE", fallback=None)

SUBSONIC_BASE_URL = secrets.get("subsonic", "BASE_URL", fallback=None)
SUBSONIC_AUTH_PARAMS = {
    "u": secrets.get("subsonic", "USER", fallback=None),
    "p": secrets.get("subsonic", "PASSWORD", fallback=None),
    "v": secrets.get("subsonic", "API_VERSION", fallback="1.16.1"),
    "c": secrets.get("subsonic", "CLIENT", fallback="naviDJ"),
}

# Will be overwritten in `configure_llm` but need a placeholder so _llm_chat can be defined early.
LLM_MODE: str = "openai"  # 'openai' | 'ollama'
LLM_MODEL: str = ""  # auto‑filled later
client: OpenAI | None = None  # global client instance

# --------------------------------------------------
# HELPER FOR CLEANING LLM OUTPUT
# --------------------------------------------------

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)

def _remove_think_tags(text: str) -> str:
    """Strip <think>...</think> blocks (Ollama 'thinking' traces) and trim whitespace."""
    return _THINK_RE.sub("", text).strip()

# --------------------------------------------------
# LLM CLIENT INITIALISATION
# --------------------------------------------------

def configure_llm(mode: str) -> None:
    """Initialise the global `client`, `LLM_MODE`, and `LLM_MODEL` based on *mode*."""
    global LLM_MODE, LLM_MODEL, client

    mode = mode.lower()
    if mode not in {"openai", "ollama"}:
        raise ValueError("Unsupported LLM_MODE. Choose 'openai' or 'ollama'.")

    LLM_MODE = mode

    if mode == "openai":
        LLM_MODEL = "gpt-4o-mini"
        client = OpenAI(api_key=DEFAULT_OPENAI_KEY)
    else:  # ollama
        LLM_MODEL = "deepseek-r1:latest"  # e.g. 'llama3' or 'mistral-7b-instruct'
        # Ollama’s shim at /v1 is OpenAI‑compatible; any string works as the key.
        client = OpenAI(api_key="ollama", base_url=DEFAULT_OLLAMA_BASE)

# --------------------------------------------------
# SUBSONIC HELPERS  (unchanged)
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

    extra_args: dict[str, object] = {}

    # if LLM_MODE == "ollama":
        # Local models tend to ramble; keep them concise.
        # extra_args["temperature"] = 0.2
        #
        # ⚠️  Do NOT pass non-standard keys like `think` or `options` here – the
        # OpenAI-python client validates arguments locally and will raise
        # TypeError before any request reaches the Ollama server.  If you need
        # advanced generation control, call Ollama’s native /generate endpoint
        # or encode parameters in the model name (e.g.  "qwen3:4b?num_predict=1024").

    # Ask for structured JSON output whenever the backend supports it.
    extra_args["response_format"] = {"type": "json_object"}

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            stream=False,
            **extra_args,
        )
    except TypeError:
        # Fallback for older openai‑python that doesn't know `response_format`.
        extra_args.pop("response_format", None)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            stream=False,
            **extra_args,
        )

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


def select_top_items(prompt: str, items: list[str], n: int, label: str) -> list[str]:
    """Returns JSON-only; caller can safely json.loads() the result."""
    system_msg = {
        "role": "system",
        "content": (
            f"You are a music‑curation assistant. Given a vibe prompt and a list of {label}, "
            f"return up to {n} {label} from the list that best fit the vibe, as a JSON array of strings."
            f"Print the objects EXACTLY as listed in the input you are given."
            f"If the prompt specifically mentions anything/anyone in the list, be SURE to include it, "
            f"but do not include ANY items that are not in the list. If it makes sense to terminate the "
            f"list before hitting {n} objects, do so.\n"
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
            return arr[:n]
    except Exception:
        pass
    return items[:n]

# --------------------------------------------------
# PLAYLIST GENERATION – updated prompt
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
    existing_playlists: list[dict] = None,
) -> list[dict]:
    """Returns JSON-only; caller can safely json.loads() the result."""
    playlist = []
    bar = tqdm(total=len(songs), desc="Building playlist", unit="song", dynamic_ncols=True)

    for chunk in chunk_list(songs, chunk_size):
        chunk_json = json.dumps(chunk).replace("```", "`\u200c`\u200c`")

        # 🎯  How many songs must this chunk contribute?
        min_needed = max(1, math.ceil(len(chunk) * required_ratio))

        sys_msg = {
            "role": "system",
            "content": (
                "You are a playlist‑builder AI.\n"
                "Rules (apply across the FINAL playlist, not just this chunk):\n"
                "• Slightly prefer songs where 'starred' is true, as long as they fit the vibe. This is not a requirement.\n"
                "• Ensure artist diversity – ensure a healthy mix of all artists, unless a specific one has been requested.\n"
                "• Obey the guideline and output schema exactly.\n"
                "• Return exactly one JSON object like:\n"
                '{"playlist": [{{"id": "…", "title": "…"}}, …]}\n'
                f"• **Include at least {min_needed} song(s) from this chunk.**\n"
                "• No other keys, arrays, or commentary.\n"
                "• Do not wrap your response in triple-backticks.\n"
            ),
        }

        user_msg = {
            "role": "user",
            "content": (
                f"Vibe prompt: {prompt}\n\nSelect any songs from the list below, honoring the rules above. Return the song id and titles EXACTLY as provided in the list of available songs.\n"
                f"Available songs: {chunk_json}\n\nReply with JSON **only** in the format described above."
            ),
        }

        raw = _strip_fences(_llm_chat([sys_msg, user_msg]))
        try:
            parsed = json.loads(raw)
            picked = parsed.get("playlist", [])

            # If the model under-delivered, pad with random extras from this chunk
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
            print(f"⚠️ JSON parse error – {e}. Raw start: {raw[:120]}")
        bar.update(len(chunk))
    bar.close()

    return playlist

# --------------------------------------------------
# PLAYLIST PUSH/UPDATE HELPERS (unchanged)
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
            print("❌ Failed to update existing playlist tracks.")
            return False
    else:
        upd = requests.get(
            f"{SUBSONIC_BASE_URL}/createPlaylist",
            params={**SUBSONIC_AUTH_PARAMS, "name": name, "songId": song_ids},
        )
        if upd.status_code != 200:
            print("❌ Failed to create playlist.")
            return False
        root = ET.fromstring(upd.content)
        plid = root.find(f".//{ns}playlist").get("id")

    desc_upd = requests.get(
        f"{SUBSONIC_BASE_URL}/updatePlaylist",
        params={**SUBSONIC_AUTH_PARAMS, "playlistId": plid, "comment": description},
    )
    if desc_upd.status_code != 200:
        print("⚠️ Tracks updated but could not set description.")

    return True

# --------------------------------------------------
# SUBSONIC BROWSING HELPERS  🆕
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
    return sorted(
        art["name"]
        for letter in idx
        for art in letter.get("artist", [])
        if art.get("name")
    )


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

def fetch_all_playlists() -> list[dict]:
    """Return all playlists from the server with their details."""
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
# 🔎  PROMPT ARTIST EXTRACTION
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
        print(f"⚠️ JSON parse error – {e}. Raw start: {raw[:120]}")
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
    print(f"🔄 Added {len(remaining[:needed])} random songs from filtered options to reach minimum length of {min_songs}.")
    return playlist[:max_songs]

# --------------------------------------------------
# 🧹  PLAYLIST ENTRY SANITISER
# --------------------------------------------------

def _sanitize_playlist(entries: List[dict], candidates: List[dict]) -> List[dict]:
    """
    Ensure each playlist entry has an 'id'. If an entry only has a 'title'
    (and optionally 'artist'), try to resolve the matching song in *candidates*
    via a case-insensitive title-and-artist match.  Drop any rows we can’t
    resolve.  This is backend-agnostic and therefore safe for both Ollama and
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
        if "id" in e:                       # already good
            cleaned.append(e)
            continue
        key = (e.get("title", "").lower(), e.get("artist", "").lower())
        resolved = id_by_pair.get(key)
        if resolved:
            cleaned.append({"id": resolved, "title": e.get("title")})
    return cleaned

# --------------------------------------------------
# MAIN (updated flow with context playlist)
# --------------------------------------------------

def _main_impl(args):
    playlist_name = args.playlist_name
    prompt = args.prompt or input("Enter a prompt for the playlist vibe: ").strip()
    if not prompt:
        print("Prompt required – exiting.")
        return

    print("→ Fetching library…")
    all_songs = fetch_all_subsonic_songs()
    if not all_songs:
        print("No songs found on the server.")
        return

    # Fetch all existing playlists
    existing_playlists = fetch_all_playlists()

    # Select context playlist songs (separate list)
    context_songs = select_context_playlist_songs(prompt, existing_playlists, all_songs)

    # Fetch full library artists/genres
    all_artists = fetch_all_artists()
    all_genres = fetch_all_genres()

    # Extract artists/genres from context playlist (weighted highest)

    # Split artist strings on commas/semicolons and strip whitespace
    artists = []
    for s in context_songs:
        if s.get("artist"):
            # normalize separators to comma (handle semicolons and bullet •)
            raw = s["artist"].replace(";", ",").replace("•", ",")
            for a in raw.split(","):
                name = a.strip()
                if name:
                    artists.append(name)
    context_artists = list(dict.fromkeys(artists))

    # Extract genres from context playlist (splitting multi-genre strings)
    # Normalize separators and split on commas/semicolons
    genres = []
    for s in context_songs:
        if s.get("genre"):
            raw = s["genre"].replace(";", ",").replace("•", ",")
            for g in raw.split(","):
                name = g.strip()
                if name:
                    genres.append(name)
    context_genres = list(dict.fromkeys(genres))

    # Stricter artist/genre counts when no context playlist

    num_artists: int = 25
    num_genres:  int = 40

    if not context_songs:
        num_artists = int(num_artists * 1.5)  # 38
        num_genres  = int(num_genres  * 2)    # 80
        

    # First, collect any artists explicitly mentioned in the prompt
    explicit_prompt_artists = extract_prompt_artists(prompt, all_artists)

    init_focus_artists = select_top_items(prompt, all_artists, num_artists, "artists")
    init_focus_genres  = select_top_items(prompt, all_genres, num_genres, "genres")

    if not context_songs:
        focus_artists = init_focus_artists
        focus_genres = init_focus_genres
    else:
        focus_artists = select_top_items(prompt, list(set(context_artists + init_focus_artists)), 15, "artists")
        focus_genres = select_top_items(prompt, list(set(context_genres + init_focus_genres)), 25, "genres")

    # 🚨  Ensure prompt-named artists lead the list and aren’t dropped by truncation
    if explicit_prompt_artists:
        ordered = explicit_prompt_artists + [a for a in focus_artists if a not in explicit_prompt_artists]
        focus_artists = ordered[: max(num_artists, len(explicit_prompt_artists))]

    print("🎯 Focus artists:", ", ".join(focus_artists))
    print("🎯 Focus genres: ", ", ".join(focus_genres))

    # Filter library based on focus
    filtered = [s for s in all_songs if (s["artist"] in focus_artists) and (s.get("genre") in focus_genres)]
    if not filtered and not context_songs:
        print("No songs match the selected artists/genres – aborting.")
        return

    # Combine filtered with context songs, no duplicates
    if context_songs:
        filtered_ids = {s["id"] for s in filtered}
        combined_songs = filtered + [s for s in context_songs if s["id"] not in filtered_ids]
    else:
        combined_songs = filtered

    random.shuffle(combined_songs)  # Shuffle to randomise order
    print(f"→ Generating playlist ({len(combined_songs)} candidate songs)…")

    # We need at least args.min_songs tracks in total → derive a ratio
    required_ratio = args.min_songs / max(1, len(combined_songs))

    chunk_size = 30 if LLM_MODE == "ollama" else 500
    playlist_items = generate_playlist(
        prompt,
        combined_songs,
        chunk_size=chunk_size,
        required_ratio=required_ratio,
    )
    print(f"🎶 Generated {len(playlist_items)} tracks from {len(combined_songs)} candidates.")
    playlist_items = ensure_min_songs(playlist_items, combined_songs, args.min_songs)

    # 🧹  Resolve any entries missing an 'id' before upload
    playlist_items = _sanitize_playlist(playlist_items, combined_songs)

    if not playlist_items:
        print("LLM returned an empty playlist.")
        return

    print(f"✅ Got {len(playlist_items)} tracks. Pushing to server…")
    song_ids = [t["id"] for t in playlist_items]
    if _update_playlist_on_server(playlist_name, song_ids, prompt):
        print(f"🎉 Playlist '{playlist_name}' updated!")
    else:
        print("❌ Failed to update playlist on server.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a playlist based on a vibe prompt.")
    parser.add_argument("--playlist_name", type=str, default="naviDJ", help="Name of the playlist to create or update.")
    parser.add_argument("--prompt", type=str, help="Vibe prompt for the playlist.")
    parser.add_argument("--min_songs", type=int, default=35, help="Minimum number of songs in the playlist.")
    parser.add_argument("--llm_mode", type=str, choices=["openai", "ollama"], default=LLM_MODE, help="Which LLM backend to use.")
    args = parser.parse_args()

    # Configure the language model **before** anything else touches it.
    configure_llm(args.llm_mode)

    # From here on the original `main` logic continues, unchanged. For brevity
    # call the existing `main()` implementation renamed to `_main_impl(args)`.
    _main_impl(args)
