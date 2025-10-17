"""
Spotify ↔ Subsonic playlist and library sync utility.

- Syncs starred/liked songs and playlists between Spotify and Subsonic/Navidrome.
- Supports both interactive and argument-driven CLI usage.

Usage:
    python portLibrary.py [--sync-starred y/n] [--sync-playlists y/n] [--import-liked y/n] [--import-playlists y/n] [--playlists "Playlist1,Playlist2"]

If arguments are omitted, the script will prompt for them interactively.
"""

import os
import configparser
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests
import xml.etree.ElementTree as ET
import argparse
from rapidfuzz import fuzz
import re
import unicodedata
from tqdm import tqdm, trange

# Load secrets from secrets.txt
secrets = configparser.ConfigParser()
secrets.read(os.path.join(os.path.dirname(__file__), "secrets.txt"))

SPOTIFY_SCOPE = secrets.get("spotify", "SCOPE", fallback=None)
SPOTIFY_CLIENT_ID = secrets.get("spotify", "CLIENT_ID", fallback=None)
SPOTIFY_CLIENT_SECRET = secrets.get("spotify", "CLIENT_SECRET", fallback=None)
SPOTIFY_REDIRECT_URI = secrets.get("spotify", "REDIRECT_URI", fallback="http://localhost/")
SPOTIFY_CACHE_PATH = secrets.get("spotify", "CACHE_PATH", fallback=".cache-bidisync")

SUBSONIC_BASE_URL = secrets.get("subsonic", "BASE_URL", fallback=None)
SUBSONIC_AUTH_PARAMS = {
    "u": secrets.get("subsonic", "USER", fallback=None),
    "p": secrets.get("subsonic", "PASSWORD", fallback=None),
    "v": secrets.get("subsonic", "API_VERSION", fallback="1.16.1"),
    "c": secrets.get("subsonic", "CLIENT", fallback="bidisync"),
}

# Set up Spotify API client
def authenticate_spotify():
    sp_oauth = SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SPOTIFY_SCOPE,
        cache_path=SPOTIFY_CACHE_PATH,
        show_dialog=False                 # <-- don’t force prompt
    )

    token_info = sp_oauth.get_cached_token()          # ① look for cache

    if token_info and sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(   # ② silent refresh
            token_info["refresh_token"]
        )

    if not token_info:                                # ③ interactive login
        auth_url = sp_oauth.get_authorize_url()
        print("Authorize here:", auth_url)
        redirected = input("Paste the full redirect URL: ").strip()
        code = sp_oauth.parse_response_code(redirected)
        token_info = sp_oauth.get_access_token(code)

    return spotipy.Spotify(auth=token_info["access_token"])

# Usage
sp = authenticate_spotify()

# Pagination Helpers
def get_all_user_playlists():
    playlists = []
    limit = 50
    offset = 0
    while True:
        resp = sp.current_user_playlists(limit=limit, offset=offset)
        items = resp.get('items', [])
        playlists.extend(items)
        if len(items) < limit:
            break
        offset += limit
    return playlists


def get_all_playlist_tracks(playlist_id):
    tracks = []
    limit = 100
    offset = 0
    while True:
        resp = sp.playlist_items(playlist_id, limit=limit, offset=offset)
        items = resp.get('items', [])
        tracks.extend(items)
        if len(items) < limit:
            break
        offset += limit
    return tracks

# Utility Functions
def clean_song_title(song_title):
    patterns = [
        r"\(feat\..*?\)", r"\(ft\..*?\)", r"-\s*feat\..*", r"-\s*ft\..*",
        r"ft\..*", r"feat\..*", r"\(.*?remaster.*?\)", r"- .*? Remaster",
        r"- Remastered.*?", r"- Single Version", r"- Live", r"- From",
        r"\[.*?\]", r"\(.*?\)"
    ]
    for pattern in patterns:
        song_title = re.sub(pattern, "", song_title, flags=re.IGNORECASE)
    song_title = re.sub(r"[-–—]\s*$", "", song_title).strip()
    return re.sub(r"\s+", " ", song_title).strip()


def normalize_text(text: str) -> str:
    """
    Unicode-normalize, strip accents and punctuation, lowercase.
    """
    nfkd = unicodedata.normalize('NFKD', text)
    no_diacritics = ''.join(c for c in nfkd if not unicodedata.combining(c))
    cleaned = ''.join(c for c in no_diacritics if c.isalnum() or c.isspace())
    return cleaned.lower().strip()


def fetch_spotify_playlist_image(playlist_id):
    images = sp.playlist(playlist_id).get("images", [])
    return images[0].get("url") if images else None


def set_subsonic_playlist_image(playlist_id, image_url):
    response = requests.post(
        f"{SUBSONIC_BASE_URL}/updatePlaylist",
        params={**SUBSONIC_AUTH_PARAMS, 'playlistId': playlist_id},
        files={"coverArt": requests.get(image_url).content}
    )
    return response.status_code == 200


def search_subsonic_song(song_title, artist_names):
    """
    Search Subsonic by normalized title and first artist; require both fuzzy-match thresholds.
    Falls back to title-only match if normalized artist is non-alphanumeric.
    """
    raw_title = clean_song_title(song_title)
    # Extract first artist and clean
    first_artist = artist_names.split(',')[0].strip()
    raw_artist = clean_song_title(first_artist)

    # Determine if artist has alphanumeric
    has_alnum = bool(re.search(r"[A-Za-z0-9]", raw_artist))

    title_query = normalize_text(raw_title)
    artist_query = normalize_text(raw_artist)

    response = requests.get(
        f"{SUBSONIC_BASE_URL}/search3",
        params={**SUBSONIC_AUTH_PARAMS, 'query': raw_title, 'type': 'song'}
    )
    root = ET.fromstring(response.content)
    ns = root.tag.split('}')[0] + '}'

    best_match = None
    best_score = 0
    for song in root.findall(f".//{ns}song"):
        xml_title = normalize_text(song.get('title') or '')
        title_score = fuzz.partial_ratio(title_query, xml_title)

        if has_alnum:
            # require artist match if valid artist present
            xml_artist = normalize_text(song.get('artist') or '')
            artist_score = fuzz.partial_ratio(artist_query, xml_artist)
            if title_score > 75 and artist_score > 75:
                avg_score = (title_score + artist_score) / 2
                if avg_score > best_score:
                    best_score = avg_score
                    best_match = song
        else:
            # fallback: title-only
            if title_score > 75 and title_score > best_score:
                best_score = title_score
                best_match = song

    return best_match.get('id') if best_match is not None else None


def get_subsonic_playlist_id(playlist_name):
    response = requests.get(
        f"{SUBSONIC_BASE_URL}/getPlaylists",
        params=SUBSONIC_AUTH_PARAMS
    )
    root = ET.fromstring(response.content)
    ns = root.tag.split('}')[0] + '}'
    for pl in root.findall(f".//{ns}playlist"):
        if pl.get('name') == playlist_name:
            return pl.get('id')
    return None


def create_or_update_subsonic_playlist(playlist_name, song_ids, description):
    pl_id = get_subsonic_playlist_id(playlist_name)
    if pl_id:
        resp = requests.get(
            f"{SUBSONIC_BASE_URL}/createPlaylist",
            params={**SUBSONIC_AUTH_PARAMS, 'playlistId': pl_id, 'songId': song_ids}
        )
        requests.get(
            f"{SUBSONIC_BASE_URL}/updatePlaylist",
            params={**SUBSONIC_AUTH_PARAMS, 'playlistId': pl_id, 'comment': description}
        )
    else:
        resp = requests.get(
            f"{SUBSONIC_BASE_URL}/createPlaylist",
            params={**SUBSONIC_AUTH_PARAMS, 'name': playlist_name, 'songId': song_ids}
        )
        new_root = ET.fromstring(resp.content)
        ns2 = new_root.tag.split('}')[0] + '}'
        new_pl = new_root.find(f".//{ns2}playlist")
        new_id = new_pl.get('id') if new_pl is not None else None
        if new_id:
            requests.get(
                f"{SUBSONIC_BASE_URL}/updatePlaylist",
                params={**SUBSONIC_AUTH_PARAMS, 'playlistId': new_id, 'comment': description}
            )
    return resp.status_code == 200

# -------------------
# Fetch all liked tracks from Spotify
# -------------------
def get_all_liked_tracks():
    """Return full list of saved tracks (Spotify 'Liked Songs')."""
    liked = []
    limit, offset = 50, 0
    while True:
        resp = sp.current_user_saved_tracks(limit=limit, offset=offset)
        items = resp.get("items", [])
        liked.extend(items)
        if len(items) < limit:
            break
        offset += limit
    return liked

# -------------------
# Star a song on Subsonic/Navidrome
# -------------------
def star_subsonic_song(song_id: str) -> bool:
    """PUT a star on a Subsonic/Navidrome song."""
    r = requests.get(
        f"{SUBSONIC_BASE_URL}/star",
        params={**SUBSONIC_AUTH_PARAMS, "id": song_id}
    )
    return r.status_code == 200

# -------------------
# Increment play-count on Subsonic/Navidrome
# -------------------
def scrobble_subsonic(song_id: str, count: int = 1):
    """Increment play-count by calling /scrobble N times (optional)."""
    for _ in range(count):
        requests.get(
            f"{SUBSONIC_BASE_URL}/scrobble",
            params={**SUBSONIC_AUTH_PARAMS, "id": song_id, "submission": "true"}
        )

# -------------------
# Sync Spotify likes and play counts to Subsonic/Navidrome
# -------------------
def sync_likes_and_playcounts():
    liked_tracks = get_all_liked_tracks()
    print(f"Found {len(liked_tracks)} liked tracks on Spotify.")

    no_match, starred = 0, 0
    progress_bar = trange(0, len(liked_tracks), 100, desc="Importing liked songs", unit="batch", dynamic_ncols=True)

    for i in progress_bar:
        batch = liked_tracks[i:i + 100]
        for item in batch:
            tr = item["track"]
            title = tr["name"]
            artists = ", ".join(a["name"] for a in tr["artists"])

            subsonic_id = search_subsonic_song(title, artists)
            if subsonic_id:
                if star_subsonic_song(subsonic_id):
                    starred += 1

                # --- Play-count sync (disabled by default) ------------------
                # spotify_plays = estimate_plays_somehow(tr)  # <-- you need your own logic here
                # if spotify_plays:
                #     scrobble_subsonic(subsonic_id, max(0, spotify_plays - 1))
            else:
                no_match += 1

    print(f"Starred {starred} tracks on Navidrome; {no_match} had no match.")

# ---------------------------------------------------------------------------
# Bidirectional Additions: Subsonic ➜ Spotify
# ---------------------------------------------------------------------------

def fetch_subsonic_starred_songs():
    """Return list of dicts with keys title, artist for all starred songs on Subsonic/Navidrome."""
    r = requests.get(f"{SUBSONIC_BASE_URL}/getStarred2", params=SUBSONIC_AUTH_PARAMS)
    root = ET.fromstring(r.content)
    ns = root.tag.split("}")[0] + "}"
    songs = []
    for song in root.findall(f".//{ns}song"):
        songs.append({
            "title": song.get("title") or "",
            "artist": song.get("artist") or "",
        })
    return songs


def search_spotify_track(title, artist):
    query = f"track:{clean_song_title(title)} artist:{artist.split(',')[0]}"
    try:
        res = sp.search(q=query, type="track", limit=1)
        items = res.get("tracks", {}).get("items", [])
        return items[0]["id"] if items else None
    except Exception:
        return None


def sync_starred_to_spotify():
    print("Fetching starred tracks from Subsonic…")
    starred = fetch_subsonic_starred_songs()
    print(f"Found {len(starred)} starred tracks on the server.")
    liked = get_all_liked_tracks()
    liked_ids = {item["track"]["id"] for item in liked}
    to_like = []
    for st in tqdm(starred, desc="Matching on Spotify", unit="song"):
        sid = search_spotify_track(st["title"], st["artist"])
        if sid and sid not in liked_ids:
            to_like.append(sid)
    print(f"Will like {len(to_like)} new tracks on Spotify.")
    for i in trange(0, len(to_like), 50, desc="Liking in batches", unit="batch"):
        batch = to_like[i:i + 50]
        sp.current_user_saved_tracks_add(batch)
    print("Starred ➜ Spotify sync complete.")


def fetch_subsonic_playlist(playlist_id):
    r = requests.get(
        f"{SUBSONIC_BASE_URL}/getPlaylist",
        params={**SUBSONIC_AUTH_PARAMS, "id": playlist_id},
    )
    root = ET.fromstring(r.content)
    ns = root.tag.split("}")[0] + "}"
    tracks = []
    for entry in root.findall(f".//{ns}entry"):
        tracks.append((entry.get("title") or "", entry.get("artist") or ""))
    return tracks


def title_key(title: str) -> str:
    return normalize_text(clean_song_title(title))


def artist_key(artist: str) -> str:
    # Only first listed artist matters for fuzzy compare
    return normalize_text(clean_song_title(artist.split(",")[0]))


def sync_playlists_to_spotify() -> None:
    print("\n▶ Syncing server playlists ➜ Spotify (title‑centric dedupe)…")

    # 1️⃣ Gather server playlists
    resp = requests.get(f"{SUBSONIC_BASE_URL}/getPlaylists", params=SUBSONIC_AUTH_PARAMS)
    root = ET.fromstring(resp.content)
    ns = root.tag.split("}")[0] + "}"
    server_pls = {pl.get("name"): pl.get("id") for pl in root.findall(f".//{ns}playlist")}

    # 2️⃣ Gather Spotify playlists
    spotify_pls = {pl["name"]: pl for pl in get_all_user_playlists()}
    user_id = sp.me()["id"]

    for name, srv_id in server_pls.items():

        if name == "naviDJ":
            continue  # Skip the naviDJ playlist

        srv_tracks = fetch_subsonic_playlist(srv_id)
        if not srv_tracks:
            continue

        # Build lookup of existing titles → list[artist_keys]
        if name in spotify_pls:
            sp_pl = spotify_pls[name]
            sp_items = get_all_playlist_tracks(sp_pl["id"])
            existing_titles = {}
            for it in sp_items:
                tr = it.get("track")
                if not tr:
                    continue
                t_key = title_key(tr["name"])
                a_key = artist_key(", ".join(a["name"] for a in tr["artists"]))
                existing_titles.setdefault(t_key, []).append(a_key)
        else:
            sp_pl = None
            existing_titles = {}

        seen_titles = {}
        to_add = []
        for ttl, art in srv_tracks:
            t_key = title_key(ttl)
            a_key = artist_key(art)

            # Stage‑run duplicate check
            staged_artists = seen_titles.get(t_key, [])
            all_artists_for_title = existing_titles.get(t_key, []) + staged_artists
            duplicate = False
            for ex_a in all_artists_for_title:
                if fuzz.partial_ratio(a_key, ex_a) >= 70:
                    duplicate = True
                    break
            if duplicate:
                continue

            # Search Spotify for a track ID
            tid = search_spotify_track(ttl, art)
            if not tid:
                continue

            to_add.append(tid)
            seen_titles.setdefault(t_key, []).append(a_key)

        if not to_add:
            continue

        if sp_pl is None:
            print(f"  • Creating new playlist ‘{name}’ with {len(to_add)} tracks…")
            sp_pl = sp.user_playlist_create(user_id, name, public=True)
        else:
            print(f"  • Adding {len(to_add)} unique tracks to Spotify playlist ‘{name}’.")

        for i in range(0, len(to_add), 100):
            sp.playlist_add_items(sp_pl["id"], to_add[i : i + 100])

    print("✅  Server ➜ Spotify playlist sync finished – title‑centric dedupe in effect.\n")

# -------------------
# Main Function
# -------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Transfer Spotify playlists to Subsonic.")
    parser.add_argument("--sync-starred", choices=['y', 'n'], help="Sync Subsonic 'starred' ➜ Spotify 'Liked Songs'?")
    parser.add_argument("--sync-playlists", choices=['y', 'n'], help="Sync Subsonic playlists ➜ Spotify?")
    parser.add_argument("--import-liked", choices=['y', 'n'], help="Import liked songs from Spotify?")
    parser.add_argument("--import-playlists", choices=['y', 'n'], help="Import playlists from Spotify?")
    parser.add_argument("--playlists", type=str, help="Comma-separated list of playlist names to migrate.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Collect y/n inputs upfront
    sync_starred = args.sync_starred or input("Sync Subsonic 'starred' ➜ Spotify 'Liked Songs'? (y/n): ").strip().lower()
    sync_playlists = args.sync_playlists or input("Sync Subsonic playlists ➜ Spotify? (y/n): ").strip().lower()
    import_liked_songs = args.import_liked or input("Do you want to import liked songs from Spotify? (y/n): ").strip().lower()
    import_playlists = args.import_playlists or input("Do you want to import playlists from Spotify? (y/n): ").strip().lower()

    if sync_starred == 'y':
        sync_starred_to_spotify()

    if sync_playlists == 'y':
        sync_playlists_to_spotify()

    if import_liked_songs == 'y':
        sync_likes_and_playcounts()

    if import_playlists == 'y':
        playlists = []
        if args.playlists:
            playlists = [name.strip() for name in args.playlists.split(',')]
        else:
            all_playlists = get_all_user_playlists()
            print("Available Spotify playlists:")
            for idx, pl in enumerate(all_playlists, start=1):
                print(f"  {idx}. {pl['name']}")

            sel = input("Enter the numbers of the playlists to migrate, separated by commas: ")
            playlists = [all_playlists[int(i.strip()) - 1]['name'] for i in sel.split(',') if i.strip().isdigit()]

        if not playlists:
            print("No valid playlists selected. Exiting.")
            return

        missing_songs, missing_albums = {}, {}

        for pl_name in playlists:
            pl = next((pl for pl in get_all_user_playlists() if pl['name'] == pl_name), None)
            if not pl:
                print(f"Playlist '{pl_name}' not found. Skipping.")
                continue

            print(f"Processing playlist: {pl['name']}")
            name, desc = pl['name'], pl.get('description', '')
            tracks = get_all_playlist_tracks(pl['id'])
            ids = []
            for item in tracks:
                tn = item['track']['name']
                artists = ", ".join(a['name'] for a in item['track']['artists'])
                alb = item['track']['album']['name']
                sid = search_subsonic_song(tn, artists)
                if sid:
                    ids.append(sid)
                else:
                    missing_albums[alb] = missing_albums.get(alb, 0) + 1
                    missing_songs[tn] = missing_songs.get(tn, 0) + 1

            img = fetch_spotify_playlist_image(pl['id'])
            if create_or_update_subsonic_playlist(name, ids, desc):
                print(f"Playlist '{name}' successfully transferred.")
                if img and set_subsonic_playlist_image(get_subsonic_playlist_id(name), img):
                    print(f"Image for '{name}' set successfully.")

        print("All playlists have been successfully transferred.")

        print("\nTop 15 albums with missing songs:")
        for alb, cnt in sorted(missing_albums.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"  - {alb}: {cnt}")

        print("\nTop 10 missing songs:")
        for sn, cnt in sorted(missing_songs.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {sn}: {cnt}")

if __name__ == "__main__":
    main()