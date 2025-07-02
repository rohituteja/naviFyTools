# naviFy Tools
# A DJ for Navidrome and Spotify <-> Navidrome/Subsonic Playlist Syncing

A set of utilities for syncing playlists between Subsonic/Navidrome and Spotify and an AI powered DJ script to use with your music server to generate playlists as you please.

## Features
- **naviDJ.py**: AI-powered playlist generator for Subsonic/Navidrome using OpenAI or Ollama LLMs.
- **portLibrary.py**: Syncs starred/liked songs and playlists between Spotify and Subsonic/Navidrome (both directions).
- **portGenres.py**: Updates local music file genres using MusicBrainz tags. The DJ script uses genres to filter songs by, and more detailed genres, such as those from MusicBrainz, helps it filter more effectively, but by no means am I asking you to overwrite all your genres on your files to use these scripts. I'd be interested to see how the DJ performs with different types of metadata and how people tag their genres.

---

## Setup

1. **Clone this repo**
2. **Install dependencies**
   - Python 3.8+
   - `pip install -r requirements.txt` (see scripts for required packages)
3. **Create a `secrets.txt` file in the repo root:**

```
[music_directory]
MUSIC_DIR = <path to your music directory here>

[llm]
# Set the default LLM backend and model for naviDJ.py, "ollama" or "openai"
MODE = openai
# e.g. gpt-4o-mini, gpt-3.5-turbo, deepseek-r1:latest, gemma3n:latest, etc.
MODEL = gpt-4o-mini

[openai]
OPENAI_KEY = <your-openai-api-key>

[ollama]
OLLAMA_BASE = <your-ollama-url>

[subsonic]
BASE_URL = <your-subsonic-url>
USER = <your-username>
PASSWORD = <your-password>
API_VERSION = 1.16.1
CLIENT = naviFy_scripts

[spotify]
SCOPE = user-read-private user-read-playback-state user-library-read user-library-modify playlist-modify-public playlist-modify-private
CLIENT_ID = <your-spotify-client-id>
CLIENT_SECRET = <your-spotify-client-secret>
REDIRECT_URI = http://localhost/
CACHE_PATH = .cache-spotify
```

An example file for you to use and rename has also been provided.

- **Get your API keys:**
  - [OpenAI API key](https://platform.openai.com/account/api-keys)
  - [Spotify Developer credentials](https://developer.spotify.com/dashboard/applications)
  - Subsonic/Navidrome: Use your server's credentials
  - Ollama: Set up your local Ollama server and use its URL if you'd like

The playlist sync script does attempt to sync over playlist images, however Navidrome doesn't do anything with images that are sent with a playlist. Despite this, the DJ script still uses a DJ.png file try and set a default image for the playlists, which is included for you if you'd like to use or replace it. 

---

## Usage

### naviDJ.py
Generate a playlist using AI:
```
python naviDJ.py --playlist_name "My Playlist" --prompt "energetic summer road trip" --min_songs 40 --llm_mode openai --llm_model gpt-4o-mini
```
- If arguments are omitted, the script will prompt for them interactively, with a default playlist name of "naviDJ".
- You can set the default LLM backend and model in `secrets.txt` under the `[llm]` section. Command-line arguments override these defaults.

### portLibrary.py
Sync playlists and likes between Spotify and Subsonic:
```
python portLibrary.py --sync-starred y --sync-playlists y --import-liked y --import-playlists y --playlists "Playlist1,Playlist2"
```
- If arguments are omitted, the script will prompt for them interactively.

### portGenres.py
Update genres for your local music library using MusicBrainz:
```
python portGenres.py /path/to/your/music --dry-run
```
- Omit `--dry-run` to actually write changes.

---

## Notes
- **Never commit your `secrets.txt` file!** It is gitignored by default.
- For detailed options, see the top of each script or run with `--help`.

---

## License
GPL-3.0
