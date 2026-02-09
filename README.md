# naviFy Tools
# A DJ for Navidrome and Spotify <-> Navidrome/Subsonic Playlist Syncing

A set of utilities for syncing playlists between Subsonic/Navidrome and Spotify and an AI powered DJ script to use with your music server to generate playlists as you please.

## Features
- **naviDJ.py**: AI-powered playlist generator for Subsonic/Navidrome using OpenAI, Ollama, or a Custom OpenAI-compatible API.
    - **Semantic Optimization**: Uses embedding-based similarity search to pre-filter your library, reducing token usage by 80-90% and speeding up generation.
    - **Chunked Generation**: Processes large song lists in configurable batches to handle context window limits and improve accuracy.
    - **Context Awareness**: Adds smarter focus via context playlists and relevant albums from your library.
- **portLibrary.py**: Syncs starred/liked songs and playlists between Spotify and Subsonic/Navidrome (both directions).
- **portGenres.py**: Updates local music file genres using MusicBrainz tags.
- **Web App**: Modern web interface with configuration management, Spotify login, model picker, and real-time output.

**Note on Open WebUI**: This project fully supports Open WebUI! You can use it as a robust proxy for Ollama by selecting "ollama" mode and pointing the URL to your Open WebUI instance. This allows you to leverage Open WebUI's features like custom context lengths and model management while using naviDJ.

## Web App (Flask UI)

A simple web interface is now available for managing your naviFy tools!

- **Run the web app:**
  ```
  python app.py
  ```
- Open your browser to [http://localhost:5000](http://localhost:5000)
- Features:
  - Edit your `secrets.txt` configuration from the web UI
  - Run the AI DJ (naviDJ.py) and Library Porter (portLibrary.py) from your browser
  - View real-time output from scripts in the browser

**Features:**
- **Spotify OAuth Integration**: Built-in Spotify authentication with token caching and refresh.
- **Configuration Management**: Edit your `secrets.txt` from the web UI.
- **Model Picker**: Choose models for OpenAI/Ollama/Custom providers from a dropdown.
- **Script Execution**: Run the AI DJ (`naviDJ.py`) and Library Porter (`portLibrary.py`) from your browser.
- **Real-time Output**: View live logs for long-running jobs in the browser.
- **Playlist Chooser**: When importing playlists from Spotify, select from your owned playlists in the UI.
- You must have a valid `secrets.txt` file set up before using the web app.

> **Heads up:**
> You can serve this Flask app and, using tools like [Tailscale](https://tailscale.com/), make it accessible over HTTPS. This enables installable PWA (Progressive Web App) features on mobile and desktop browsers.

---

## Setup

1. **Clone this repo**
2. **Install dependencies**
   - Python 3.8+
   - `pip install -r requirements.txt`
3. **Create a `secrets.txt` file in the repo root:**

```ini
[music_directory]
# Optional: Path to your music directory
MUSIC_DIR = <path to your music directory here>

[llm]
# Set the default LLM backend: "openai", "ollama", or "custom"
MODE = openai
# LLM model name (e.g. gpt-4o-mini, gemma3n:latest)
MODEL = gpt-4o-mini
# Number of songs to process at once. Lower values reduce token usage but may be slower.
# Default: 500
CHUNK_SIZE = 500

[openai]
OPENAI_KEY = <your-openai-api-key>
# Recommended: Embedding model for semantic search
EMBEDDING_MODEL = text-embedding-3-small

[ollama]
# URL for local Ollama (http://localhost:11434/v1) or Open WebUI (http://localhost:3000/api)
OLLAMA_BASE = <your-ollama-or-openwebui-url>
# API Key (Required for Open WebUI, optional for local Ollama)
API_KEY = <your-api-key>
# Recommended: Embedding model for semantic search
EMBEDDING_MODEL = nomic-embed-text

[custom]
# For OpenAI-compatible APIs
BASE_URL = <your-custom-api-base-url>
API_KEY = <your-custom-api-key>

[subsonic]
BASE_URL = <your-subsonic-url>
USER = <your-username>
PASSWORD = <your-password>
API_VERSION = 1.16.1
CLIENT = naviFy_scripts

[spotify]
CLIENT_ID = <your-spotify-client-id>
CLIENT_SECRET = <your-spotify-client-secret>
REDIRECT_URI = http://localhost:5000/
CACHE_PATH = .cache-spotify
SCOPE = user-read-private user-read-playback-state user-library-read user-library-modify playlist-read-private playlist-modify-public playlist-modify-private
```

An example file for you to use and rename has also been provided.

- **Get your API keys:**
  - [OpenAI API key](https://platform.openai.com/account/api-keys)
  - [Spotify Developer credentials](https://developer.spotify.com/dashboard/applications) (see setup guide below)
  - Subsonic/Navidrome: Use your server's credentials
  - Ollama: Set up your local Ollama server and use its URL if you'd like

### Spotify API Setup Guide

To use the Spotify OAuth authentication, you need to create a Spotify application:

1. **Create a Spotify App:**
   - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   - Click "Create App"
   - Fill in the required information:
     - **App name**: `naviFy Tools` (or any name you prefer)
     - **App description**: `Music library sync and AI DJ tools`
     - **Redirect URI**: `http://localhost:5000/` (recommended) or `http://localhost:5000/spotify/callback`
     - **Website**: Can be left blank or set to your domain
   - Click "Save"

2. **Get Your Credentials:**
   - Copy the **Client ID** and **Client Secret** to your `secrets.txt`.
3. **Configure Redirect URIs:**
   - In your Spotify app settings, add the redirect URI that matches your setup:
     - **Local development**: `http://localhost:5000/` (or `http://localhost:5000/spotify/callback`)
     - **Custom domain**: `https://yourdomain.com/` (if hosting publicly)
     - **Custom port**: `http://localhost:8080/` (if using a different port)
   - **Important**: The redirect URI in your Spotify app settings must exactly match the `REDIRECT_URI` in your `secrets.txt` file.
   - The web app supports both returning to `/` with a `?code=...` and `/spotify/callback`.

4. **Update Your Configuration:**
   - In your `secrets.txt`, set:
     ```
     [spotify]
     CLIENT_ID = your-client-id-here
     CLIENT_SECRET = your-client-secret-here
     REDIRECT_URI = http://localhost:5000/
     ```

**Note:** If you change the port or domain where you host the Flask app, remember to update both your Spotify app settings and the `REDIRECT_URI` in your `secrets.txt` file.

## Semantic Optimization (Recommended)

When using Ollama or OpenAI, you can configure an `EMBEDDING_MODEL` to significantly improve performance. 

**Benefits:**
- **Drastically Faster**: Reduces the song list sent to the LLM, making generation much faster.
- **Cost Efficient**: Drops token usage by up to 90% while maintaining identical playlist quality.
- **Batch Processing**: Automatically uses batch API calls for local caches to speed up initial library indexing.

**Setup for Ollama:**
1. Pull an embedding model:
   ```bash
   ollama pull nomic-embed-text
   ```
2. Add `EMBEDDING_MODEL = nomic-embed-text` to your `[ollama]` section in `secrets.txt`.

**Setup for OpenAI:**
1. Add `EMBEDDING_MODEL = text-embedding-3-small` to your `[openai]` section in `secrets.txt`.

**Embedding Cache:**
Embeddings are cached locally (e.g., `embeddings_cache_nomic-embed-text.pkl`) to avoid redundant calls. The cache automatically updates if your library grows or you change models.

---

## Usage

### Web App (Recommended)
1. **Start the web app:** `python app.py`
2. **Open browser** to [http://localhost:5000](http://localhost:5000)
3. **Configure Settings**: Use the configuration tab to manage your `secrets.txt` easily.

### Command Line Usage

#### naviDJ.py
Generate a playlist using AI:
```bash
python naviDJ.py --playlist_name "My Playlist" --prompt "energetic summer road trip" --min_songs 40 --chunk_size 500
```
- **Arguments**:
  - `--playlist_name`: Name of the playlist (default: naviDJ).
  - `--prompt`: The vibe description (required if not interactive).
  - `--min_songs`: Target number of songs (default: 35).
  - `--chunk_size`: Songs per batch (default: 500). Overrides `secrets.txt`.
  - `--llm_mode`: Backend to use (`openai` or `ollama`).
  - `--llm_model`: Specific model to use.

#### portLibrary.py
Sync playlists and likes:
```bash
python portLibrary.py --sync-starred y --sync-playlists y --import-liked y --import-playlists y
```
- **Note**: By default, this script ignores playlists named `naviDJ` to avoid circular syncs. When syncing TO Spotify, it only adds missing songs (does not overwrite).

#### portGenres.py
Update local music tags:
```bash
python portGenres.py /path/to/your/music --dry-run
```
- Uses MusicBrainz to update `GENRE` tags. Richer genres help `naviDJ` filter more effectively.

---

## Troubleshooting

- **"INVALID_CLIENT: Invalid redirect URI"**: Ensure your Spotify dashboard URI matches `secrets.txt` EXACTLY.
- **Slow Generation**: Ensure `EMBEDDING_MODEL` is configured to enable semantic pre-filtering.
- **Parse Errors**: Check the real-time logs in the web app. If using local models, ensure they are capable of outputting valid JSON (recommended: `gpt-4o-mini`, `gemma3n`, `llama3.1`).

---

## License
GPL-3.0
