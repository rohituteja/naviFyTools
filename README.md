# naviFy Tools
# A DJ for Navidrome and Spotify <-> Navidrome/Subsonic Playlist Syncing

A set of utilities for syncing playlists between Subsonic/Navidrome and Spotify and an AI powered DJ script to use with your music server to generate playlists as you please.

## Features
- **naviDJ.py**: AI-powered playlist generator for Subsonic/Navidrome using OpenAI, Ollama, or a Custom OpenAI-compatible API. Adds smarter focus via context playlists and optionally relevant albums from your library.
- **portLibrary.py**: Syncs starred/liked songs and playlists between Spotify and Subsonic/Navidrome (both directions).
- **portGenres.py**: Updates local music file genres using MusicBrainz tags. The DJ script uses genres to filter songs by, and more detailed genres, such as those from MusicBrainz, helps it filter more effectively, but by no means am I asking you to overwrite all your genres on your files to use these scripts. I'd be interested to see how the DJ performs with different types of metadata and how people tag their genres.
- **Web App**: Modern web interface with configuration management, Spotify login, model picker, real-time output, and a playlist chooser for selecting owned Spotify playlists when importing.

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

```
[music_directory]
MUSIC_DIR = <path to your music directory here>

[llm]
# Set the default LLM backend and model for naviDJ.py: "openai", "ollama", or "custom"
MODE = openai
# e.g. gpt-4o-mini, gpt-3.5-turbo, deepseek-r1:latest, gemma3n:latest, etc.
MODEL = gpt-4o-mini

[openai]
OPENAI_KEY = <your-openai-api-key>

[ollama]
OLLAMA_BASE = <your-ollama-url>

[custom]
# Use a custom OpenAI-compatible API (optional)
BASE_URL = <your-custom-api-base-url>  # e.g. https://your-endpoint.example.com/v1
API_KEY = <your-custom-api-key>

[subsonic]
BASE_URL = <your-subsonic-url>
USER = <your-username>
PASSWORD = <your-password>
API_VERSION = 1.16.1
CLIENT = naviFy_scripts

[spotify]
SCOPE = user-read-private user-read-playback-state user-library-read user-library-modify playlist-read-private playlist-modify-public playlist-modify-private
CLIENT_ID = <your-spotify-client-id>
CLIENT_SECRET = <your-spotify-client-secret>
REDIRECT_URI = http://localhost:5000/        # or use http://localhost:5000/spotify/callback
CACHE_PATH = .cache-spotify
```

An example file for you to use and rename has also been provided.

- **Get your API keys:**
  - [OpenAI API key](https://platform.openai.com/account/api-keys)
  - [Spotify Developer credentials](https://developer.spotify.com/dashboard/applications) (see setup guide below)
  - Subsonic/Navidrome: Use your server's credentials
  - Ollama: Set up your local Ollama server and use its URL if you'd like

### Spotify API Setup Guide

To use the Spotify OAuth authentication in the web app, you need to create a Spotify application:

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
   - After creating the app, you'll see your **Client ID** and **Client Secret**
   - Copy these values to your `secrets.txt` file

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

### Embedding Models (Optional but Recommended)

When using Ollama or OpenAI as your LLM backend, you can optionally configure an embedding model to significantly improve performance. Embedding models are used for semantic search to pre-filter large lists of artists, genres, albums, and songs before sending them to the LLM. This reduces token usage by 80-90% while maintaining identical output quality.

**Benefits:**
- **Faster playlist generation**: Pre-filtering reduces the amount of data sent to the LLM
- **Lower token costs**: Fewer tokens means lower API costs (for paid services) or faster processing
- **Same quality**: The LLM still makes all final decisions, so output quality is identical

**Setup for Ollama:**
1. **Pull an embedding model** using Ollama:
   ```bash
   ollama pull nomic-embed-text
   ```
   Or for faster processing (lower quality):
   ```bash
   ollama pull all-minilm
   ```

2. **Configure in `secrets.txt`**:
   ```
   [ollama]
   OLLAMA_BASE = http://localhost:11434/v1
   EMBEDDING_MODEL = nomic-embed-text
   # Optional: Set context window size (default: 8192)
   CONTEXT_LENGTH = 8192
   ```

**Context Length for Ollama:**
The `CONTEXT_LENGTH` parameter controls the context window size for Ollama models. A larger context window allows the model to consider more tokens at once when generating responses, which can improve generation speed and prevent truncation. The default is 8192 tokens, but you can adjust this based on your model's capabilities.

**Embedding Cache:**
Embeddings are cached locally to avoid redundant API calls. The cache is automatically invalidated and regenerated when:
- The embedding model is changed in configuration
- The library size increases (new songs/artists/albums are added)

This ensures that embeddings are always up-to-date and retrieval accuracy remains high.

**Setup for OpenAI:**
1. **Configure in `secrets.txt`** (no need to pull models, they're available via API):
   ```
   [openai]
   OPENAI_KEY = <your-openai-api-key>
   EMBEDDING_MODEL = text-embedding-3-small
   ```

**Recommended Models:**
- **Ollama:**
  - **nomic-embed-text** (default): Best quality, recommended for most users
  - **mxbai-embed-large**: High quality alternative
  - **all-minilm**: Fastest option, good for speed-focused setups
- **OpenAI:**
  - **text-embedding-3-small** (default): Best balance of quality and cost
  - **text-embedding-3-large**: Highest quality, higher cost
  - **text-embedding-ada-002**: Older model, still functional

**Important Notes:**
- Embedding models work with both **Ollama and OpenAI** backends (not custom APIs)
- If no embedding model is configured, the script works exactly as before - this is fully backward compatible
- For Ollama: The embedding model must be pulled before use: `ollama pull <model-name>`
- For OpenAI: Embedding models are available via API, no local installation needed
- Embeddings are cached locally in `embeddings_cache.pkl` to avoid redundant API calls

The playlist sync script does attempt to sync over playlist images, however Navidrome doesn't do anything with images that are sent with a playlist. Despite this, the DJ script still uses a DJ.png file try and set a default image for the playlists, which is included for you if you'd like to use or replace it. 

---

## Usage

### Web App (Recommended)
The easiest way to use naviFy Tools is through the web interface:

1. **Start the web app:**
   ```bash
   python app.py
   ```

2. **Open your browser** to [http://localhost:5000](http://localhost:5000)

3. **Configure your settings** in the Configuration section

4. **Use the tools:**
  - **naviDJ Tab**: Generate AI playlists with custom prompts; choose LLM backend and model.
  - **Library Porter Tab**: Sync playlists between Spotify and your music server; select owned playlists to import.
  - **Spotify Authentication**: Login with Spotify directly in the browser.

### Command Line Usage

#### naviDJ.py
Generate a playlist using AI:
```
python naviDJ.py --playlist_name "My Playlist" --prompt "energetic summer road trip" --min_songs 40 --llm_mode openai --llm_model gpt-4o-mini
```
- If arguments are omitted, the script will use its defaults or prompt for them interactively if needed. The default playlist name is "naviDJ".
- You can set the default LLM backend and model in `secrets.txt` under the `[llm]` section. Command-line arguments override these defaults.
- Supports `--llm_mode` values `openai`, `ollama`, and `custom` (for an OpenAI-compatible API).

#### portLibrary.py
Sync playlists and likes between Spotify and Subsonic:
```
python portLibrary.py --sync-starred y --sync-playlists y --import-liked y --import-playlists y --playlists "Playlist1,Playlist2"
```
- If arguments are omitted, the script will prompt for them interactively.
- **Note**: By default, this script ignores syncing playlists named `naviDJ`. When syncing TO Spotify, it only adds "missing" songs (it does not replace existing content). If you use a different DJ playlist name, repeated syncs will append.

#### portGenres.py
Update genres for your local music library using MusicBrainz:
```
python portGenres.py /path/to/your/music --dry-run
```
- Omit `--dry-run` to actually write changes.
- This script runs against your music files and updates the `GENRE` tag using MusicBrainz artist/album tags. It falls back to your existing tag if MB returns nothing. The DJ script benefits from richer and more granular genres, but it will work with your existing metadata too.

---

## Troubleshooting

### Common Issues

**"INVALID_CLIENT: Invalid redirect URI"**
- Make sure your Spotify app's redirect URI exactly matches the `REDIRECT_URI` in your `secrets.txt`
- Update both your Spotify app settings and `secrets.txt` if you change ports

**"Spotify credentials not configured"**
- Ensure your `secrets.txt` has the correct `CLIENT_ID` and `CLIENT_SECRET` values
- Check that the Spotify section exists in your configuration

**Scripts not working**
- Verify your `secrets.txt` is properly configured
- Check that your Subsonic/Navidrome server is accessible
- Ensure you have the required Python packages installed

### Getting Help

- **Never commit your `secrets.txt` file!** It is gitignored by default.
- For detailed options, see the top of each script or run with `--help`.
- Check the real-time output in the web app for detailed error messages.

---

## License
GPL-3.0
