# naviFy Tools
# A DJ for Navidrome and Spotify <-> Navidrome/Subsonic Playlist Syncing

A set of utilities for syncing playlists between Subsonic/Navidrome and Spotify and an AI powered DJ script to use with your music server to generate playlists as you please.

## Features
- **naviDJ.py**: AI-powered playlist generator for Subsonic/Navidrome using OpenAI or Ollama LLMs.
- **portLibrary.py**: Syncs starred/liked songs and playlists between Spotify and Subsonic/Navidrome (both directions).
- **portGenres.py**: Updates local music file genres using MusicBrainz tags. The DJ script uses genres to filter songs by, and more detailed genres, such as those from MusicBrainz, helps it filter more effectively, but by no means am I asking you to overwrite all your genres on your files to use these scripts. I'd be interested to see how the DJ performs with different types of metadata and how people tag their genres.
- **Web App**: Modern web interface for easy access to all tools with real-time output and configuration management.

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
- **Spotify OAuth Integration**: The web app now includes built-in Spotify authentication! Users can log in with Spotify directly from the browser, and the app will automatically handle token management and refresh.
- **Configuration Management**: Edit your `secrets.txt` configuration from the web UI
- **Script Execution**: Run the AI DJ (naviDJ.py) and Library Porter (portLibrary.py) from your browser
- **Real-time Output**: View real-time output from scripts in the browser
- You must have a valid `secrets.txt` file set up before using the web app

> **Heads up:**
> You can serve this Flask app and, using tools like [Tailscale](https://tailscale.com/), make it accessible over HTTPS. This enables installable PWA (Progressive Web App) features on mobile and desktop browsers.

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
     - **Redirect URI**: `http://localhost:5000/` (for local development)
     - **Website**: Can be left blank or set to your domain
   - Click "Save"

2. **Get Your Credentials:**
   - After creating the app, you'll see your **Client ID** and **Client Secret**
   - Copy these values to your `secrets.txt` file

3. **Configure Redirect URIs:**
   - In your Spotify app settings, add the redirect URI that matches your setup:
     - **Local development**: `http://localhost:5000/`
     - **Custom domain**: `https://yourdomain.com/` (if hosting publicly)
     - **Custom port**: `http://localhost:8080/` (if using a different port)
   - **Important**: The redirect URI in your Spotify app settings must exactly match the `REDIRECT_URI` in your `secrets.txt` file

4. **Update Your Configuration:**
   - In your `secrets.txt`, set:
     ```
     [spotify]
     CLIENT_ID = your-client-id-here
     CLIENT_SECRET = your-client-secret-here
     REDIRECT_URI = http://localhost:5000/
     ```

**Note:** If you change the port or domain where you host the Flask app, remember to update both your Spotify app settings and the `REDIRECT_URI` in your `secrets.txt` file.

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
   - **naviDJ Tab**: Generate AI playlists with custom prompts
   - **Library Porter Tab**: Sync playlists between Spotify and your music server
   - **Spotify Authentication**: Login with Spotify directly in the browser

### Command Line Usage

#### naviDJ.py
Generate a playlist using AI:
```
python naviDJ.py --playlist_name "My Playlist" --prompt "energetic summer road trip" --min_songs 40 --llm_mode openai --llm_model gpt-4o-mini
```
- If arguments are omitted, the script will use its defaults or prompt for them interactively if needed. The default playlist name is "naviDJ".
- You can set the default LLM backend and model in `secrets.txt` under the `[llm]` section. Command-line arguments override these defaults.

#### portLibrary.py
Sync playlists and likes between Spotify and Subsonic:
```
python portLibrary.py --sync-starred y --sync-playlists y --import-liked y --import-playlists y --playlists "Playlist1,Playlist2"
```
- If arguments are omitted, the script will prompt for them interactively.
- **Note**: By default, this script is designed to ignore syncing playlists named 'naviDJ'. When syncing TO Spotify, the script will simply add "missing" songs, so if you use the DJ with a different playlist name, instead of porting the updated version to Spotify and replacing it, it will simply keep appending the songs to the playlist. 

#### portGenres.py
Update genres for your local music library using MusicBrainz:
```
python portGenres.py /path/to/your/music --dry-run
```
- Omit `--dry-run` to actually write changes.
- This script is meant to run on your music files and edit the genre tags for them, grabbing the information from MusicBrainz and defaulting to whatever you have in the file already as a fallback. The DJ script was optimized based on the genre variety that is provided by this script, however it should work regardless so don't feel the need to update your metadata unnecessarily. The way the DJ script filters songs will be impacted by the diversity of the genres you tagged your music with, feel free to share your experiences with how it works!

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
