from flask import Flask, render_template, request, jsonify, Response, send_from_directory, redirect, url_for, session
import configparser
import os
import sys
from threading import Thread, Lock
from datetime import datetime, timezone
from queue import Queue
import time
from functools import partial
import subprocess
import requests
from openai import OpenAI
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
from ollama_utils import normalize_ollama_url

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management

# Import your existing scripts
import naviDJ

# Global queue for script output
output_queues = {}

# Popen handles for currently-running naviDJ subprocesses, keyed by task_id,
# so a run can be cancelled from the browser (see /cancel_dj/<task_id>).
dj_processes = {}

# Last N naviDJ runs (prompt, decision log, tracks, feedback), persisted so
# runs can be reviewed/annotated from the History tab.
HISTORY_FILE = os.path.join(os.path.dirname(__file__), 'playlist_history.json')
HISTORY_MAX = 10
# ponytail: in-process lock only; enough for the single-process Flask dev
# server, add file locking if this ever runs multi-process.
history_lock = Lock()


def read_history():
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def write_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


# Config keys (case-insensitive, matched by key name regardless of section)
# that hold secret values. These are never sent to the browser and are only
# overwritten on save if the submitted value is non-blank.
SECRET_KEY_NAMES = {"openai_key", "api_key", "password", "client_secret"}


def is_secret_key(key):
    return (key or "").lower() in SECRET_KEY_NAMES


def read_secrets():
    secrets = configparser.ConfigParser()
    secrets.read('secrets.txt')
    return secrets


def masked_secrets():
    """
    Build a safe-for-the-browser view of secrets.txt: secret values (API keys,
    passwords, client secrets - see SECRET_KEY_NAMES) are never included, only
    a boolean flag ("<section>.<key>") indicating whether one is currently
    configured. Non-secret settings pass through unchanged.
    """
    secrets = read_secrets()
    masked = configparser.ConfigParser()
    flags = {}
    for section in secrets.sections():
        masked.add_section(section)
        for key, value in secrets[section].items():
            if is_secret_key(key):
                flags[f"{section}.{key}"] = bool(value)
                masked[section][key] = ""
            else:
                masked[section][key] = value
    return masked, flags


def write_secrets(config_data):
    """
    Merge submitted config into secrets.txt. Secret fields (see
    is_secret_key) are only overwritten when the submitted value is
    non-blank - the browser never receives real secret values, so a blank
    submission means "leave this one alone", not "clear it". Non-secret
    settings round-trip normally, including intentional clears.
    """
    secrets = configparser.ConfigParser()
    current = read_secrets()

    # Start from the existing on-disk config.
    for section in current.sections():
        secrets.add_section(section)
        for key, value in current[section].items():
            secrets[section][key] = value

    # Layer submitted values on top.
    for section, fields in (config_data or {}).items():
        if not isinstance(fields, dict):
            continue
        if not secrets.has_section(section):
            secrets.add_section(section)
        for key, val in fields.items():
            val = "" if val is None else str(val)
            if is_secret_key(key) and val == "":
                continue  # blank secret field = keep existing value
            secrets[section][key] = val

    with open('secrets.txt', 'w') as f:
        secrets.write(f)

def get_spotify_oauth():
    """Create Spotify OAuth object with current configuration."""
    secrets = read_secrets()
    if not secrets.has_section("spotify"):
        return None
        
    client_id = secrets.get("spotify", "client_id", fallback=None)
    client_secret = secrets.get("spotify", "client_secret", fallback=None)
    redirect_uri = secrets.get("spotify", "redirect_uri", fallback="http://localhost:5000/callback")
    scope = secrets.get("spotify", "scope", fallback="user-read-private user-read-playback-state user-library-read user-library-modify playlist-modify-public playlist-modify-private playlist-read-private")
    cache_path = secrets.get("spotify", "cache_path", fallback=".cache-spotify")
    
    if not client_id or not client_secret:
        return None
    
    return SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scope,
        cache_path=cache_path,
        show_dialog=False
    )

def get_valid_token(sp_oauth):
    """Retrieve cached token and refresh it if expired.
    If refresh fails due to 'invalid_grant', discard cached token.
    """
    if not sp_oauth:
        return None
    token_info = sp_oauth.get_cached_token()
    if token_info and sp_oauth.is_token_expired(token_info):
        try:
            token_info = sp_oauth.refresh_access_token(token_info["refresh_token"])
        except Exception as e:
            if "invalid_grant" in str(e).lower():
                cache_path = sp_oauth.cache_path
                if os.path.exists(cache_path):
                    try:
                        os.remove(cache_path)
                    except Exception:
                        pass
            return None
    return token_info

def check_spotify_auth():
    """Check if user is authenticated with Spotify."""
    sp_oauth = get_spotify_oauth()
    if not sp_oauth:
        return {"authenticated": False, "error": "Spotify credentials not configured"}
    
    try:
        token_info = get_valid_token(sp_oauth)
        
        if token_info:
            # Test the token by making a simple API call
            sp = spotipy.Spotify(auth=token_info["access_token"])
            user = sp.current_user()
            if user:
                return {
                    "authenticated": True, 
                    "user": user.get("display_name", "Unknown"),
                    "email": user.get("email", "")
                }
            else:
                return {"authenticated": False, "error": "Failed to get user info"}
        else:
            return {"authenticated": False, "error": "No cached token found"}
    except Exception as e:
        return {"authenticated": False, "error": str(e)}

def get_available_models(api_type, api_key=None, base_url=None):
    """Fetch available models from the specified API provider."""
    try:
        if api_type == "openai":
            if not api_key:
                return {"error": "OpenAI API key required"}
            client = OpenAI(api_key=api_key)
            models = client.models.list()
            return [model.id for model in models.data]
        elif api_type == "ollama":
            if not base_url:
                return {"error": "Ollama base URL required"}
            ollama_base = normalize_ollama_url(base_url)
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            # Try /tags first (Ollama native)
            try:
                response = requests.get(f"{ollama_base}/tags", headers=headers, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    return [model["name"] for model in data.get("models", [])]
            except Exception:
                pass
                
            # Try /models as fallback (Open WebUI / OpenAI compatible)
            try:
                response = requests.get(f"{ollama_base}/models", headers=headers, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    # OpenAI format is usually a list under 'data'
                    if isinstance(data, dict) and "data" in data:
                        return [model["id"] for model in data["data"]]
                    # Open WebUI fallback format
                    elif isinstance(data, dict) and "models" in data:
                        return [model["name"] for model in data.get("models", [])]
                    elif isinstance(data, list):
                        return [model.get("id") or model.get("name") for model in data]
            except Exception as e:
                return {"error": f"Failed to fetch models from both /tags and /models: {str(e)}"}
                
            error_msg = f"Failed to fetch models. Status: {response.status_code}"
            if 'response' in locals():
                try:
                    error_msg += f" - {response.text}"
                except:
                    pass
            return {"error": error_msg}
        elif api_type == "custom":
            if not base_url or not api_key:
                return {"error": "Custom API base URL and API key required"}
            try:
                client = OpenAI(api_key=api_key, base_url=base_url)
                models = client.models.list()
                return [model.id for model in models.data]
            except Exception as e:
                return {"error": f"Failed to fetch models from custom API: {str(e)}"}
        else:
            return {"error": "Invalid API type"}
    except Exception as e:
        return {"error": f"Error fetching models: {str(e)}"}

def script_output_reader(queue, process):
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            queue.put(output.strip())
    process.stdout.close()

@app.route('/')
def index():
    # Handle Spotify OAuth callback if code is present
    if request.args.get('code'):
        return spotify_callback()

    config, secret_flags = masked_secrets()
    return render_template('index.html', config=config, secret_flags=secret_flags)

@app.route('/update_config', methods=['POST'])
def update_config():
    config_data = request.json
    try:
        write_secrets(config_data)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/run_dj', methods=['POST'])
def run_dj():
    data = request.json or {}
    queue = Queue()
    task_id = f"dj_{time.time()}"
    output_queues[task_id] = queue

    def run():
        captured = []
        process = None
        try:
            secrets = read_secrets()
            args = [sys.executable, '-u', os.path.join(os.path.dirname(__file__), 'naviDJ.py')]
            if data.get('playlist_name'):
                args += ['--playlist_name', str(data.get('playlist_name'))]
            if data.get('prompt'):
                args += ['--prompt', str(data.get('prompt'))]
            if data.get('min_songs'):
                args += ['--min_songs', str(data.get('min_songs'))]
            # Per-run overrides of naviDJ's own --llm_mode/--llm_model args
            # (falls back to secrets.txt when not provided).
            if data.get('llm_mode'):
                args += ['--llm_mode', str(data.get('llm_mode'))]
            if data.get('llm_model'):
                args += ['--llm_model', str(data.get('llm_model'))]

            # Use chunk_size from config
            chunk_size = secrets.get('llm', 'chunk_size', fallback='500')
            args += ['--chunk_size', str(chunk_size)]

            try:
                process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            except Exception as e:
                queue.put(f"ERROR: Failed to start naviDJ: {e}")
                return

            dj_processes[task_id] = process

            if process.stdout:
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        line = output.strip()
                        queue.put(line)
                        captured.append(line)

            rc = process.poll()
            if rc not in (0, None):
                err_line = f"ERROR: naviDJ exited with code {rc}"
                captured.append(err_line)  # persist to history log (was queue-only)
                queue.put(err_line)

            playlist = None
            for line in reversed(captured):
                if line.startswith('PLAYLIST_JSON:'):
                    try:
                        playlist = json.loads(line[len('PLAYLIST_JSON:'):])
                    except ValueError:
                        pass
                    break
            tracks = (playlist or {}).get('tracks') or []
            entry = {
                'id': task_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'prompt': str(data.get('prompt', '')),
                'playlist_name': (playlist or {}).get('playlist_name') or str(data.get('playlist_name', '')),
                'llm_mode': str(data.get('llm_mode') or secrets.get('llm', 'mode', fallback='')),
                'llm_model': str(data.get('llm_model') or secrets.get('llm', 'model', fallback='')),
                'min_songs': str(data.get('min_songs', '')),
                'track_count': len(tracks),
                'tracks': tracks,
                'log': captured,  # full decision process as emitted by naviDJ
                'success': rc == 0 and playlist is not None,
                'feedback': None,
            }
            with history_lock:
                history = read_history()
                history.append(entry)
                write_history(history[-HISTORY_MAX:])
        except Exception as e:
            queue.put(f"ERROR: {e}")
        finally:
            dj_processes.pop(task_id, None)
            queue.put(None)  # Signal completion

    Thread(target=run).start()
    return jsonify({"task_id": task_id})


@app.route('/cancel_dj/<task_id>', methods=['POST'])
def cancel_dj(task_id):
    """Terminate a running naviDJ subprocess started by /run_dj."""
    process = dj_processes.get(task_id)
    if not process:
        return jsonify({"status": "not_found"}), 404
    try:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    q = output_queues.get(task_id)
    if q:
        q.put("ERROR: Run cancelled by user.")
    return jsonify({"status": "cancelled"})


@app.route('/playlist_history')
def playlist_history():
    with history_lock:
        history = read_history()
    history.reverse()  # most recent first
    return jsonify(history)


@app.route('/playlist_history/<entry_id>/feedback', methods=['POST'])
def playlist_history_feedback(entry_id):
    text = ((request.get_json(silent=True) or {}).get('text') or '').strip()
    if not text:
        return jsonify({'error': 'Feedback text required'}), 400
    with history_lock:
        history = read_history()
        for entry in history:
            if entry.get('id') == entry_id:
                entry['feedback'] = {'text': text, 'at': datetime.now(timezone.utc).isoformat()}
                write_history(history)
                return jsonify(entry)
    return jsonify({'error': 'Entry not found'}), 404

@app.route('/run_library', methods=['POST'])
def run_library():
    data = request.json or {}
    queue = Queue()
    task_id = f"lib_{time.time()}"
    output_queues[task_id] = queue

    def run():
        try:
            args = [sys.executable, '-u', os.path.join(os.path.dirname(__file__), 'portLibrary.py')]
            if data.get('sync_starred'):
                args += ['--sync-starred', str(data.get('sync_starred'))]
            if data.get('sync_playlists'):
                args += ['--sync-playlists', str(data.get('sync_playlists'))]
            if data.get('import_liked'):
                args += ['--import-liked', str(data.get('import_liked'))]
            if data.get('import_playlists'):
                args += ['--import-playlists', str(data.get('import_playlists'))]
            if data.get('playlists'):
                args += ['--playlists', str(data.get('playlists'))]

            process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if process.stdout:
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        queue.put(output.strip())
        except Exception as e:
            queue.put(f"ERROR: {str(e)}")
        finally:
            queue.put(None)  # Signal completion

    Thread(target=run).start()
    return jsonify({"task_id": task_id})

@app.route('/stream/<task_id>')
def stream(task_id):
    def generate():
        queue = output_queues.get(task_id)
        if not queue:
            return
            
        while True:
            output = queue.get()
            if output is None:  # End signal
                break
            yield f"data: {output}\n\n"
            
        # Cleanup
        del output_queues[task_id]
        
    return Response(generate(), mimetype='text/event-stream')

@app.route('/get_models/<api_type>')
def get_models(api_type):
    """Get available models for the specified API type."""
    secrets = read_secrets()
    
    if api_type == "openai":
        api_key = secrets.get("openai", "openai_key", fallback=None)
        return jsonify(get_available_models("openai", api_key=api_key))
    elif api_type == "ollama":
        base_url = secrets.get("ollama", "ollama_base", fallback=None)
        api_key = secrets.get("ollama", "api_key", fallback=None)
        return jsonify(get_available_models("ollama", api_key=api_key, base_url=base_url))
    elif api_type == "custom":
        api_key = secrets.get("custom", "api_key", fallback=None)
        base_url = secrets.get("custom", "base_url", fallback=None)
        return jsonify(get_available_models("custom", api_key=api_key, base_url=base_url))
    else:
        return jsonify({"error": "Invalid API type"})

@app.route('/get_embedding_models/<api_type>')
def get_embedding_models(api_type):
    """Get available embedding models for the specified API type."""
    secrets = read_secrets()
    
    if api_type == "ollama":
        base_url = secrets.get("ollama", "ollama_base", fallback=None)
        if not base_url:
            return jsonify({"error": "Ollama base URL not configured"})
        
        try:
            # Normalize the URL for Ollama API calls
            ollama_base = normalize_ollama_url(base_url)
            api_key = secrets.get("ollama", "api_key", fallback=None)
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            all_models = []
            
            # Try /tags first (Ollama native)
            try:
                response = requests.get(f"{ollama_base}/tags", headers=headers, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    all_models = [model["name"] for model in data.get("models", [])]
            except Exception:
                pass
            
            # Try /models if no models found yet
            if not all_models:
                try:
                    response = requests.get(f"{ollama_base}/models", headers=headers, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, dict) and "data" in data:
                            all_models = [model["id"] for model in data["data"]]
                        elif isinstance(data, dict) and "models" in data:
                            all_models = [model["name"] for model in data.get("models", [])]
                        elif isinstance(data, list):
                            all_models = [model.get("id") or model.get("name") for model in data]
                except Exception:
                    pass
            
            if not all_models:
                return jsonify({"error": "Failed to fetch models from both /tags and /models"})
            
            # Filter for models containing "embed" in the name
            embedding_models = [model for model in all_models if "embed" in model.lower()]
            return jsonify(embedding_models)
            
        except Exception as e:
            return jsonify({"error": f"Error fetching embedding models: {str(e)}"})
    
    elif api_type == "openai":
        # Return hardcoded list of OpenAI embedding models
        openai_embedding_models = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002"
        ]
        return jsonify(openai_embedding_models)
    
    elif api_type == "custom":
        # Custom API doesn't support embeddings
        return jsonify([])
    
    else:
        return jsonify({"error": "Invalid API type"})

@app.route('/spotify/login')
def spotify_login():
    """Initiate Spotify OAuth login."""
    sp_oauth = get_spotify_oauth()
    if not sp_oauth:
        return jsonify({"error": "Spotify credentials not configured"}), 400
    
    auth_url = sp_oauth.get_authorize_url()
    return jsonify({"auth_url": auth_url})

@app.route('/spotify/callback')
def spotify_callback():
    """Handle Spotify OAuth callback."""
    sp_oauth = get_spotify_oauth()
    if not sp_oauth:
        return jsonify({"error": "Spotify credentials not configured"}), 400
    
    try:
        code = request.args.get('code')
        if not code:
            return jsonify({"error": "No authorization code received"}), 400
        
        token_info = sp_oauth.get_access_token(code)
        if token_info:
            return redirect('/?spotify_auth=success')
        else:
            return redirect('/?spotify_auth=error')
    except Exception as e:
        return redirect('/?spotify_auth=error')

@app.route('/spotify/logout', methods=['GET', 'POST'])
def spotify_logout():
    """Logout from Spotify by clearing cached token."""
    sp_oauth = get_spotify_oauth()
    if sp_oauth:
        try:
            # Clear the cache file - use the cache_path from secrets
            secrets = read_secrets()
            cache_path = secrets.get("spotify", "cache_path", fallback=".cache-spotify")
            if os.path.exists(cache_path):
                os.remove(cache_path)
        except Exception as e:
            pass  # Ignore errors when clearing cache
    
    return jsonify({"status": "success"})

@app.route('/spotify/auth_status')
def spotify_auth_status():
    """Check Spotify authentication status."""
    return jsonify(check_spotify_auth())

@app.route('/spotify/playlists')
def spotify_playlists():
    """Return user's owned Spotify playlists (name/id/track count)."""
    sp_oauth = get_spotify_oauth()
    if not sp_oauth:
        return jsonify({"error": "Spotify credentials not configured"}), 400
    try:
        token_info = get_valid_token(sp_oauth)
        if not token_info:
            return jsonify({"error": "Not authenticated"}), 401

        sp = spotipy.Spotify(auth=token_info["access_token"])
        me = sp.current_user()
        if not me:
            return jsonify({"error": "Failed to fetch user"}), 500

        user_id = me.get("id")
        playlists = []
        limit = 50
        offset = 0
        while True:
            resp = sp.current_user_playlists(limit=limit, offset=offset)
            items = resp.get('items', [])
            # filter to only owned by the current user
            for pl in items:
                owner = (pl.get('owner') or {}).get('id')
                if owner == user_id:
                    playlists.append({
                        "id": pl.get('id'),
                        "name": pl.get('name'),
                        "tracks_total": (pl.get('tracks') or {}).get('total', 0)
                    })
            if len(items) < limit:
                break
            offset += limit

        return jsonify(playlists)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_config')
def get_config():
    """
    Get current configuration for the frontend. Secret values (API keys,
    passwords, client secrets) are never included - only a boolean flag per
    field indicating whether one is currently set. See masked_secrets().
    """
    config, secret_flags = masked_secrets()
    result = {}
    for section in config.sections():
        result[section] = dict(config[section])
    result["_secrets_set"] = secret_flags
    return jsonify(result)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, ''), 'DJ.png', mimetype='image/png')

@app.route('/DJ.png')
def dj_icon():
    return send_from_directory(os.path.join(app.root_path, ''), 'DJ.png', mimetype='image/png')

if __name__ == '__main__':
    # Werkzeug's debug mode enables the interactive debugger, which is a
    # remote-code-execution risk if this port is ever reachable beyond
    # localhost. Default to off; opt in explicitly via FLASK_DEBUG=1 for
    # local development.
    debug_mode = os.environ.get('FLASK_DEBUG', '').strip().lower() in ('1', 'true', 'yes', 'on')
    app.run(debug=debug_mode)
