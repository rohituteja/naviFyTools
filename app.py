from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import configparser
import os
import sys
from threading import Thread
from queue import Queue
import time
from functools import partial
import subprocess
import requests
from openai import OpenAI

app = Flask(__name__)

# Import your existing scripts
import naviDJ
import portLibrary

# Global queue for script output
output_queues = {}

def read_secrets():
    secrets = configparser.ConfigParser()
    secrets.read('secrets.txt')
    return secrets

def write_secrets(config_data):
    secrets = configparser.ConfigParser()
    current = read_secrets()  # Read existing config
    
    # Update with new values while preserving existing structure
    for section in current.sections():
        if section not in secrets:
            secrets.add_section(section)
        for key in current[section]:
            if section in config_data and key in config_data[section]:
                secrets[section][key] = config_data[section][key]
            else:
                secrets[section][key] = current[section][key]
    
    # Add new sections if they don't exist
    for section in config_data:
        if section not in secrets:
            secrets.add_section(section)
        for key in config_data[section]:
            secrets[section][key] = config_data[section][key]
    
    with open('secrets.txt', 'w') as f:
        secrets.write(f)

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
            # Extract the base URL without the /v1 suffix for Ollama API
            ollama_base = base_url.replace("/v1", "")
            response = requests.get(f"{ollama_base}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            else:
                return {"error": f"Failed to fetch models: {response.status_code}"}
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
    secrets = read_secrets()
    return render_template('index.html', config=secrets)

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
    data = request.json
    queue = Queue()
    task_id = f"dj_{time.time()}"
    output_queues[task_id] = queue

    def run():
        try:
            args = [sys.executable, os.path.join(os.path.dirname(__file__), 'naviDJ.py')]
            if data.get('playlist_name'):
                args += ['--playlist_name', str(data.get('playlist_name'))]
            if data.get('prompt'):
                args += ['--prompt', str(data.get('prompt'))]
            if data.get('min_songs'):
                args += ['--min_songs', str(data.get('min_songs'))]
            process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    queue.put(output.strip())
        finally:
            queue.put(None)  # Signal completion

    Thread(target=run).start()
    return jsonify({"task_id": task_id})

@app.route('/run_library', methods=['POST'])
def run_library():
    data = request.json
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
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    queue.put(output.strip())
        except Exception as e:
            queue.put(f"Error: {str(e)}")
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
        return jsonify(get_available_models("ollama", base_url=base_url))
    elif api_type == "custom":
        api_key = secrets.get("custom", "api_key", fallback=None)
        base_url = secrets.get("custom", "base_url", fallback=None)
        return jsonify(get_available_models("custom", api_key=api_key, base_url=base_url))
    else:
        return jsonify({"error": "Invalid API type"})

@app.route('/get_config')
def get_config():
    """Get current configuration for the frontend."""
    secrets = read_secrets()
    config = {}
    
    # Convert ConfigParser to dict
    for section in secrets.sections():
        config[section] = dict(secrets[section])
    
    return jsonify(config)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, ''), 'DJ.png', mimetype='image/png')

@app.route('/DJ.png')
def dj_icon():
    return send_from_directory(os.path.join(app.root_path, ''), 'DJ.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
