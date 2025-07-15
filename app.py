from flask import Flask, render_template, request, jsonify, Response
import configparser
import os
import sys
from threading import Thread
from queue import Queue
import time
from functools import partial
import subprocess

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
    
    with open('secrets.txt', 'w') as f:
        secrets.write(f)

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
            if data.get('llm_mode'):
                args += ['--llm_mode', str(data.get('llm_mode'))]
            if data.get('llm_model'):
                args += ['--llm_model', str(data.get('llm_model'))]
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

if __name__ == '__main__':
    app.run(debug=True)
