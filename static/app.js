// Configuration form handling
document.getElementById('configForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const config = {};
    
    for (let [key, value] of formData.entries()) {
        const [section, field] = key.split('.');
        if (!config[section]) config[section] = {};
        config[section][field] = value;
    }
    
    try {
        const response = await fetch('/update_config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const result = await response.json();
        if (result.status === 'success') {
            alert('Configuration updated successfully!');
        } else {
            alert('Error updating configuration: ' + result.message);
        }
    } catch (error) {
        alert('Error updating configuration: ' + error);
    }
});

// LLM Configuration Management
let currentLlmMode = 'openai';
let currentLlmModel = '';
let currentEmbeddingModel = '';

// Function to show/hide API configuration sections
function toggleApiConfig(mode) {
    // Hide all config sections
    document.getElementById('openaiConfig').style.display = 'none';
    document.getElementById('ollamaConfig').style.display = 'none';
    document.getElementById('customConfig').style.display = 'none';
    
    // Show the selected config section
    if (mode === 'openai') {
        document.getElementById('openaiConfig').style.display = 'block';
    } else if (mode === 'ollama') {
        document.getElementById('ollamaConfig').style.display = 'block';
    } else if (mode === 'custom') {
        document.getElementById('customConfig').style.display = 'block';
    }
}

// Function to fetch and populate models
async function fetchModels(apiType) {
    const modelSelect = document.getElementById('llmModelSelect');
    modelSelect.innerHTML = '<option value="">Loading models...</option>';
    
    try {
        const response = await fetch(`/get_models/${apiType}`);
        const result = await response.json();
        
        if (result.error) {
            modelSelect.innerHTML = `<option value="">Error: ${result.error}</option>`;
            return;
        }
        
        if (Array.isArray(result)) {
            modelSelect.innerHTML = '<option value="">Select a model...</option>';
            result.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                if (model === currentLlmModel) {
                    option.selected = true;
                }
                modelSelect.appendChild(option);
            });
            
            // If we have a current model but it wasn't found in the list, 
            // try to set it anyway (in case of slight naming differences)
            if (currentLlmModel && !modelSelect.value) {
                const existingOption = Array.from(modelSelect.options).find(opt => 
                    opt.value.toLowerCase().includes(currentLlmModel.toLowerCase()) ||
                    currentLlmModel.toLowerCase().includes(opt.value.toLowerCase())
                );
                if (existingOption) {
                    existingOption.selected = true;
                    currentLlmModel = existingOption.value;
                }
            }
        } else {
            modelSelect.innerHTML = '<option value="">No models available</option>';
        }
    } catch (error) {
        modelSelect.innerHTML = '<option value="">Error loading models</option>';
        console.error('Error fetching models:', error);
    }
}

// Function to fetch and populate embedding models
async function fetchEmbeddingModels(apiType) {
    const embeddingSelect = document.getElementById('embeddingModelSelect');
    const embeddingContainer = document.getElementById('embeddingModelContainer');
    
    // Show/hide based on API type
    if (apiType === 'ollama' || apiType === 'openai') {
        embeddingContainer.style.display = 'block';
        embeddingSelect.innerHTML = '<option value="">Loading models...</option>';
        
        // Update the name attribute based on API type
        embeddingSelect.name = `${apiType}.embedding_model`;
        embeddingSelect.setAttribute('data-section', apiType);
        embeddingSelect.setAttribute('data-key', 'embedding_model');
        
        try {
            const response = await fetch(`/get_embedding_models/${apiType}`);
            const result = await response.json();
            
            if (result.error) {
                embeddingSelect.innerHTML = `<option value="">Error: ${result.error}</option>`;
                return;
            }
            
            if (Array.isArray(result) && result.length > 0) {
                embeddingSelect.innerHTML = '<option value="">Select a model...</option>';
                result.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    if (model === currentEmbeddingModel) {
                        option.selected = true;
                    }
                    embeddingSelect.appendChild(option);
                });
                
                // If we have a current model but it wasn't found in the list, 
                // try to set it anyway
                if (currentEmbeddingModel && !embeddingSelect.value) {
                    const existingOption = Array.from(embeddingSelect.options).find(opt => 
                        opt.value.toLowerCase().includes(currentEmbeddingModel.toLowerCase()) ||
                        currentEmbeddingModel.toLowerCase().includes(opt.value.toLowerCase())
                    );
                    if (existingOption) {
                        existingOption.selected = true;
                        currentEmbeddingModel = existingOption.value;
                    }
                }
            } else {
                embeddingSelect.innerHTML = '<option value="">No embedding models available</option>';
            }
        } catch (error) {
            embeddingSelect.innerHTML = '<option value="">Error loading embedding models</option>';
            console.error('Error fetching embedding models:', error);
        }
    } else {
        embeddingContainer.style.display = 'none';
    }
}


// LLM mode change handler for config
document.getElementById('llmModeSelect')?.addEventListener('change', async function(e) {
    const mode = e.target.value;
    currentLlmMode = mode;
    toggleApiConfig(mode);
    await fetchModels(mode);
    await fetchEmbeddingModels(mode);
});

// LLM model change handler for config
document.getElementById('llmModelSelect')?.addEventListener('change', function(e) {
    currentLlmModel = e.target.value;
});

// Embedding model change handler for config
document.getElementById('embeddingModelSelect')?.addEventListener('change', function(e) {
    currentEmbeddingModel = e.target.value;
});

// Thinking toggle change handler
document.getElementById('thinkingToggle')?.addEventListener('change', async function(e) {
    const value = e.target.checked ? 'on' : 'off';
    try {
        const response = await fetch('/update_config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ llm: { thinking_enabled: value } })
        });
        const result = await response.json();
        if (result.status === 'success') {
            console.log('Thinking mode:', value);
        } else {
            e.target.checked = !e.target.checked;
            alert('Error updating configuration: ' + result.message);
        }
    } catch (error) {
        e.target.checked = !e.target.checked;
        console.error('Error updating thinking mode:', error);
    }
});


// --------------------------------------------------
// naviDJ tab: prompt history, structured progress, tracklist, cancel,
// per-run model override.
// --------------------------------------------------

const DJ_PROMPT_HISTORY_KEY = 'naviDJ_promptHistory';
const DJ_PROMPT_HISTORY_MAX = 8;

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str == null ? '' : String(str);
    return div.innerHTML;
}

function loadPromptHistory() {
    try {
        const raw = localStorage.getItem(DJ_PROMPT_HISTORY_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch (e) {
        return [];
    }
}

function savePromptToHistory(prompt) {
    if (!prompt || !prompt.trim()) return;
    const trimmed = prompt.trim();
    let history = loadPromptHistory().filter(p => p !== trimmed);
    history.unshift(trimmed);
    history = history.slice(0, DJ_PROMPT_HISTORY_MAX);
    try {
        localStorage.setItem(DJ_PROMPT_HISTORY_KEY, JSON.stringify(history));
    } catch (e) {
        // localStorage unavailable (private mode, quota, etc.) - not fatal
    }
    renderPromptHistory();
}

function renderPromptHistory() {
    const container = document.getElementById('djPromptHistory');
    if (!container) return;
    const history = loadPromptHistory();
    if (history.length === 0) {
        container.innerHTML = '';
        container.classList.add('d-none');
        return;
    }
    container.classList.remove('d-none');
    container.innerHTML = '<span class="prompt-history-label">recent:</span> ' +
        history.map((p, i) => `<button type="button" class="prompt-chip" data-idx="${i}">${escapeHtml(p.length > 40 ? p.slice(0, 40) + '…' : p)}</button>`).join(' ');
    container.querySelectorAll('.prompt-chip').forEach(btn => {
        btn.addEventListener('click', () => {
            const idx = parseInt(btn.getAttribute('data-idx'), 10);
            const h = loadPromptHistory();
            const promptInput = document.getElementById('djPromptInput');
            if (h[idx] !== undefined && promptInput) {
                promptInput.value = h[idx];
                promptInput.focus();
            }
        });
    });
}

// Ordered stages naviDJ.py emits via "STAGE: <name>" marker lines.
const DJ_STAGES = [
    'Fetching Library',
    'Gathering Metadata',
    'Semantic Pre-filtering',
    'Context Analysis',
    'Selecting Focus Metadata',
    'Filtering Candidates',
    'Generating Playlist',
    'Finalizing Playlist',
    'Uploading to Server',
    'Complete'
];

// Recognizable naviDJ.py stat lines, pulled out of the raw log into a clean
// summary panel. First match wins per line.
const DJ_STAT_PATTERNS = [
    { re: /^Using LLM backend: (\S+) \(model: (.+)\)$/, label: m => `Model: ${m[1]} / ${m[2]}` },
    { re: /^Library fetch complete: (\d+) songs/, label: m => `Library: ${m[1]} songs` },
    { re: /^Gathered metadata: (\d+) artists, (\d+) genres, (\d+) albums/, label: m => `Metadata: ${m[1]} artists, ${m[2]} genres, ${m[3]} albums` },
    { re: /^Implied era detected: (.+)$/, label: m => `Era bonus: ${m[1]}` },
    { re: /^Explicit artists identified: (.+)$/, label: m => `Explicit artists: ${m[1]}` },
    { re: /^Final candidate pool: (\d+) songs/, label: m => `Candidate pool: ${m[1]} songs` },
    { re: /^Variant dedup: (.+)$/, label: m => `Dedup: ${m[1]}` },
    { re: /^Final playlist generated: (\d+) tracks/, label: m => `Generated: ${m[1]} tracks` },
    { re: /^Artist cap \([^)]*\) enforced: (.+)$/, label: m => `Artist cap: ${m[1]}` },
    { re: /^Playlist '(.+)' successfully updated on server/, label: m => `Uploaded playlist: ${m[1]}` },
];

function setupDjForm() {
    const form = document.getElementById('djForm');
    if (!form) return;

    const submitBtn = document.getElementById('djSubmitBtn');
    const cancelBtn = document.getElementById('djCancelBtn');
    const promptInput = document.getElementById('djPromptInput');
    const progressPanel = document.getElementById('djProgressPanel');
    const stageListEl = document.getElementById('djStageList');
    const statsEl = document.getElementById('djStats');
    const resultPanel = document.getElementById('djResultPanel');
    const resultHeading = document.getElementById('djResultHeading');
    const trackListEl = document.getElementById('djTrackList');
    const rawLogToggle = document.getElementById('djRawLogToggle');
    const outputDiv = document.getElementById('djOutput');

    renderPromptHistory();

    if (rawLogToggle && outputDiv) {
        rawLogToggle.addEventListener('click', (e) => {
            e.preventDefault();
            const hidden = outputDiv.classList.toggle('d-none');
            rawLogToggle.innerHTML = hidden ? 'show raw log &#9656;' : 'hide raw log &#9662;';
        });
    }

    let currentTaskId = null;
    let statLines = [];
    let stageEls = {}; // stage name -> <li> element, rebuilt each run

    function resetRunUi() {
        statLines = [];
        stageEls = {};
        if (stageListEl) {
            stageListEl.innerHTML = '';
            DJ_STAGES.forEach(s => {
                const li = document.createElement('li');
                li.className = 'stage-item';
                const dot = document.createElement('span');
                dot.className = 'stage-dot';
                li.appendChild(dot);
                li.appendChild(document.createTextNode(s));
                stageListEl.appendChild(li);
                stageEls[s] = li;
            });
        }
        if (statsEl) statsEl.innerHTML = '';
        if (progressPanel) progressPanel.classList.remove('d-none');
        if (resultPanel) resultPanel.classList.add('d-none');
        if (trackListEl) trackListEl.innerHTML = '';
        if (outputDiv) {
            outputDiv.innerHTML = '';
            outputDiv.classList.add('d-none');
        }
        if (rawLogToggle) rawLogToggle.innerHTML = 'show raw log &#9656;';
    }

    function markStage(stageName) {
        const idx = DJ_STAGES.indexOf(stageName);
        if (idx === -1) return;
        DJ_STAGES.forEach((s, i) => {
            const li = stageEls[s];
            if (!li) return;
            li.classList.remove('stage-active', 'stage-done');
            if (i < idx || stageName === 'Complete') {
                li.classList.add('stage-done');
            } else if (i === idx) {
                li.classList.add('stage-active');
            }
        });
    }

    function addStat(html) {
        if (!statsEl) return;
        statLines.push(html);
        statsEl.innerHTML = statLines.map(t => `<div>${t}</div>`).join('');
    }

    function renderTracklist(payload) {
        if (!resultPanel || !trackListEl) return;
        const tracks = (payload && payload.tracks) || [];
        const name = (payload && payload.playlist_name) || 'playlist';
        resultPanel.classList.remove('d-none');
        if (resultHeading) resultHeading.textContent = `"${name}" — ${tracks.length} tracks`;
        trackListEl.innerHTML = tracks.map(t => {
            let line = `${escapeHtml(t.title || 'Unknown title')} — ${escapeHtml(t.artist || 'Unknown artist')}`;
            if (t.album) line += ` — ${escapeHtml(t.album)}`;
            if (t.year) line += ` (${escapeHtml(t.year)})`;
            return `<li>${line}</li>`;
        }).join('');
    }

    function setRunning(isRunning) {
        if (submitBtn) {
            submitBtn.disabled = isRunning;
            submitBtn.innerHTML = isRunning
                ? '<span class="spinner-border spinner-border-sm me-2"></span>generating...'
                : '<b>generate your mix</b>';
        }
        if (cancelBtn) {
            cancelBtn.classList.toggle('d-none', !isRunning);
            cancelBtn.disabled = false;
        }
    }

    if (cancelBtn) {
        cancelBtn.addEventListener('click', async () => {
            if (!currentTaskId) return;
            cancelBtn.disabled = true;
            try {
                await fetch(`/cancel_dj/${currentTaskId}`, { method: 'POST' });
            } catch (e) {
                // Stream's onerror handler will still clean up the UI even
                // if this request itself fails.
            }
        });
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(form);
        const data = {};
        for (let [key, value] of formData.entries()) {
            if (value === '') continue; // don't send blank optional overrides
            data[key] = value;
        }

        savePromptToHistory(promptInput ? promptInput.value : '');
        resetRunUi();
        setRunning(true);

        try {
            const response = await fetch('/run_dj', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await response.json();

            if (!result.task_id) {
                addStat(`<span class="log-error">Error: ${escapeHtml(result.error || 'Failed to start run.')}</span>`);
                setRunning(false);
                return;
            }

            currentTaskId = result.task_id;
            const eventSource = new EventSource(`/stream/${result.task_id}`);

            eventSource.onmessage = (event) => {
                const line = event.data;

                // PLAYLIST_JSON: hidden from the raw log, rendered as a tracklist.
                const plMatch = line.match(/^PLAYLIST_JSON:\s*(.+)$/);
                if (plMatch) {
                    try {
                        renderTracklist(JSON.parse(plMatch[1]));
                    } catch (err) {
                        console.error('Failed to parse PLAYLIST_JSON', err);
                    }
                    return; // don't echo to raw log
                }

                // STAGE: marker lines drive the progress list.
                const stageMatch = line.match(/^STAGE:\s*(.+)$/);
                if (stageMatch) {
                    markStage(stageMatch[1].trim());
                }

                // Known stat lines get pulled into the summary panel.
                for (const pattern of DJ_STAT_PATTERNS) {
                    const m = line.match(pattern.re);
                    if (m) {
                        addStat(pattern.label(m));
                        break;
                    }
                }

                const isError = /^error:/i.test(line.trim());
                if (isError) addStat(`<span class="log-error">${escapeHtml(line)}</span>`);

                if (outputDiv) {
                    const rendered = isError ? `<span class="log-error">${escapeHtml(line)}</span>` : escapeHtml(line);
                    const isProgressBar = (line.includes('|') && /\d+%\|/.test(line)) || line.includes('song/s');
                    if (isProgressBar) {
                        const lines = outputDiv.innerHTML.split('<br>');
                        if (lines.length > 1) {
                            lines[lines.length - 1] = rendered;
                            outputDiv.innerHTML = lines.join('<br>');
                        } else {
                            outputDiv.innerHTML = rendered + '<br>';
                        }
                    } else {
                        outputDiv.innerHTML += rendered + '<br>';
                    }
                    outputDiv.scrollTop = outputDiv.scrollHeight;
                }
            };

            eventSource.onerror = () => {
                eventSource.close();
                currentTaskId = null;
                setRunning(false);
            };
        } catch (error) {
            addStat(`<span class="log-error">Error: ${escapeHtml(String(error))}</span>`);
            setRunning(false);
        }
    });
}

// Helper function to handle script execution and output streaming
// (used for the library porter tab; the DJ tab has its own richer handler,
// see setupDjForm() below)
function handleScriptExecution(formId, outputId, endpoint) {
    document.getElementById(formId).addEventListener('submit', async (e) => {
        e.preventDefault();
        const form = e.target;
        const submitBtn = form.querySelector('button[type="submit"]');
        const outputDiv = document.getElementById(outputId);
        outputDiv.innerHTML = '';
        // Show output only when script is run
        outputDiv.classList.remove('d-none');

        // Check Spotify authentication for library porter
        if (formId === 'libraryForm') {
            if (!spotifyAuthStatus || !spotifyAuthStatus.authenticated) {
                const shouldContinue = confirm('You are not authenticated with Spotify. The library porter may not work correctly. Do you want to continue anyway?');
                if (!shouldContinue) {
                    return;
                }
            }
        }

        const formData = new FormData(form);
        const data = {};
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }
        // Convert toggle switches to 'y'/'n'
        if (formId === 'libraryForm') {
            const toggleMap = [
                ['sync_starred', 'toggleSyncStarred'],
                ['sync_playlists', 'toggleSyncPlaylists'],
                ['import_liked', 'toggleImportLiked'],
                ['import_playlists', 'toggleImportPlaylists']
            ];
            toggleMap.forEach(([key, id]) => {
                const el = document.getElementById(id);
                if (el) {
                    data[key] = el.checked ? 'y' : 'n';
                }
            });
        }
        // Append selected playlists from chooser as comma-separated names
        if (formId === 'libraryForm') {
            const list = document.getElementById('playlistList');
            if (list) {
                const checked = Array.from(list.querySelectorAll('input[type="checkbox"][data-pl-name]:checked'))
                    .map(cb => cb.getAttribute('data-pl-name'));
                if (checked.length > 0) {
                    data['playlists'] = checked.join(',');
                }
            }
        }

        // Double-submit guard
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.dataset.originalText = submitBtn.dataset.originalText || submitBtn.innerHTML;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>running...';
        }
        const finishRun = () => {
            if (submitBtn) {
                submitBtn.disabled = false;
                if (submitBtn.dataset.originalText) submitBtn.innerHTML = submitBtn.dataset.originalText;
            }
        };

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await response.json();

            if (result.task_id) {
                const eventSource = new EventSource(`/stream/${result.task_id}`);

                eventSource.onmessage = (event) => {
                    let line = event.data;

                    // Show output if hidden (in case of async race)
                    outputDiv.classList.remove('d-none');

                    const isError = /^error:/i.test(line.trim());
                    // Detect progress bar lines (tqdm or similar)
                    const isProgressBar = (line.includes('|') && /\d+%\|/.test(line)) || line.includes('song/s') || line.includes('Building playlist:');
                    const rendered = isError ? `<span class="log-error">${line}</span>` : line;

                    if (isProgressBar) {
                        // Overwrite the last line
                        const lines = outputDiv.innerHTML.split('<br>');
                        if (lines.length > 1) {
                            lines[lines.length - 1] = rendered;
                            outputDiv.innerHTML = lines.join('<br>');
                        } else {
                            outputDiv.innerHTML = rendered + '<br>';
                        }
                    } else {
                        // Normal output: append as new line
                        outputDiv.innerHTML += rendered + '<br>';
                    }
                    outputDiv.scrollTop = outputDiv.scrollHeight;
                };

                eventSource.onerror = () => {
                    eventSource.close();
                    finishRun();
                };
            } else {
                finishRun();
                if (result.error) {
                    outputDiv.classList.remove('d-none');
                    outputDiv.innerHTML += `<span class="log-error">Error: ${result.error}</span><br>`;
                }
            }
        } catch (error) {
            outputDiv.classList.remove('d-none');
            outputDiv.innerHTML += `<span class="log-error">Error: ${error}</span><br>`;
            finishRun();
        }
    });
}

// Set up form handlers
setupDjForm();
handleScriptExecution('libraryForm', 'libraryOutput', '/run_library');

// Make Enter in the DJ prompt textarea submit the form
document.getElementById('djPromptInput').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        document.getElementById('djForm').requestSubmit();
    }
});

// Bootstrap tab handling
var triggerTabList = [].slice.call(document.querySelectorAll('a[data-bs-toggle="tab"]'))
triggerTabList.forEach(function (triggerEl) {
    var tabTrigger = new bootstrap.Tab(triggerEl)
    triggerEl.addEventListener('click', function (event) {
        event.preventDefault()
        tabTrigger.show()
    })
}) 

// Config dropdown triangle rotation
const configSection = document.getElementById('configSection');
const configTriangle = document.getElementById('configTriangle');
if (configSection && configTriangle) {
    configSection.addEventListener('show.bs.collapse', function () {
        configTriangle.style.transform = 'rotate(90deg)';
    });
    configSection.addEventListener('hide.bs.collapse', function () {
        configTriangle.style.transform = 'rotate(0deg)';
    });
    // Set initial state
    if (configSection.classList.contains('show')) {
        configTriangle.style.transform = 'rotate(90deg)';
    } else {
        configTriangle.style.transform = 'rotate(0deg)';
    }
} 

// DJ options dropdown triangle rotation and collapse logic
const djOptionsCollapse = document.getElementById('djOptionsCollapse');
const djOptionsTriangle = document.getElementById('djOptionsTriangle');
const djOptionsHeader = document.getElementById('djOptionsHeader');
if (djOptionsCollapse && djOptionsTriangle && djOptionsHeader) {
    djOptionsCollapse.addEventListener('show.bs.collapse', function () {
        djOptionsTriangle.style.transform = 'rotate(90deg)';
    });
    djOptionsCollapse.addEventListener('hide.bs.collapse', function () {
        djOptionsTriangle.style.transform = 'rotate(0deg)';
    });
    // Set initial state
    if (djOptionsCollapse.classList.contains('show')) {
        djOptionsTriangle.style.transform = 'rotate(90deg)';
    } else {
        djOptionsTriangle.style.transform = 'rotate(0deg)';
    }
    // By default, collapse on small screens
    if (window.innerWidth < 768) {
        const bsCollapse = bootstrap.Collapse.getOrCreateInstance(djOptionsCollapse);
        bsCollapse.hide();
    }
} 

// Spotify Authentication Management
let spotifyAuthStatus = null;
let playlistsLoaded = false;

async function maybeLoadOwnedPlaylists() {
    const chooser = document.getElementById('playlistChooser');
    const list = document.getElementById('playlistList');
    const search = document.getElementById('playlistSearch');
    const selectAll = document.getElementById('selectAllPlaylists');
    const manual = document.getElementById('playlistManualInput');
    if (!chooser || !list) return;
    if (!spotifyAuthStatus || !spotifyAuthStatus.authenticated) {
        chooser.style.display = 'none';
        if (manual) manual.style.display = '';
        return;
    }
    if (playlistsLoaded && list.children.length > 0) {
        chooser.style.display = 'block';
        if (manual) manual.style.display = 'none';
        return;
    }
    try {
        const resp = await fetch('/spotify/playlists');
        const data = await resp.json();
        if (Array.isArray(data)) {
            list.innerHTML = '';
            data.forEach(pl => {
                const id = `pl_${pl.id}`;
                const row = document.createElement('div');
                row.className = 'form-check';
                row.dataset.filterText = (pl.name || '').toLowerCase();
                row.innerHTML = `
                    <input class="form-check-input" type="checkbox" id="${id}" data-pl-name="${pl.name || ''}">
                    <label class="form-check-label" for="${id}">
                        ${pl.name || ''} <span class="text-muted">(${pl.tracks_total || 0})</span>
                    </label>`;
                list.appendChild(row);
            });
            chooser.style.display = 'block';
            if (manual) manual.style.display = 'none';
            playlistsLoaded = true;

            // Search filter
            if (search) {
                search.addEventListener('input', () => {
                    const q = search.value.toLowerCase().trim();
                    Array.from(list.children).forEach(row => {
                        const t = row.dataset.filterText || '';
                        row.style.display = t.includes(q) ? '' : 'none';
                    });
                });
            }
            // Select all
            if (selectAll) {
                selectAll.addEventListener('change', () => {
                    const visibleBoxes = Array.from(list.querySelectorAll('input[type="checkbox"]'))
                        .filter(cb => cb.closest('.form-check').style.display !== 'none');
                    visibleBoxes.forEach(cb => cb.checked = selectAll.checked);
                });
            }
        }
    } catch (e) {
        // Silently ignore; chooser stays hidden
        if (manual) manual.style.display = '';
    }
}

// Function to check Spotify authentication status
async function checkSpotifyAuth() {
    const authStatus = document.getElementById('spotifyAuthStatus');
    const authText = document.getElementById('spotifyAuthText');
    const authSpinner = document.getElementById('spotifyAuthSpinner');
    const loginBtn = document.getElementById('spotifyLoginBtn');
    const logoutBtn = document.getElementById('spotifyLogoutBtn');
    const userInfo = document.getElementById('spotifyUserInfo');
    const userName = document.getElementById('spotifyUserName');

    // Show spinner
    authSpinner.classList.remove('d-none');
    authText.textContent = 'checking authentication...';

    try {
        const response = await fetch('/spotify/auth_status');
        const result = await response.json();
        spotifyAuthStatus = result;

        if (result.authenticated) {
            authText.textContent = 'authenticated with spotify';
            authText.className = 'text-success';
            loginBtn.classList.add('d-none');
            logoutBtn.classList.remove('d-none');
            userInfo.classList.remove('d-none');
            userName.textContent = result.user || 'Unknown User';
        } else {
            authText.textContent = result.error || 'not authenticated';
            authText.className = 'text-warning';
            loginBtn.classList.remove('d-none');
            logoutBtn.classList.add('d-none');
            userInfo.classList.add('d-none');
        }
    } catch (error) {
        authText.textContent = 'error checking authentication';
        authText.className = 'text-danger';
        loginBtn.classList.remove('d-none');
        logoutBtn.classList.add('d-none');
        userInfo.classList.add('d-none');
    } finally {
        authSpinner.classList.add('d-none');
    }
}

// Spotify login handler
document.getElementById('spotifyLoginBtn')?.addEventListener('click', async function() {
    const loginBtn = document.getElementById('spotifyLoginBtn');
    const originalText = loginBtn.innerHTML;
    
    loginBtn.disabled = true;
    loginBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>connecting...';
    
    try {
        const response = await fetch('/spotify/login');
        const result = await response.json();
        
        if (result.auth_url) {
            // Redirect to Spotify authorization in the same window
            window.location.href = result.auth_url;
        } else {
            alert('Error: ' + (result.error || 'Failed to get authorization URL'));
            loginBtn.disabled = false;
            loginBtn.innerHTML = originalText;
        }
    } catch (error) {
        alert('Error connecting to Spotify: ' + error);
        loginBtn.disabled = false;
        loginBtn.innerHTML = originalText;
    }
});

// Spotify logout handler
document.getElementById('spotifyLogoutBtn')?.addEventListener('click', async function() {
    const logoutBtn = document.getElementById('spotifyLogoutBtn');
    const originalText = logoutBtn.innerHTML;
    
    logoutBtn.disabled = true;
    logoutBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>logging out...';
    
    try {
        const response = await fetch('/spotify/logout', { method: 'POST' });
        const result = await response.json();
        
        if (result.status === 'success') {
            await checkSpotifyAuth();
        } else {
            alert('Error logging out: ' + (result.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Error logging out: ' + error);
    } finally {
        logoutBtn.disabled = false;
        logoutBtn.innerHTML = originalText;
    }
});

// Check Spotify auth status when library tab is shown
document.querySelector('a[href="#libraryTab"]')?.addEventListener('click', function() {
    // Small delay to ensure tab is visible
    setTimeout(async () => {
        await checkSpotifyAuth();
        await maybeLoadOwnedPlaylists();
    }, 100);
});

// --------------------------------------------------
// history tab: last 10 naviDJ runs from the server, each expandable with
// tracklist, decision log, and a feedback box.
// --------------------------------------------------

function renderHistoryTracklist(tracks) {
    return (tracks || []).map(t => {
        let line = `${escapeHtml(t.title || 'Unknown title')} — ${escapeHtml(t.artist || 'Unknown artist')}`;
        if (t.album) line += ` — ${escapeHtml(t.album)}`;
        if (t.year) line += ` (${escapeHtml(t.year)})`;
        return `<li>${line}</li>`;
    }).join('');
}

function renderHistoryEntry(entry) {
    const panel = document.createElement('div');
    panel.className = 'dj-panel history-entry';

    const failed = entry.success === false
        ? ' <span class="log-error">failed</span>' : '';
    const prompt = entry.prompt || '';
    const promptHtml = prompt.length <= 140
        ? `<div>${escapeHtml(prompt)}</div>`
        : `<details><summary>${escapeHtml(prompt.slice(0, 140) + '…')}</summary><div>${escapeHtml(prompt)}</div></details>`;
    const logHtml = (entry.log || []).map(escapeHtml).join('<br>');

    panel.innerHTML = `
        <h6>${escapeHtml(entry.playlist_name || 'playlist')} — ${escapeHtml(entry.track_count)} tracks${failed}</h6>
        <div class="history-meta">${escapeHtml(new Date(entry.timestamp).toLocaleString())} · ${escapeHtml(entry.llm_mode || '')} / ${escapeHtml(entry.llm_model || '')}</div>
        <div><span class="form-label">prompt</span>${promptHtml}</div>
        <details><summary>tracklist (${escapeHtml(entry.track_count)})</summary>
            <ol class="track-list">${renderHistoryTracklist(entry.tracks)}</ol>
        </details>
        <details><summary>decision log</summary>
            <div class="output-container history-log">${logHtml}</div>
        </details>
        <div class="mt-2">
            <span class="form-label">your feedback</span>
            <textarea class="form-control history-feedback" rows="3"></textarea>
            <div class="mt-2 d-flex gap-2 align-items-center">
                <button type="button" class="btn btn-primary btn-sm history-feedback-save">save feedback</button>
                <span class="history-feedback-status"></span>
            </div>
        </div>`;

    const textarea = panel.querySelector('.history-feedback');
    const saveBtn = panel.querySelector('.history-feedback-save');
    const status = panel.querySelector('.history-feedback-status');
    textarea.value = entry.feedback ? entry.feedback.text : '';
    if (entry.feedback) {
        status.textContent = `saved ${new Date(entry.feedback.at).toLocaleString()}`;
    }

    saveBtn.addEventListener('click', async () => {
        const text = textarea.value.trim();
        status.classList.remove('log-error');
        if (!text) {
            status.classList.add('log-error');
            status.textContent = 'feedback cannot be empty';
            return;
        }
        saveBtn.disabled = true;
        status.textContent = 'saving...';
        try {
            const resp = await fetch(`/playlist_history/${encodeURIComponent(entry.id)}/feedback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const result = await resp.json();
            if (resp.ok) {
                status.textContent = 'saved just now';
            } else {
                status.classList.add('log-error');
                status.textContent = result.error || 'failed to save';
            }
        } catch (err) {
            status.classList.add('log-error');
            status.textContent = String(err);
        } finally {
            saveBtn.disabled = false;
        }
    });

    return panel;
}

async function loadPlaylistHistory() {
    const container = document.getElementById('historyList');
    if (!container) return;
    try {
        const resp = await fetch('/playlist_history');
        const entries = await resp.json();
        container.innerHTML = '';
        if (!Array.isArray(entries) || entries.length === 0) {
            container.innerHTML = '<div class="history-meta">no playlist runs yet — generate a mix and it will show up here.</div>';
            return;
        }
        entries.forEach(entry => container.appendChild(renderHistoryEntry(entry)));
    } catch (err) {
        container.innerHTML = `<div class="log-error">${escapeHtml('error loading history: ' + err)}</div>`;
    }
}

// Refetch on every activation so newly completed runs appear.
document.querySelector('a[href="#historyTab"]')?.addEventListener('click', function() {
    setTimeout(loadPlaylistHistory, 100);
});

// Check auth status on page load if we're on the library tab
document.addEventListener('DOMContentLoaded', function() {
    // Initialize LLM configuration on page load
    async function initializeLlmConfig() {
        const llmModeSelect = document.getElementById('llmModeSelect');
        if (llmModeSelect) {
            // Get the initially selected mode from the dropdown
            currentLlmMode = llmModeSelect.value;
            
            // Show the correct API configuration section
            toggleApiConfig(currentLlmMode);
            
            // Fetch current config to get the selected model
            try {
                const response = await fetch('/get_config');
                const config = await response.json();
                
                if (config.llm && config.llm.model) {
                    currentLlmModel = config.llm.model;
                }
                
                // Set thinking toggle state
                const thinkingToggle = document.getElementById('thinkingToggle');
                if (thinkingToggle && config.llm && config.llm.thinking_enabled) {
                    thinkingToggle.checked = config.llm.thinking_enabled.toLowerCase() === 'on';
                }
                
                // Get embedding model from appropriate section
                if (currentLlmMode === 'ollama' && config.ollama && config.ollama.embedding_model) {
                    currentEmbeddingModel = config.ollama.embedding_model;
                } else if (currentLlmMode === 'openai' && config.openai && config.openai.embedding_model) {
                    currentEmbeddingModel = config.openai.embedding_model;
                }
                
                // Fetch and populate models for the current mode
                await fetchModels(currentLlmMode);
                await fetchEmbeddingModels(currentLlmMode);
            } catch (error) {
                console.error('Error initializing LLM config:', error);
                // Still try to fetch models even if config fetch fails
                await fetchModels(currentLlmMode);
                await fetchEmbeddingModels(currentLlmMode);
            }
        }
    }
    
    // Initialize LLM configuration
    initializeLlmConfig();
    
    // Check for auth success/error messages in URL
    const urlParams = new URLSearchParams(window.location.search);
    const spotifyAuth = urlParams.get('spotify_auth');
    if (spotifyAuth === 'success') {
        // Switch to library tab if not already there
        const libraryTabLink = document.querySelector('a[href="#libraryTab"]');
        const libraryTab = document.getElementById('libraryTab');
        if (libraryTabLink && libraryTab) {
            // Activate the library tab
            const tab = new bootstrap.Tab(libraryTabLink);
            tab.show();
        }
        
        setTimeout(checkSpotifyAuth, 500);
        // Clear the URL parameter
        const newUrl = window.location.pathname;
        window.history.replaceState({}, document.title, newUrl);
    } else if (spotifyAuth === 'error') {
        alert('Spotify authentication failed. Please try again.');
        // Clear the URL parameter
        const newUrl = window.location.pathname;
        window.history.replaceState({}, document.title, newUrl);
    }
    
    // Check auth status if we're on the library tab
    const libraryTab = document.getElementById('libraryTab');
    if (libraryTab && libraryTab.classList.contains('show')) {
        (async () => {
            await checkSpotifyAuth();
            await maybeLoadOwnedPlaylists();
        })();
    }
});

if ('serviceWorker' in navigator && window.location.protocol === 'https:') {
  window.addEventListener('load', function() {
    navigator.serviceWorker.register('/static/service-worker.js').then(function(registration) {
      // Registration successful
    }, function(err) {
      // Registration failed
      console.warn('ServiceWorker registration failed: ', err);
    });
  });
} 