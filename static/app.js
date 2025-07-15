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

// Helper function to handle script execution and output streaming
function handleScriptExecution(formId, outputId, endpoint) {
    document.getElementById(formId).addEventListener('submit', async (e) => {
        e.preventDefault();
        const form = e.target;
        const outputDiv = document.getElementById(outputId);
        outputDiv.innerHTML = '';
        // Show output only when script is run
        outputDiv.classList.remove('d-none');
        
        const formData = new FormData(form);
        const data = {};
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }
        
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

                    // Detect progress bar lines (tqdm or similar)
                    const isProgressBar = (line.includes('|') && /\d+%\|/.test(line)) || line.includes('song/s') || line.includes('Building playlist:');

                    if (isProgressBar) {
                        // Overwrite the last line
                        const lines = outputDiv.innerHTML.split('<br>');
                        if (lines.length > 1) {
                            lines[lines.length - 1] = line;
                            outputDiv.innerHTML = lines.join('<br>');
                        } else {
                            outputDiv.innerHTML = line + '<br>';
                        }
                    } else {
                        // Normal output: append as new line
                        outputDiv.innerHTML += line + '<br>';
                    }
                    outputDiv.scrollTop = outputDiv.scrollHeight;
                };
                
                eventSource.onerror = () => {
                    eventSource.close();
                };
            }
        } catch (error) {
            outputDiv.classList.remove('d-none');
            outputDiv.innerHTML += 'Error: ' + error + '<br>';
        }
    });
}

// Set up form handlers
handleScriptExecution('djForm', 'djOutput', '/run_dj');
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