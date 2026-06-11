const API_BASE = '/api/v1/documents';
const QUERY_BASE = '/api/v1/query';

// Map job status to UI progress and styling
const STATUS_MAP = {
    'PENDING': { progress: 10, class: 'status-processing' },
    'PARSING': { progress: 30, class: 'status-processing' },
    'CHUNKING': { progress: 60, class: 'status-processing' },
    'EMBEDDING': { progress: 90, class: 'status-processing' },
    'COMPLETED': { progress: 100, class: 'status-completed' },
    'FAILED': { progress: 100, class: 'status-failed' }
};

class Dashboard {
    constructor() {
        this.jobListContainer = document.getElementById('job-list-container');
        this.fileInput = document.getElementById('file-input');
        this.dropZone = document.getElementById('drop-zone');
        this.activeJobs = new Set();
        
        this.init();
    }

    async init() {
        // Setup Event Listeners
        this.dropZone.onclick = () => this.fileInput.click();
        this.fileInput.onchange = (e) => this.handleUpload(e.target.files[0]);
        
        // Query Section elements
        this.askBtn = document.getElementById('ask-btn');
        this.queryInput = document.getElementById('query-input');
        this.askBtn.onclick = () => this.handleQuery();

        // Simple drag-and-drop feedback
        this.dropZone.ondragover = (e) => { 
            e.preventDefault(); 
            this.dropZone.style.boxShadow = '0 0 30px var(--primary-glow)';
        };
        this.dropZone.ondragleave = () => { 
            this.dropZone.style.boxShadow = 'none'; 
        };
        this.dropZone.ondrop = (e) => { 
            e.preventDefault(); 
            this.dropZone.style.boxShadow = 'none';
            this.handleUpload(e.dataTransfer.files[0]); 
        };

        // Load initial data
        await this.refreshJobs();
        
        // Start the status polling loop
        setInterval(() => this.pollIfActive(), 2000);
    }

    async handleQuery() {
        const query = this.queryInput.value.trim();
        if (!query) return;

        const loading = document.getElementById('query-loading');
        const resultArea = document.getElementById('query-result');
        
        loading.classList.remove('hidden');
        resultArea.classList.add('hidden');
        this.askBtn.disabled = true;

        try {
            const res = await fetch(`${QUERY_BASE}/ask`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Query failed');
            }

            const data = await res.json();
            const { situation, task, action, result } = data.answer;

            document.getElementById('res-situation').textContent = situation;
            document.getElementById('res-task').textContent = task;
            document.getElementById('res-action').textContent = action;
            document.getElementById('res-result').textContent = result;

            resultArea.classList.remove('hidden');
        } catch (e) {
            alert(`Query Error: ${e.message}`);
        } finally {
            loading.classList.add('hidden');
            this.askBtn.disabled = false;
        }
    }

    async refreshJobs() {
        try {
            const res = await fetch(`${API_BASE}/jobs`);
            if (!res.ok) throw new Error('API Error');
            const jobs = await res.json();
            
            // Clear current list and re-render (keeps it simple for now)
            this.jobListContainer.innerHTML = '';
            jobs.forEach(job => this.renderJob(job));
        } catch (e) {
            console.error('Failed to load jobs:', e);
        }
    }

    async handleUpload(file) {
        if (!file) return;
        
        const docType = document.querySelector('input[name="doc-type"]:checked').value;
        const formData = new FormData();
        formData.append('file', file);
        formData.append('doc_type', docType);

        try {
            const res = await fetch(`${API_BASE}/upload`, { 
                method: 'POST', 
                body: formData 
            });
            
            if (!res.ok) throw new Error('Upload failed');
            
            // Refresh list to show the new "PENDING" job immediately
            await this.refreshJobs();
        } catch (e) {
            alert('Upload failed. Please try again.');
            console.error(e);
        }
    }

    renderJob(job) {
        const config = STATUS_MAP[job.status] || STATUS_MAP['PENDING'];
        
        const jobCard = document.createElement('div');
        jobCard.className = 'job-item';
        jobCard.innerHTML = `
            <div class="job-header">
                <div class="job-info">
                    <span class="filename">${job.filename}</span>
                    <span class="status-badge ${config.class}">${job.status}</span>
                </div>
                <button class="delete-btn" onclick="dashboard.handleDelete(${job.document_id}, event)" title="Delete Document">
                    <svg viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd" />
                    </svg>
                </button>
            </div>
            <div class="progress-container">
                <div class="progress-fill" style="width: ${config.progress}%"></div>
            </div>
            ${job.error ? `<p class="error-msg">⚠️ ${job.error}</p>` : ''}
        `;

        this.jobListContainer.appendChild(jobCard);

        // Track if we need to continue polling for this job
        if (!['COMPLETED', 'FAILED'].includes(job.status)) {
            this.activeJobs.add(job.job_id);
        } else {
            this.activeJobs.delete(job.job_id);
        }
    }

    async handleDelete(docId, event) {
        if (!docId) return;
        
        // Visual confirmation
        if (!confirm('Are you sure you want to delete this document and all its indexed data? This cannot be undone.')) {
            return;
        }

        const btn = event.currentTarget;
        const card = btn.closest('.job-item');
        
        try {
            btn.disabled = true;
            card.style.opacity = '0.5';
            
            const res = await fetch(`${API_BASE}/${docId}`, { method: 'DELETE' });
            if (!res.ok) throw new Error('Delete failed');
            
            // Success animation
            card.style.transform = 'translateX(50px)';
            card.style.opacity = '0';
            setTimeout(() => card.remove(), 300);
            
            // Clean up active jobs if necessary
            await this.refreshJobs();
        } catch (e) {
            btn.disabled = false;
            card.style.opacity = '1';
            alert(`Error: ${e.message}`);
        }
    }

    async pollIfActive() {
        // If there are any jobs in non-terminal states, refresh the whole list
        if (this.activeJobs.size > 0) {
            await this.refreshJobs();
        }
    }
}

// Initialize the dashboard on load and make it global for the onclick handlers
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
});
