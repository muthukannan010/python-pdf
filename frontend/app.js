/*
 * app.js - Frontend for PDF Search Assistant
 * Handles file uploads, search, document management, and the chat UI.
 */

'use strict';

const API = '/api/v1';

// grab DOM elements we need
const chatHistory = document.getElementById('chat-history');
const chatEmptyState = document.getElementById('chat-empty-state');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const typingIndicator = document.getElementById('typing-indicator');

let isSearching = false;

// --- Toast notifications ---
function showToast(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast toast--${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(12px)';
    toast.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}

function scrollToBottom() {
  setTimeout(() => {
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }, 50);
}

// --- Health check ---
async function checkHealth() {
  const badge = document.getElementById('health-badge');
  try {
    const res = await fetch(`${API}/health`);
    const data = await res.json();
    if (data.status === 'ok') {
      badge.className = 'badge badge--green';
      badge.innerHTML = `<span class="badge-dot"></span> Online`;
    } else {
      badge.className = 'badge badge--red';
      badge.innerHTML = `<span class="badge-dot"></span> Degraded`;
    }
  } catch {
    badge.className = 'badge badge--red';
    badge.innerHTML = `<span class="badge-dot"></span> Offline`;
  }
}

// --- Settings panel ---
function setupSettings() {
  const btn = document.getElementById('settings-btn');
  const dropdown = document.getElementById('settings-dropdown');
  const kwSlider = document.getElementById('kw-weight-input');
  const kwVal = document.getElementById('kw-weight-val');

  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    dropdown.classList.toggle('hidden');
  });

  // close when clicking outside
  document.addEventListener('click', (e) => {
    if (!dropdown.contains(e.target) && !btn.contains(e.target)) {
      dropdown.classList.add('hidden');
    }
  });

  kwSlider.addEventListener('input', () => {
    kwVal.textContent = kwSlider.value;
  });
}

// --- File upload ---
function setupFileInput() {
  const fileInput = document.getElementById('file-input');
  const chatTrigger = document.getElementById('chat-upload-trigger');
  const sidebarTrigger = document.getElementById('sidebar-upload-trigger');
  const emptyTrigger = document.getElementById('empty-upload-trigger');

  const openPicker = () => fileInput.click();

  if (chatTrigger) chatTrigger.addEventListener('click', openPicker);
  if (sidebarTrigger) sidebarTrigger.addEventListener('click', openPicker);
  if (emptyTrigger) emptyTrigger.addEventListener('click', openPicker);

  fileInput.addEventListener('change', async () => {
    if (fileInput.files[0]) {
      await uploadDocument(fileInput.files[0]);
      fileInput.value = '';
    }
  });
}

async function uploadDocument(file) {
  if (!file) return;

  if (!file.name.toLowerCase().endsWith('.pdf') && file.type !== 'application/pdf') {
    showToast('Only PDF files are supported.', 'error');
    return;
  }
  if (file.size > 50 * 1024 * 1024) {
    showToast('File exceeds the 50 MB limit.', 'error');
    return;
  }

  const indicator = document.getElementById('upload-progress-indicator');
  indicator.classList.remove('hidden');
  chatEmptyState.classList.add('hidden');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API}/documents/upload`, {
      method: 'POST',
      body: formData,
    });

    indicator.classList.add('hidden');

    if (res.ok) {
      const data = await res.json();
      showToast(`'${data.document_name}' uploaded successfully!`, 'success');
      appendSystemMessage(`✓ **${data.document_name}** has been processed successfully. You can now ask questions about this document.`);
      await loadDocuments();
      await checkHealth();
    } else if (res.status === 409) {
      const data = await res.json();
      showToast('Duplicate document.', 'error');
      appendSystemMessage(`⚠ ${data.detail}`);
    } else {
      const data = await res.json().catch(() => ({ detail: 'Upload failed.' }));
      showToast('Upload failed.', 'error');
      appendSystemMessage(`✗ ${data.detail || 'Upload failed.'}`);
    }
  } catch (err) {
    indicator.classList.add('hidden');
    showToast('Network error.', 'error');
    appendSystemMessage('✗ Network error during upload.');
  }
}

// --- Document list ---
async function loadDocuments() {
  const list = document.getElementById('doc-list');
  const empty = document.getElementById('doc-empty');
  const countEl = document.getElementById('doc-count');

  try {
    const res = await fetch(`${API}/documents`);
    const data = await res.json();
    const docs = data.documents || [];

    countEl.textContent = docs.length;

    if (docs.length === 0) {
      empty.classList.remove('hidden');
      list.querySelectorAll('.doc-item').forEach(el => el.remove());
      return;
    }

    empty.classList.add('hidden');
    list.querySelectorAll('.doc-item').forEach(el => el.remove());

    docs.forEach(doc => {
      const item = document.createElement('div');
      item.className = 'doc-item';
      item.innerHTML = `
        <div class="doc-item__icon">📄</div>
        <div class="doc-item__info">
          <span class="doc-item__name" title="${escapeHtml(doc.document_name)}">${escapeHtml(doc.document_name)}</span>
          <span class="doc-item__meta">${doc.page_count} pages</span>
        </div>
        <button class="btn btn--danger btn--sm" title="Delete"
                data-delete-id="${escapeHtml(doc.document_id)}">✕</button>
      `;
      list.appendChild(item);
    });
  } catch {
    showToast('Could not load document list.', 'error');
  }
}

async function deleteDocument(docId) {
  if (!confirm(`Delete document '${docId}'?`)) return;
  try {
    const res = await fetch(`${API}/documents/${encodeURIComponent(docId)}`, { method: 'DELETE' });
    if (res.ok) {
      showToast(`'${docId}' deleted.`, 'success');
      await loadDocuments();
      await checkHealth();
    } else {
      const data = await res.json().catch(() => ({ detail: 'Delete failed.' }));
      showToast(data.detail || 'Delete failed.', 'error');
    }
  } catch {
    showToast('Network error.', 'error');
  }
}

document.getElementById('doc-list').addEventListener('click', (e) => {
  const btn = e.target.closest('[data-delete-id]');
  if (btn) deleteDocument(btn.dataset.deleteId);
});

// --- Chat + Search ---
function setupChatControls() {
  // auto-resize the textarea as user types
  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 150) + 'px';
    sendBtn.disabled = chatInput.value.trim().length === 0;
  });

  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (chatInput.value.trim().length > 0 && !isSearching) {
        performSearch();
      }
    }
  });

  sendBtn.addEventListener('click', (e) => {
    e.preventDefault();
    if (chatInput.value.trim().length > 0 && !isSearching) {
      performSearch();
    }
  });

  // suggested query buttons
  document.querySelectorAll('.suggested-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      chatInput.value = btn.textContent.replace(/"/g, '');
      sendBtn.disabled = false;
      performSearch();
    });
  });
}

function appendUserMessage(text) {
  chatEmptyState.classList.add('hidden');
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble-wrapper user';
  bubble.innerHTML = `<div class="chat-bubble user">${escapeHtml(text)}</div>`;
  chatHistory.insertBefore(bubble, typingIndicator);
  scrollToBottom();
}

function appendSystemMessage(text) {
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble-wrapper assistant';
  let safe = escapeHtml(text).replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  bubble.innerHTML = `
    <div class="chat-bubble assistant" style="background: var(--bg-hover); color: var(--text-secondary); font-size: 0.85rem; border: none; max-width: 90%;">
      ${safe}
    </div>
  `;
  chatHistory.insertBefore(bubble, typingIndicator);
  scrollToBottom();
}

function appendAssistantMessage(results) {
  if (!results || results.length === 0) {
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble-wrapper assistant';
    bubble.innerHTML = `
      <div class="chat-bubble assistant">
        I couldn't find relevant information in the uploaded documents.<br><br>
        <span style="color: var(--text-muted); font-size: 0.9em;">Try using different keywords or asking the question in another way.</span>
      </div>
    `;
    chatHistory.insertBefore(bubble, typingIndicator);
    scrollToBottom();
    return;
  }

  results.forEach(r => {
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble-wrapper assistant';

    // preserve <mark> tags from highlighting while escaping everything else
    let rawText = r.text;
    let safeText = rawText.replace(/<mark>/gi, '[[MARK]]').replace(/<\/mark>/gi, '[[/MARK]]');
    safeText = escapeHtml(safeText);
    safeText = safeText.replace(/\[\[MARK\]\]/g, '<mark>').replace(/\[\[\/MARK\]\]/g, '</mark>');

    const docId = r.source_path.replace('data/pdfs/', '').replace('.pdf', '');

    bubble.innerHTML = `
      <div class="chat-bubble assistant">
        <div class="source-ref">
          <div class="source-ref-header" style="flex-direction: column; align-items: flex-start; gap: 4px; font-size: 0.85em; color: var(--text-muted); margin-bottom: 0.5rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5rem;">
            <div><span style="font-weight: 600;">${escapeHtml(r.document_name)}</span> (Page ${r.page_number})</div>
            <div>Keyword Match: <span style="color: var(--text-primary); font-weight: 500;">${r.keyword_score}%</span></div>
            <div>Semantic Match: <span style="color: var(--text-primary); font-weight: 500;">${r.semantic_score}%</span></div>
            <div>Final Relevance: <span style="color: var(--text-primary); font-weight: 500;">${r.score}%</span></div>
          </div>
        </div>
        ${safeText}
        <button class="btn btn--ghost btn--sm" style="display: block; margin-top: 0.5rem;" onclick="openPdfViewer('${encodeURIComponent(docId)}', ${r.page_number}, '${escapeHtml(r.document_name)}')">
          [ View Page ${r.page_number} ]
        </button>
      </div>
    `;
    chatHistory.insertBefore(bubble, typingIndicator);
  });

  scrollToBottom();
}

async function performSearch() {
  if (isSearching) return;
  const query = chatInput.value.trim();
  if (!query) return;

  const topK = parseInt(document.getElementById('top-k-input').value, 10);
  const kwWeight = parseFloat(document.getElementById('kw-weight-input').value);

  appendUserMessage(query);
  chatInput.value = '';
  chatInput.style.height = 'auto';
  sendBtn.disabled = true;
  isSearching = true;
  typingIndicator.classList.remove('hidden');
  scrollToBottom();

  try {
    const res = await fetch(`${API}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: topK, keyword_weight: kwWeight }),
    });

    typingIndicator.classList.add('hidden');
    isSearching = false;

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Search failed.' }));
      appendSystemMessage(`⚠ Error: ${err.detail || 'Search failed.'}`);
      return;
    }

    const data = await res.json();
    appendAssistantMessage(data.results);

  } catch {
    typingIndicator.classList.add('hidden');
    isSearching = false;
    appendSystemMessage('⚠ Network error during search. Is the server running?');
  }
}

// --- PDF Viewer Modal ---
function openPdfViewer(encodedDocId, pageNumber, docName) {
  const modal = document.getElementById('pdf-modal');
  const iframe = document.getElementById('pdf-iframe');
  const title = document.getElementById('modal-title');

  title.textContent = `${docName} — Page ${pageNumber}`;
  iframe.src = `${API}/documents/${encodedDocId}/view#page=${pageNumber}`;
  modal.classList.remove('hidden');
  document.getElementById('modal-close-btn').focus();
}

function closePdfViewer() {
  document.getElementById('pdf-modal').classList.add('hidden');
  document.getElementById('pdf-iframe').src = 'about:blank';
}

document.getElementById('modal-close-btn').addEventListener('click', closePdfViewer);
document.getElementById('pdf-modal').addEventListener('click', (e) => {
  if (e.target === document.getElementById('pdf-modal')) closePdfViewer();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closePdfViewer();
});

// make available for inline onclick handlers
window.openPdfViewer = openPdfViewer;

// --- Init ---
document.addEventListener('DOMContentLoaded', async () => {
  setupFileInput();
  setupSettings();
  setupChatControls();
  await checkHealth();
  await loadDocuments();
});
