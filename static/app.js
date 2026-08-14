/*
 * LifeSearch UI controller (Phase B).
 * Vanilla JS, no external dependencies. Talks only to the local server.
 */
(function () {
  "use strict";

  var form = document.getElementById("search-form");
  var input = document.getElementById("search-input");
  var button = document.getElementById("search-button");
  var statusLine = document.getElementById("status-line");
  var resultsEl = document.getElementById("results");

  var emptyState = document.getElementById("empty-state");
  var noResultsState = document.getElementById("no-results-state");
  var noResultsText = document.getElementById("no-results-text");
  var errorState = document.getElementById("error-state");
  var errorText = document.getElementById("error-text");
  var loadingState = document.getElementById("loading-state");

  var modal = document.getElementById("doc-modal");
  var modalTitle = document.getElementById("doc-title");
  var modalBody = document.getElementById("doc-body");

  var lastQuery = "";

  function showState(name) {
    emptyState.hidden = true;
    noResultsState.hidden = true;
    errorState.hidden = true;
    loadingState.hidden = true;
    resultsEl.hidden = true;
    if (name === "empty") emptyState.hidden = false;
    else if (name === "no-results") noResultsState.hidden = false;
    else if (name === "error") errorState.hidden = false;
    else if (name === "loading") loadingState.hidden = false;
    else if (name === "results") resultsEl.hidden = false;
  }

  function escapeHtml(value) {
    if (value === null || value === undefined) return "";
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatScore(score) {
    var s = typeof score === "number" ? score : 0;
    return Math.round(s * 100) / 100;
  }

  function formatBytes(bytes) {
    var b = typeof bytes === "number" ? bytes : 0;
    if (b < 1024) return b + " B";
    if (b < 1024 * 1024) return (b / 1024).toFixed(1) + " KB";
    return (b / (1024 * 1024)).toFixed(1) + " MB";
  }

  function renderEvidence(items) {
    if (!Array.isArray(items) || items.length === 0) return "";
    var chips = items.map(function (ev) {
      var type = ev.type === "episode" ? "Episode" : ev.type === "memory" ? "Memory" : "Linked";
      var cls = ev.type === "episode" ? "evidence-episode" : ev.type === "memory" ? "evidence-memory" : "evidence-other";
      return (
        '<li class="evidence-chip ' + cls + '">' +
        '<span class="evidence-type">' + escapeHtml(type) + "</span>" +
        '<span class="evidence-title">' + escapeHtml(ev.title || "") + "</span></li>"
      );
    });
    return '<ul class="evidence-list">' + chips.join("") + "</ul>";
  }

  function renderEpisodes(episodes) {
    if (!Array.isArray(episodes) || episodes.length === 0) return "";
    var items = episodes.map(function (ep) {
      var range = [ep.start_ts, ep.end_ts].filter(Boolean).join(" – ");
      return (
        '<li class="meta-item"><span class="meta-title">' + escapeHtml(ep.title || "") + "</span>" +
        (range ? '<span class="meta-sub">' + escapeHtml(range) + "</span>" : "") + "</li>"
      );
    });
    return '<div class="result-block"><h4 class="block-title">Episodes</h4><ul class="meta-list">' + items.join("") + "</ul></div>";
  }

  function renderMemories(memories) {
    if (!Array.isArray(memories) || memories.length === 0) return "";
    var items = memories.map(function (m) {
      var topics = Array.isArray(m.topics) ? m.topics.join(", ") : "";
      return (
        '<li class="meta-item"><span class="meta-title">' + escapeHtml(m.title || "") + "</span>" +
        (topics ? '<span class="meta-sub">' + escapeHtml(topics) + "</span>" : "") + "</li>"
      );
    });
    return '<div class="result-block"><h4 class="block-title">Memories</h4><ul class="meta-list">' + items.join("") + "</ul></div>";
  }

  function renderCard(result) {
    var docId = escapeHtml(result.document_id || "");
    var fileName = escapeHtml(result.file_name || "Untitled");
    var mime = escapeHtml(result.mime_type || "");
    var snippet = escapeHtml(result.snippet || "");
    var score = formatScore(result.score);
    var why = escapeHtml(result.why || "");
    var path = escapeHtml(result.path || "");

    var mimeBadge = mime ? '<span class="badge mime-badge">' + mime + "</span>" : "";
    var scoreBadge = '<span class="badge score-badge" title="Relevance score">' + score + "</span>";

    var evidenceHtml = renderEvidence(result.evidence);
    var episodesHtml = renderEpisodes(result.episodes);
    var memoriesHtml = renderMemories(result.memories);

    var detailBtn = docId
      ? '<button type="button" class="detail-button" data-doc-id="' + docId + '">View details</button>'
      : "";
    var pinBtn = docId
      ? '<button type="button" class="feedback-button pin-button" data-doc-id="' + docId + '" data-action="pin">Pin</button>'
      : "";
    var ignoreBtn = docId
      ? '<button type="button" class="feedback-button ignore-button" data-doc-id="' + docId + '" data-action="ignore">Not relevant</button>'
      : "";

    var actions = (detailBtn || pinBtn || ignoreBtn)
      ? '<div class="card-actions">' + detailBtn + pinBtn + ignoreBtn + "</div>"
      : "";

    return (
      '<article class="result-card" data-doc-id="' + docId + '">' +
        '<header class="card-header">' +
          '<h3 class="card-title">' + fileName + "</h3>" +
          '<div class="card-badges">' + mimeBadge + scoreBadge + "</div>" +
        "</header>" +
        (snippet ? '<p class="card-snippet">' + snippet + "</p>" : "") +
        (why ? '<p class="card-why"><span class="why-label">Why:</span> ' + why + "</p>" : "") +
        (evidenceHtml ? '<div class="result-block"><h4 class="block-title">Evidence</h4>' + evidenceHtml + "</div>" : "") +
        episodesHtml +
        memoriesHtml +
        '<details class="card-path"><summary>File location</summary>' +
          '<p class="path-text">' + path + "</p>" +
        "</details>" +
        actions +
      "</article>"
    );
  }

  function renderResults(data) {
    var results = Array.isArray(data.results) ? data.results : [];
    var took = typeof data.took_ms === "number" ? data.took_ms : null;

    if (results.length === 0) {
      noResultsText.textContent =
        "No results found" + (lastQuery ? ' for “' + lastQuery + "”" : "") + ".";
      showState("no-results");
      statusLine.textContent = "";
      return;
    }

    resultsEl.innerHTML = results.map(renderCard).join("");
    showState("results");
    var status = "Found " + results.length + " result" + (results.length === 1 ? "" : "s");
    if (took !== null) status += " in " + took + " ms";
    statusLine.textContent = status;
  }

  function runSearch(query) {
    lastQuery = query;
    showState("loading");
    button.disabled = true;
    fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query, k: 10 }),
    })
      .then(function (resp) {
        return resp.json().then(function (data) {
          return { ok: resp.ok, data: data };
        }, function () {
          return { ok: resp.ok, data: null };
        });
      })
      .then(function (out) {
        if (!out.ok) {
          var msg = out.data && out.data.error && out.data.error.message
            ? out.data.error.message
            : "Search failed";
          var detail = out.data && out.data.error && out.data.error.detail
            ? out.data.error.detail
            : "";
          errorText.textContent = detail ? msg + ": " + detail : msg;
          showState("error");
          statusLine.textContent = "";
          return;
        }
        renderResults(out.data || { results: [] });
      })
      .catch(function () {
        errorText.textContent =
          "We couldn’t reach the search service. Please make sure LifeSearch is running.";
        showState("error");
        statusLine.textContent = "";
      })
      .then(function () {
        button.disabled = false;
      });
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var q = input.value.trim();
    if (!q) {
      input.focus();
      return;
    }
    runSearch(q);
  });

  // Event delegation for "View details" and feedback buttons.
  resultsEl.addEventListener("click", function (e) {
    var fb = e.target.closest(".feedback-button");
    if (fb) {
      var fbId = fb.getAttribute("data-doc-id");
      var action = fb.getAttribute("data-action");
      if (fbId && action) sendFeedback(fbId, action, fb);
      return;
    }
    var btn = e.target.closest(".detail-button");
    if (!btn) return;
    var id = btn.getAttribute("data-doc-id");
    if (id) openDocument(id, btn);
  });

  var feedbackStatus = document.getElementById("feedback-status");
  var feedbackStatusTimer = null;

  function showFeedbackConfirmation(message) {
    if (!feedbackStatus) return;
    feedbackStatus.textContent = message;
    feedbackStatus.hidden = false;
    if (feedbackStatusTimer) clearTimeout(feedbackStatusTimer);
    feedbackStatusTimer = setTimeout(function () {
      feedbackStatus.hidden = true;
    }, 1500);
  }

  function sendFeedback(docId, action, btn) {
    if (!docId) return;
    var query = lastQuery || "";
    fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query, document_id: String(docId), action: action }),
    })
      .then(function (resp) {
        return resp.json().then(function (d) {
          return { ok: resp.ok, data: d };
        }, function () {
          return { ok: resp.ok, data: null };
        });
      })
      .then(function (out) {
        if (out.ok && out.data && out.data.ok) {
          if (btn) {
            if (action === "pin") {
              btn.textContent = "Pinned";
              btn.disabled = true;
            } else if (action === "ignore") {
              btn.textContent = "Noted";
              btn.disabled = true;
            }
          }
          showFeedbackConfirmation(
            action === "pin" ? "Pinned — thanks for the signal." : "Thanks for the feedback."
          );
        }
        // On failure, do nothing visible; feedback must never break search/open.
      })
      .catch(function () {
        // Network errors are ignored; the core UX remains fully functional.
      });
  }

  function openDocument(id, btn) {
    btn.disabled = true;
    fetch("/document/" + encodeURIComponent(id))
      .then(function (resp) {
        return resp.json().then(function (data) {
          return { ok: resp.ok, data: data };
        }, function () {
          return { ok: resp.ok, data: null };
        });
      })
      .then(function (out) {
        if (!out.ok || !out.data || !out.data.document) {
          showDocError("Could not load document details.");
          return;
        }
        renderDocument(out.data.document);
        openModal();
        // Opening a result is itself a positive signal.
        sendFeedback(id, "click", null);
      })
      .catch(function () {
        showDocError("Could not load document details.");
      })
      .then(function () {
        btn.disabled = false;
      });
  }

  function renderDocument(doc) {
    modalTitle.textContent = doc.file_name || "Document";
    var rows = [
      ["File name", doc.file_name],
      ["Type", doc.mime_type],
      ["Size", formatBytes(doc.size)],
      ["Modified", doc.modified_at],
      ["Indexed", doc.indexed_at],
      ["Location", doc.path],
    ];
    var html = '<dl class="doc-meta">';
    rows.forEach(function (r) {
      if (r[1] === null || r[1] === undefined || r[1] === "") return;
      html += "<dt>" + escapeHtml(r[0]) + "</dt><dd>" + escapeHtml(r[1]) + "</dd>";
    });
    html += "</dl>";
    var text = doc.extracted_text || "";
    html += '<h3 class="doc-section-title">Extracted text</h3>';
    html += '<pre class="doc-text">' + escapeHtml(text) + "</pre>";
    modalBody.innerHTML = html;
  }

  function showDocError(msg) {
    modalTitle.textContent = "Unable to load";
    modalBody.textContent = msg;
    openModal();
  }

  function openModal() {
    modal.hidden = false;
    var closeBtn = modal.querySelector(".modal-close");
    if (closeBtn) closeBtn.focus();
  }

  function closeModal() {
    modal.hidden = true;
    modalBody.innerHTML = "";
  }

  modal.addEventListener("click", function (e) {
    if (e.target && e.target.getAttribute && e.target.getAttribute("data-close") === "true") {
      closeModal();
    }
  });

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !modal.hidden) closeModal();
  });

  // Initial state.
  showState("empty");

  // -------------------------------------------------------------------------
  // Indexing panel (Phase C): on-demand folder indexing + status/progress.
  // -------------------------------------------------------------------------
  var indexForm = document.getElementById("index-form");
  var indexPath = document.getElementById("index-path");
  var indexReindex = document.getElementById("index-reindex");
  var indexButton = document.getElementById("index-button");
  var indexStatus = document.getElementById("index-status");
  var indexStateLabel = document.getElementById("index-state-label");
  var indexLastIndexed = document.getElementById("index-last-indexed");
  var indexProgress = document.getElementById("index-progress");
  var indexProgressBar = document.getElementById("index-progress-bar");
  var indexProgressText = document.getElementById("index-progress-text");
  var indexProcessed = document.getElementById("index-processed");
  var indexSkipped = document.getElementById("index-skipped");
  var indexErrors = document.getElementById("index-errors");
  var indexCurrentFolder = document.getElementById("index-current-folder");
  var indexError = document.getElementById("index-error");

  var indexPollTimer = null;

  function showIndexError(msg) {
    indexError.textContent = msg;
    indexError.hidden = false;
  }

  function clearIndexError() {
    indexError.textContent = "";
    indexError.hidden = true;
  }

  function formatDateTime(ms) {
    if (ms === null || ms === undefined || ms === 0) return "";
    try {
      var d = new Date(ms);
      if (isNaN(d.getTime())) return "";
      return d.toLocaleString();
    } catch (e) {
      return "";
    }
  }

  function startIndexPolling() {
    if (indexPollTimer) clearInterval(indexPollTimer);
    indexPollTimer = setInterval(fetchIndexStatus, 1000);
    fetchIndexStatus();
  }

  function stopIndexPolling() {
    if (indexPollTimer) {
      clearInterval(indexPollTimer);
      indexPollTimer = null;
    }
  }

  function renderIndexStatus(s) {
    if (!s) return;
    var inProgress = !!s.indexing_in_progress;
    if (inProgress) {
      indexStateLabel.textContent = "Indexing…";
    } else if (s.last_error) {
      indexStateLabel.textContent = "Finished with errors";
    } else {
      indexStateLabel.textContent = "Idle";
    }

    var pct = typeof s.progress_percent === "number" ? s.progress_percent : 0;
    indexProgressBar.style.width = pct + "%";
    if (indexProgress) indexProgress.setAttribute("aria-valuenow", String(pct));
    indexProgressText.textContent = "Progress: " + pct + "%";

    indexProcessed.textContent = s.processed != null ? s.processed : 0;
    indexSkipped.textContent = s.skipped != null ? s.skipped : 0;
    indexErrors.textContent = s.errors != null ? s.errors : 0;

    indexCurrentFolder.textContent = s.current_folder
      ? "Current folder: " + s.current_folder
      : "";

    indexLastIndexed.textContent = s.last_indexed_ms
      ? "Last indexed: " + formatDateTime(s.last_indexed_ms)
      : "";

    if (s.last_error) {
      showIndexError("Last indexing failed: " + s.last_error);
    } else if (!inProgress) {
      clearIndexError();
    }
  }

  function fetchIndexStatus() {
    fetch("/index/status")
      .then(function (resp) {
        return resp.json().then(function (d) { return d; }, function () { return null; });
      })
      .then(function (status) {
        if (!status) return;
        renderIndexStatus(status);
        if (!status.indexing_in_progress) {
          stopIndexPolling();
          indexButton.disabled = false;
        }
      })
      .catch(function () {
        // Transient; the next poll will retry. Do not spam errors.
      });
  }

  function runIndex() {
    var raw = indexPath.value.trim();
    if (!raw) {
      showIndexError("Enter at least one absolute folder path to index.");
      return;
    }
    // Accept multiple paths: one per line, or comma/semicolon separated.
    var paths = raw.split(/[\n,;]+/).map(function (p) { return p.trim(); }).filter(Boolean);
    if (paths.length === 0) {
      showIndexError("Enter at least one absolute folder path to index.");
      return;
    }
    var reindex = !!indexReindex.checked;

    indexButton.disabled = true;
    clearIndexError();

    fetch("/index", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paths: paths, reindex: reindex }),
    })
      .then(function (resp) {
        return resp.json().then(function (data) {
          return { ok: resp.ok, status: resp.status, data: data };
        }, function () {
          return { ok: resp.ok, status: resp.status, data: null };
        });
      })
      .then(function (out) {
        if (out.ok && out.status === 202) {
          // Scheduled. Show progress and begin polling status.
          indexStateLabel.textContent = "Indexing…";
          startIndexPolling();
          return;
        }
        var msg =
          out.data && out.data.error && out.data.error.message
            ? out.data.error.message
            : "Could not start indexing.";
        var detail =
          out.data && out.data.error && out.data.error.detail
            ? out.data.error.detail
            : "";
        showIndexError(detail ? msg + ": " + detail : msg);
        indexButton.disabled = false;
      })
      .catch(function () {
        showIndexError(
          "We couldn’t reach the indexing service. Please make sure LifeSearch is running."
        );
        indexButton.disabled = false;
      });
  }

  if (indexForm) {
    indexForm.addEventListener("submit", function (e) {
      e.preventDefault();
      runIndex();
    });
  }

  // Reflect any already-running/in-progress job on load.
  if (indexStatus) {
    fetch("/index/status")
      .then(function (resp) {
        return resp.json().then(function (d) { return d; }, function () { return null; });
      })
      .then(function (s) {
        if (!s) return;
        renderIndexStatus(s);
        if (s.indexing_in_progress) startIndexPolling();
      })
      .catch(function () {});
  }
})();
