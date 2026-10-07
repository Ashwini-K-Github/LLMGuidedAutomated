// Cocoon Data Cleaning Framework - Frontend Controller

const STATE = {
    currentStep: 1,
    filename: "",
    filesize: "0 KB",
    profile: null,
    anomalies: null,
    recommendations: null,
    history: null,
    cleanedPreview: null
};

// API Endpoints Base
const API_BASE = '/api';

// Sample Inconsistent Datasets for Quick Demo Sandbox
const SAMPLE_DATASETS = {
    students: `StudentID,Name,Email,Phone,Age,GPA,EnrollmentDate,Major
101,John Doe,john.doe@gmail.com,9876543210,20,3.8,2023-09-01,CS
102,Alice Smith,alice@gmail,123-456,21,3.9,09/02/2023,cs
103,Bob Jones,bob@yahoo.com,9998887776,NaN,3.5,03-09-2023,Computer Science
104,Charlie Brown,,98765 43210,22,6.5,2023-09-04,CS
105,Diana Prince,diana@gmail.com,,999,NaN,2023-09-05,CSE
106,Ethan Hunt,ethan@corp.com,8887776665,21,3.7,2023-09-06,unknown
107,Fiona Gallagher,fiona@gmail.com,1234567,20,3.6,NaN,CS
101,John Doe,john.doe@gmail.com,9876543210,20,3.8,2023-09-01,CS`,

    employee: `Employee_ID,Full_Name,Work_Email,Contact_No,Salary,Joining_Date,Department,Is_Active
EMP001,John Connor,john@company.com,9876543210,85000,2022-01-15,Sales,Yes
EMP002,Sarah Connor,sarah@company.com,98765 43211,92000,16-01-2022,sales,Yes
EMP003,Ellen Ripley,ellen@company.com,9876543212,78000,2022/01/17,HR,No
EMP004,James Bond,,9876543213,NaN,2022-01-18,hr,Yes
EMP005,Indiana Jones,indy@company.com,,5000000,2022-01-19,Archaeology,Yes
EMP003,Ellen Ripley,ellen@company.com,9876543212,78000,2022/01/17,HR,No
EMP006,Peter Parker,peter@gmail,12345,60000,NaN,Sales,yes
EMP007,Bruce Wayne,bruce@wayne.com,9876543214,-5000,2022-01-21,Executive,n/a`,

    ecommerce: `Transaction_ID,Customer_Email,Product_Category,Price,Quantity,Order_Date,Payment_Status
TXN1001,alice@gmail.com,Electronics,299.99,1,2024-03-01,True
TXN1002,bob@yahoo.com,clothing,19.99,2,02/03/2024,True
TXN1003,charlie@gmail,Electronics,NaN,1,2024-03-03,False
TXN1004,diana@yahoo.com,Clothing,49.99,999,2024-03-04,true
TXN1005,,Home & Kitchen,15.50,3,2024-03-05,false
TXN1001,alice@gmail.com,Electronics,299.99,1,2024-03-01,True
TXN1006,ethan@gmail.com,electronics,1200.00,1,06-03-2024,Yes`,

    healthcare: `Patient_ID,Patient_Name,Blood_Group,Admit_Date,Emergency_Contact,Age,Discharge_Status
PAT001,Tony Stark,A Positive,2025-05-01,9876543210,48,True
PAT002,Steve Rogers,a+,02/05/2025,98765 43211,105,True
PAT003,Bruce Banner,B Neg,2025-05-03,9876543212,45,False
PAT004,Natasha Romanoff,b-,04-05-2025,,32,true
PAT005,Thor Odinson,AB Pos,,9876543214,1500,n/a
PAT006,Clint Barton,o+,2025-05-06,12345,NaN,false
PAT001,Tony Stark,A Positive,2025-05-01,9876543210,48,True`,

    banking: `Account_Number,Customer_Email,Account_Type,Balance,Open_Date,KYC_Status
ACC7001,clark@dailyplanet.com,Savings,1500.50,2023-11-01,True
ACC7002,lois@dailyplanet.com,Checking,25000.00,02/11/2023,True
ACC7003,lex@lexcorp.com,Savings,-500000.00,03-11-2023,False
ACC7004,bruce@wayne.com,savings,999999999.99,2023-11-04,true
ACC7005,,Checking,100.00,,n/a
ACC7001,clark@dailyplanet.com,Savings,1500.50,2023-11-01,True`
};

// Initialize Application UI
document.addEventListener("DOMContentLoaded", () => {
    initThemeToggle();
    initSettingsPanel();
    initUploadZone();
    initQuickDemo();
    initNavigationControls();
    checkAPIProviders();
    initCollapsibleInspector();
    initPhase2ML(); // Hook Phase 2 initializations
});

// Initialize Dark/Light Theme Toggle
function initThemeToggle() {
    const themeCheckbox = document.getElementById("theme-toggle-checkbox");
    if (!themeCheckbox) return;
    
    // Check local storage for preference
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme === "light") {
        document.body.classList.add("light-theme");
        themeCheckbox.checked = true;
    }
    
    themeCheckbox.addEventListener("change", () => {
        if (themeCheckbox.checked) {
            document.body.classList.add("light-theme");
            localStorage.setItem("theme", "light");
        } else {
            document.body.classList.remove("light-theme");
            localStorage.setItem("theme", "dark");
        }
    });
}

// Settings Panel Syncing
function initSettingsPanel() {
    const providerSelect = document.getElementById("provider-select");
    const apiKeyGroup = document.getElementById("api-key-group");
    const statusText = document.querySelector("#connection-status .status-text");

    providerSelect.addEventListener("change", () => {
        const val = providerSelect.value;
        if (val === "Mock (Default)") {
            apiKeyGroup.style.display = "none";
            statusText.textContent = "Engine Ready: Mock Mode";
        } else {
            apiKeyGroup.style.display = "flex";
            statusText.textContent = `Engine Configured: ${val}`;
        }
    });
}

// Collapsible Prompt Inspector
function initCollapsibleInspector() {
    const llmToggle = document.getElementById("btn-toggle-llm-inspector");
    const llmBody = document.getElementById("llm-inspector-body");
    const llmPanel = document.getElementById("llm-inspector-panel");
    if (llmToggle && llmBody && llmPanel) {
        llmToggle.addEventListener("click", () => {
            const isOpen = llmPanel.classList.toggle("open");
            llmBody.style.display = isOpen ? "flex" : "none";
        });
    }
}

// Check if server is online and detect environment keys
async function checkAPIProviders() {
    try {
        const response = await fetch(`${API_BASE}/providers`);
        const data = await response.json();
        const providerSelect = document.getElementById("provider-select");
        const statusText = document.querySelector("#connection-status .status-text");
        
        if (data.env_openai || data.env_anthropic) {
            statusText.innerHTML = `Engine Ready: Envs loaded`;
            const infoText = [];
            if (data.env_openai) infoText.push("OpenAI");
            if (data.env_anthropic) infoText.push("Anthropic");
            console.log(`Preloaded environment keys found: ${infoText.join(", ")}`);
        }
    } catch (err) {
        console.error("FastAPI server offline or unreachable:", err);
        const statusIndicator = document.querySelector("#connection-status .status-indicator");
        const statusText = document.querySelector("#connection-status .status-text");
        statusIndicator.className = "status-indicator error";
        statusText.textContent = "Engine Offline: Start backend";
    }
}

// Step Navigation Routing
function navigateToStep(stepNum) {
    if (stepNum < 1 || stepNum > 6) return;
    
    // Hide main wizard step indicator when in Phase 2
    const mainIndicator = document.querySelector(".step-indicator");
    if (mainIndicator) {
        mainIndicator.style.display = (stepNum === 6) ? "none" : "flex";
    }
    
    // Deactivate all step nodes and views
    document.querySelectorAll(".step-node").forEach(node => {
        const num = parseInt(node.getAttribute("data-step"));
        node.classList.remove("active");
        if (num < stepNum) {
            node.classList.add("completed");
            node.classList.add("clickable");
        } else {
            node.classList.remove("completed");
            if (num > stepNum && !node.classList.contains("clickable")) {
                node.classList.remove("clickable");
            }
        }
    });
    
    const nodeEl = document.getElementById(`step-node-${stepNum}`);
    if (nodeEl) nodeEl.classList.add("active");
    
    // Switch views
    document.querySelectorAll(".step-view").forEach(view => {
        view.classList.remove("active");
    });
    
    const viewEl = document.getElementById(`step-view-${stepNum}`);
    if (viewEl) viewEl.classList.add("active");
    
    STATE.currentStep = stepNum;
}

function initNavigationControls() {
    document.querySelectorAll(".step-node").forEach(node => {
        node.addEventListener("click", () => {
            if (node.classList.contains("clickable")) {
                const step = parseInt(node.getAttribute("data-step"));
                navigateToStep(step);
            }
        });
    });

    document.getElementById("btn-goto-analyze").addEventListener("click", () => {
        navigateToStep(3);
        startSemanticAnalysis();
    });

    document.getElementById("btn-back-to-profile").addEventListener("click", () => {
        navigateToStep(2);
    });

    document.getElementById("btn-apply-cleaning").addEventListener("click", () => {
        executeCleaningOperations();
    });

    document.getElementById("btn-goto-validation").addEventListener("click", () => {
        navigateToStep(5);
        runValidationReport();
    });

    document.getElementById("btn-download-csv").addEventListener("click", () => {
        window.location.href = `${API_BASE}/export?format=csv`;
    });
    
    document.getElementById("btn-download-xlsx").addEventListener("click", () => {
        window.location.href = `${API_BASE}/export?format=xlsx`;
    });

    document.getElementById("btn-download-ods").addEventListener("click", () => {
        window.location.href = `${API_BASE}/export?format=ods`;
    });

    document.getElementById("btn-download-report").addEventListener("click", () => {
        window.location.href = `${API_BASE}/report`;
    });

    document.getElementById("btn-restart").addEventListener("click", () => {
        document.querySelectorAll(".step-node").forEach(node => {
            node.classList.remove("clickable", "completed");
        });
        navigateToStep(1);
    });
}

// File Size Formatter
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// Health Quality Score Calculator
function calculateAndRenderHealthScore(profile, anomalies, isAfter = false, precalculatedScore = null) {
    const rows = profile.num_rows;
    const cols = profile.num_cols;
    if (rows === 0) return 0;
    
    let score = precalculatedScore;
    if (score === null) {
        const totalCells = rows * cols;
        
        // Calculate total missing (NaNs + Disguised Nulls)
        let missing = 0;
        profile.columns.forEach(c => {
            missing += c.missing_count + (c.disguised_nulls_count || 0);
        });
        
        let duplicates = profile.duplicate_rows || 0;
        
        let outliers = 0;
        if (anomalies && anomalies.outliers) {
            Object.values(anomalies.outliers).forEach(o => {
                outliers += o.count;
            });
        }
        
        const totalIssues = missing + (duplicates * cols) + outliers;
        score = Math.max(10, Math.round(100 * (1 - (totalIssues / totalCells))));
    }
    
    // Render Gauge
    const suffix = isAfter ? 'after' : 'before';
    const textEl = document.getElementById(`score-text-${suffix}`);
    const barEl = document.getElementById(`score-ring-bar-${suffix}`);
    const labelEl = document.getElementById(`score-label-${suffix}`);
    
    if (textEl) textEl.textContent = `${score}%`;
    if (barEl) {
        const circumference = 238.76;
        const offset = circumference * (1 - score / 100);
        barEl.style.strokeDashoffset = offset;
    }
    
    if (labelEl) {
        if (score >= 90) {
            labelEl.textContent = 'Excellent';
            labelEl.className = 'score-label label-excellent';
        } else if (score >= 75) {
            labelEl.textContent = 'Good';
            labelEl.className = 'score-label label-excellent';
        } else {
            labelEl.textContent = 'Needs Cleaning';
            labelEl.className = 'score-label label-needs-cleaning';
        }
    }
    
    return score;
}

// File Upload Engine (Module 1)
function initUploadZone() {
    const zone = document.getElementById("upload-zone");
    const fileInput = document.getElementById("file-input");

    zone.addEventListener("click", () => fileInput.click());

    zone.addEventListener("dragover", (e) => {
        e.preventDefault();
        zone.classList.add("dragover");
    });

    zone.addEventListener("dragleave", () => {
        zone.classList.remove("dragover");
    });

    zone.addEventListener("drop", (e) => {
        e.preventDefault();
        zone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleUploadedFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleUploadedFile(fileInput.files[0]);
        }
    });
}

// Handle Mock datasets sandbox trigger
function initQuickDemo() {
    document.querySelectorAll(".test-data-btn").forEach(btn => {
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            const datasetType = btn.getAttribute("data-type");
            const csvContent = SAMPLE_DATASETS[datasetType];
            
            const blob = new Blob([csvContent], { type: 'text/csv' });
            const file = new File([blob], `${datasetType}_dirty.csv`, { type: 'text/csv' });
            
            handleUploadedFile(file);
        });
    });
}

function showUploadError(message, hint) {
    const banner = document.getElementById("upload-error");
    const msgEl = document.getElementById("upload-error-message");
    const hintEl = document.getElementById("upload-error-hint");
    if (!banner || !msgEl) return;
    msgEl.textContent = message;
    if (hintEl) hintEl.textContent = hint || "";
    banner.style.display = "flex";
}

function clearUploadError() {
    const banner = document.getElementById("upload-error");
    if (banner) banner.style.display = "none";
}

// Maps a server message to a concrete next step for the user.
function uploadErrorHint(message) {
    const m = (message || "").toLowerCase();
    if (m.includes("xlrd")) return "Re-save the file as .xlsx, or run: pip install xlrd";
    if (m.includes("unsupported file format")) return "Accepted: .csv, .tsv, .xlsx, .xls, .ods";
    if (m.includes("exceeds")) return "Split the file or sample a subset of the rows.";
    if (m.includes("no data rows") || m.includes("empty")) return "Check that the sheet has rows below the header.";
    if (m.includes("parse")) return "Check the delimiter and that every row has the same number of fields.";
    if (m.includes("profiling")) return "The file loaded but profiling failed - please report this message.";
    return "";
}

async function handleUploadedFile(file) {
    clearUploadError();
    const formData = new FormData();
    formData.append("file", file);
    
    const dropzoneContent = document.querySelector(".dropzone-content");
    const originalHTML = dropzoneContent.innerHTML;
    dropzoneContent.innerHTML = `
        <div class="spinner-container" style="width: 40px; height: 40px; margin-bottom: 1rem;">
            <div class="double-bounce1"></div>
            <div class="double-bounce2"></div>
        </div>
        <h3>Uploading & Profiling...</h3>
        <p>Analyzing statistical schemas</p>
    `;

    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: "POST",
            body: formData
        });
        
        if (!response.ok) {
            let detail = `Server returned HTTP ${response.status}.`;
            try {
                const err = await response.json();
                detail = err.detail || detail;
            } catch (parseErr) {
                // Non-JSON error body; keep the status-code message.
            }
            throw new Error(detail);
        }

        let data;
        try {
            data = await response.json();
        } catch (parseErr) {
            throw new Error("The server response was not valid JSON. This usually means the dataset contains values that could not be encoded.");
        }
        
        // Save to state
        STATE.filename = data.filename;
        STATE.filesize = formatFileSize(file.size);
        STATE.profile = data.profile;
        STATE.anomalies = data.anomalies;
        
        // Update header metadata labels
        document.getElementById("meta-filename").textContent = STATE.filename;
        document.getElementById("meta-filesize").textContent = STATE.filesize;
        document.getElementById("meta-rows").textContent = data.profile.num_rows;
        document.getElementById("meta-cols").textContent = data.profile.num_cols;
        
        // Compute health score
        calculateAndRenderHealthScore(data.profile, data.anomalies, false, data.quality_score);
        
        // Populate Profiling details
        renderProfilingScreen(data.profile, data.anomalies, data.preview);
        
        navigateToStep(2);
        
    } catch (err) {
        showUploadError(err.message, uploadErrorHint(err.message));
        console.error("Upload failed:", err);
    } finally {
        dropzoneContent.innerHTML = originalHTML;
    }
}

// Populate Profiling (Module 2 & 3)
function renderProfilingScreen(profile, anomalies, preview) {
    // 1. KPI cards
    document.getElementById("profile-shape").textContent = `${profile.num_rows || 0} Rows x ${profile.num_cols || 0} Cols`;
    
    // Count total anomalies safely
    let anomalyCount = 0;
    if (anomalies) {
        if (anomalies.missing_values) {
            Object.values(anomalies.missing_values).forEach(m => {
                if (m && typeof m.count === 'number') {
                    anomalyCount += m.count;
                }
            });
        }
        if (anomalies.duplicates && Array.isArray(anomalies.duplicates)) {
            anomalyCount += anomalies.duplicates.length;
        }
        if (anomalies.outliers) {
            Object.values(anomalies.outliers).forEach(o => {
                if (o && typeof o.count === 'number') {
                    anomalyCount += o.count;
                }
            });
        }
        if (anomalies.casing_inconsistencies) {
            Object.values(anomalies.casing_inconsistencies).forEach(c => {
                if (c) {
                    anomalyCount += Object.keys(c).length;
                }
            });
        }
    }
    
    document.getElementById("profile-anomalies").textContent = `${anomalyCount} Detected`;
    document.getElementById("profile-duplicates").textContent = `${profile.duplicate_rows || 0} Rows`;
    document.getElementById("profile-memory").textContent = profile.memory_usage || "Unknown";

    // 2. Render Quality Issues Detailed Table
    const qualityBody = document.getElementById("quality-issues-body");
    let qualityHTML = "";
    if (profile.columns && Array.isArray(profile.columns)) {
        profile.columns.forEach(col => {
            const colAnomMissing = (anomalies && anomalies.missing_values) ? anomalies.missing_values[col.name] : null;
            const colAnomOutlier = (anomalies && anomalies.outliers) ? anomalies.outliers[col.name] : null;
            const colAnomCasing = (anomalies && anomalies.casing_inconsistencies) ? anomalies.casing_inconsistencies[col.name] : null;
            
            let totalMissing = (col.missing_count || 0) + (col.disguised_nulls_count || 0);
            let missingPct = profile.num_rows ? ((totalMissing / profile.num_rows) * 100).toFixed(1) : "0.0";
            let outlierCount = colAnomOutlier ? (colAnomOutlier.count || 0) : 0;
            
            let casingInfo = "Standard Casing";
            if (colAnomCasing) {
                const variations = Object.values(colAnomCasing || {}).flat().filter(x => x !== undefined && x !== null);
                casingInfo = `Case variation: [${variations.slice(0, 3).map(String).join(", ")}]`;
            } else if (col.disguised_nulls_count > 0) {
                casingInfo = `Disguised nulls found: ${col.disguised_nulls_count}`;
            }
            
            let statusBadge = totalMissing > 0 || outlierCount > 0 || colAnomCasing ? 
                `<span class="rec-badge outliers"><i class="fa-solid fa-triangle-exclamation"></i> Issues</span>` : 
                `<span class="rec-badge active"><i class="fa-solid fa-circle-check"></i> Clean</span>`;
            
            qualityHTML += `
                <tr class="${totalMissing > 0 || outlierCount > 0 || colAnomCasing ? 'has-anomaly-row' : ''}">
                    <td><b>${escapeHTML(col.name)}</b></td>
                    <td><span class="sample-pill">${escapeHTML(casingInfo)}</span></td>
                    <td>${totalMissing}</td>
                    <td>${missingPct}%</td>
                    <td>${outlierCount}</td>
                    <td>${statusBadge}</td>
                </tr>
            `;
        });
    }
    qualityBody.innerHTML = qualityHTML;

    // 3. Render Raw Data Preview Table
    const previewTable = document.getElementById("raw-preview-table");
    if (preview && preview.length > 0) {
        const cols = Object.keys(preview[0]);
        let headersHTML = "<tr>";
        cols.forEach(c => headersHTML += `<th>${escapeHTML(c)}</th>`);
        headersHTML += "</tr>";
        
        let bodyHTML = "";
        preview.forEach(row => {
            bodyHTML += "<tr>";
            cols.forEach(c => {
                const rawVal = row[c];
                const valStr = rawVal === null || rawVal === undefined ? `<span style="color:var(--text-muted);font-style:italic;">Null</span>` : escapeHTML(String(rawVal));
                bodyHTML += `<td>${valStr}</td>`;
            });
            bodyHTML += "</tr>";
        });
        
        previewTable.querySelector("thead").innerHTML = headersHTML;
        previewTable.querySelector("tbody").innerHTML = bodyHTML;
    }

    // 4. Render Columns profile metadata table
    const profileBody = document.getElementById("columns-profile-body");
    let profileHTML = "";
    
    if (profile.columns && Array.isArray(profile.columns)) {
        profile.columns.forEach(col => {
            let samplePillsHTML = `<div class="sample-pill-container">`;
            if (col.sample_values && Array.isArray(col.sample_values)) {
                // Limit rendering to first 8 sample values for responsiveness on large datasets (10k+ rows)
                col.sample_values.slice(0, 8).forEach(v => {
                    let valStr = "";
                    if (v === null || v === undefined) {
                        valStr = "null";
                    } else if (typeof v === "number" && isNaN(v)) {
                        valStr = "NaN";
                    } else {
                        valStr = String(v);
                    }
                    samplePillsHTML += `<span class="sample-pill">${escapeHTML(valStr)}</span>`;
                });
            }
            samplePillsHTML += `</div>`;
            
            const hasMissing = (anomalies && anomalies.missing_values) ? anomalies.missing_values[col.name] : null;
            const hasOutlier = (anomalies && anomalies.outliers) ? anomalies.outliers[col.name] : null;
            const hasCasing = (anomalies && anomalies.casing_inconsistencies) ? anomalies.casing_inconsistencies[col.name] : null;
            
            let alerts = "";
            if (hasMissing) alerts += `<span class="rec-badge nulls" style="margin-left: 0.5rem;"><i class="fa-solid fa-circle-question"></i> ${hasMissing.count || 0} nulls</span>`;
            if (hasOutlier) alerts += `<span class="rec-badge outliers" style="margin-left: 0.5rem;"><i class="fa-solid fa-triangle-exclamation"></i> ${hasOutlier.count || 0} outliers</span>`;
            if (hasCasing) alerts += `<span class="rec-badge mapping" style="margin-left: 0.5rem;"><i class="fa-solid fa-spell-check"></i> casing issue</span>`;

            profileHTML += `
                <tr>
                    <td style="font-weight: 500;">${escapeHTML(col.name)}${alerts}</td>
                    <td style="font-family: var(--font-mono); font-size: 0.75rem;">${col.type || "unknown"}</td>
                    <td>${(col.missing_count || 0) + (col.disguised_nulls_count || 0)}</td>
                    <td>${col.unique_count || 0}</td>
                    <td>${samplePillsHTML}</td>
                </tr>
            `;
        });
    }
    
    profileBody.innerHTML = profileHTML;
}

// Helper to escape HTML in any value rendered into innerHTML.
// Defined once: there used to be a second definition later in this file that
// silently overrode this one and turned falsy values such as 0 into "".
function escapeHTML(value) {
    if (value === null || value === undefined) return "";
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Query Recommendations (Module 4 & 5)
async function startSemanticAnalysis() {
    const loadingCard = document.getElementById("analysis-loading");
    const resultsContainer = document.getElementById("analysis-results");
    
    loadingCard.style.display = "block";
    resultsContainer.style.display = "none";
    
    const provider = document.getElementById("provider-select").value;
    const apiKey = document.getElementById("api-key-input").value;
    
    try {
        const response = await fetch(`${API_BASE}/analyze`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ provider, api_key: apiKey })
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Analysis recommendation lookup failed.");
        }
        
        const data = await response.json();
        STATE.recommendations = data.recommendations;
        
        // Cache LLM Prompts & Responses
        document.getElementById("inspector-system-prompt").textContent = data.system_prompt || "Mock System Prompt (Default rules applied)";
        document.getElementById("inspector-user-prompt").textContent = data.user_prompt || "Mock User Profile (Default parameters applied)";
        document.getElementById("inspector-raw-response").textContent = typeof data.recommendations === 'object' ? JSON.stringify(data.recommendations, null, 2) : data.recommendations;
        
        // Render estimated changes badges
        const summaryBadgeContainer = document.getElementById("estimated-changes-counts");
        if (summaryBadgeContainer) {
            let badges = [];
            if (data.recommendations.drop_duplicates) {
                badges.push(`<span class="rec-badge outliers"><i class="fa-solid fa-clone"></i> Duplicate Purge</span>`);
            }
            
            let imputationCount = 0;
            let castingCount = 0;
            let clippingCount = 0;
            let disguisedCount = 0;
            let mappingsCount = 0;
            
            data.recommendations.columns.forEach(c => {
                if (c.imputation_strategy) imputationCount++;
                if (c.type_cast) castingCount++;
                if (c.outlier_strategy) clippingCount++;
                if (c.disguised_nulls && c.disguised_nulls.length > 0) disguisedCount++;
                if (c.value_mappings && Object.keys(c.value_mappings).length > 0) mappingsCount++;
            });
            
            if (imputationCount > 0) badges.push(`<span class="rec-badge nulls"><i class="fa-solid fa-arrows-to-dot"></i> Impute: ${imputationCount} cols</span>`);
            if (castingCount > 0) badges.push(`<span class="rec-badge type-cast"><i class="fa-solid fa-arrow-down-up-across-line"></i> Cast: ${castingCount} cols</span>`);
            if (clippingCount > 0) badges.push(`<span class="rec-badge outliers"><i class="fa-solid fa-scissors"></i> Clip: ${clippingCount} cols</span>`);
            if (disguisedCount > 0) badges.push(`<span class="rec-badge nulls"><i class="fa-solid fa-eye-slash"></i> Disguised: ${disguisedCount} cols</span>`);
            if (mappingsCount > 0) badges.push(`<span class="rec-badge mapping"><i class="fa-solid fa-spell-check"></i> Maps: ${mappingsCount} cols</span>`);
            
            summaryBadgeContainer.innerHTML = badges.join(" ");
        }

        renderRecommendations(data.recommendations);
        
        loadingCard.style.display = "none";
        resultsContainer.style.display = "block";
        
    } catch (err) {
        alert(`Error running semantic analysis: ${err.message}`);
        navigateToStep(2);
    }
}

function renderRecommendations(recs) {
    document.getElementById("analysis-summary-text").textContent = recs.explanation;
    
    const recsList = document.getElementById("recommendations-list");
    recsList.innerHTML = "";
    
    // Add duplicate check recommendation if suggested
    if (recs.drop_duplicates) {
        const dupItem = document.createElement("div");
        dupItem.className = "rec-item active-cleaning";
        dupItem.setAttribute("data-type", "duplicates");
        
        dupItem.innerHTML = `
            <div class="rec-checkbox-wrapper">
                <div class="custom-checkbox"><i class="fa-solid fa-check"></i></div>
            </div>
            <div class="rec-details">
                <div class="rec-meta">
                    <span class="rec-col-name">Duplicate Rows</span>
                    <span class="rec-badge outliers">Structural (100% Confidence)</span>
                </div>
                <div class="rec-explanation">
                    Identified redundant duplicate rows in the dataset. Checked rows will be purged to ensure strict data uniqueness.
                </div>
            </div>
        `;
        dupItem.addEventListener("click", () => {
            dupItem.classList.toggle("active-cleaning");
        });
        recsList.appendChild(dupItem);
    }
    
    // Columns recommendations
    recs.columns.forEach(colRec => {
        const item = document.createElement("div");
        item.className = "rec-item active-cleaning";
        item.setAttribute("data-column", colRec.column);
        
        let badgesHTML = "";
        let modificationsHTML = "";
        
        const conf = colRec.confidence || 95;
        badgesHTML += `<span class="rec-badge active" style="background:rgba(16,185,129,0.08);color:var(--color-success); border-color:rgba(16,185,129,0.2);"><i class="fa-solid fa-shield-check"></i> ${conf}% Confidence</span>`;
        
        if (colRec.type_cast) {
            badgesHTML += `<span class="rec-badge type-cast">Cast: ${colRec.type_cast}</span>`;
        }
        if (colRec.disguised_nulls && colRec.disguised_nulls.length > 0) {
            badgesHTML += `<span class="rec-badge nulls">Disguised Nulls</span>`;
            modificationsHTML += `
                <div class="rec-modifications">
                    <div class="mod-title">NULL Standardizations:</div>
                    <div class="mod-code">${colRec.disguised_nulls.join(", ")} &rarr; NaN</div>
                </div>
            `;
        }
        if (colRec.value_mappings && Object.keys(colRec.value_mappings).length > 0) {
            badgesHTML += `<span class="rec-badge mapping">Standardize Map</span>`;
            
            let mappingList = [];
            for (const [k, v] of Object.entries(colRec.value_mappings)) {
                mappingList.push(`"${k}": "${v}"`);
            }
            let mapSnippet = mappingList.join(", ");
            if (colRec.value_mappings_preview_only) {
                mapSnippet += `, ... and ${colRec.value_mappings_total_count - Object.keys(colRec.value_mappings).length} more standardizations (summarized for UI responsiveness)`;
            }
            modificationsHTML += `
                <div class="rec-modifications">
                    <div class="mod-title">Categorical Maps:</div>
                    <div class="mod-code">${mapSnippet}</div>
                </div>
            `;
        }
        if (colRec.outlier_strategy) {
            badgesHTML += `<span class="rec-badge outliers">${colRec.outlier_strategy} Outliers</span>`;
        }
        if (colRec.imputation_strategy) {
            badgesHTML += `<span class="rec-badge type-cast">Impute: ${colRec.imputation_strategy}</span>`;
        }
        
        // Add reason field
        const reasonStr = colRec.reason || "Underlying data checks align with semantic expectations.";
        
        item.innerHTML = `
            <div class="rec-checkbox-wrapper">
                <div class="custom-checkbox"><i class="fa-solid fa-check"></i></div>
            </div>
            <div class="rec-details">
                <div class="rec-meta">
                    <span class="rec-col-name">${escapeHTML(colRec.column)}</span>
                    ${badgesHTML}
                </div>
                <div class="rec-explanation">
                    ${escapeHTML(colRec.explanation)}
                </div>
                <div class="rec-reason" style="margin-top:0.35rem; font-size:0.8rem; color:var(--text-muted); font-style:italic;">
                    <b>Reason:</b> ${escapeHTML(reasonStr)}
                </div>
                ${modificationsHTML}
            </div>
        `;
        
        item.addEventListener("click", () => {
            item.classList.toggle("active-cleaning");
        });
        
        recsList.appendChild(item);
    });
    
    const toggleBtn = document.getElementById("btn-toggle-all-checks");
    toggleBtn.textContent = "Deselect All";
    toggleBtn.onclick = () => {
        const allItems = recsList.querySelectorAll(".rec-item");
        const anySelected = Array.from(allItems).some(item => item.classList.contains("active-cleaning"));
        
        allItems.forEach(item => {
            if (anySelected) {
                item.classList.remove("active-cleaning");
            } else {
                item.classList.add("active-cleaning");
            }
        });
        
        toggleBtn.textContent = anySelected ? "Select All" : "Deselect All";
    };
}

// Execute Cleaning (Module 6)
async function executeCleaningOperations() {
    const activeCols = [];
    let dropDuplicates = false;
    
    document.querySelectorAll("#recommendations-list .rec-item").forEach(item => {
        if (item.classList.contains("active-cleaning")) {
            const colName = item.getAttribute("data-column");
            const isDup = item.getAttribute("data-type") === "duplicates";
            
            if (colName) activeCols.push(colName);
            if (isDup) dropDuplicates = true;
        }
    });
    
    navigateToStep(4);
    
    const consoleOutput = document.getElementById("console-output");
    consoleOutput.innerHTML = `
        <div class="console-line comment"># Initializing Python pandas cleaning session...</div>
        <div class="console-line info">[INFO] Connecting to Cocoon Pandas Engine</div>
        <div class="console-line info">[INFO] Active target list: ${activeCols.join(", ") || 'none'} (Duplicates: ${dropDuplicates})</div>
    `;
    
    STATE.recommendations.drop_duplicates = dropDuplicates;
    
    // Processing indicators
    const progressFill = document.getElementById("cleaning-progress-fill");
    const progressPct = document.getElementById("cleaning-progress-pct");
    const execTimeVal = document.getElementById("cleaning-exec-time");
    const opsCountVal = document.getElementById("cleaning-ops-count");
    const consoleExecutedOps = document.getElementById("console-executed-ops");
    
    if (progressFill) progressFill.style.width = "0%";
    if (progressPct) progressPct.textContent = "0%";
    if (execTimeVal) execTimeVal.textContent = "0.0s";
    if (opsCountVal) opsCountVal.textContent = "0";
    if (consoleExecutedOps) consoleExecutedOps.textContent = "0 Operations Executed";
    
    const startTime = performance.now();
    const timerInterval = setInterval(() => {
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
        if (execTimeVal) execTimeVal.textContent = `${elapsed}s`;
    }, 100);
    
    try {
        const response = await fetch(`${API_BASE}/clean`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ active_cols: activeCols })
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Cleaning execution failed.");
        }
        
        const data = await response.json();
        
        STATE.history = data.history;
        STATE.cleanedPreview = data.preview;
        
        // Progress animation filling
        let currentProg = 0;
        const totalItems = data.history.length || 1;
        
        for (let index = 0; index < data.history.length; index++) {
            const item = data.history[index];
            currentProg = Math.round(((index + 1) / totalItems) * 100);
            
            if (progressFill) progressFill.style.width = `${currentProg}%`;
            if (progressPct) progressPct.textContent = `${currentProg}%`;
            
            // Log code execution to console
            const cLine = document.createElement("div");
            cLine.className = "console-line comment";
            cLine.textContent = `# Step ${index + 1}: ${item.action} on '${item.column}' - ${item.details}`;
            consoleOutput.appendChild(cLine);
            
            const codeLine = document.createElement("div");
            codeLine.className = "console-line code";
            codeLine.textContent = item.code;
            consoleOutput.appendChild(codeLine);
            
            consoleOutput.scrollTop = consoleOutput.scrollHeight;
            await sleep(250);
        }
        
        clearInterval(timerInterval);
        const totalTime = ((performance.now() - startTime) / 1000).toFixed(1);
        if (execTimeVal) execTimeVal.textContent = `${totalTime}s`;
        if (opsCountVal) opsCountVal.textContent = data.history.length;
        if (consoleExecutedOps) consoleExecutedOps.textContent = `${data.history.length} Operations Executed`;
        if (progressFill) progressFill.style.width = "100%";
        if (progressPct) progressPct.textContent = "100%";
        
        const endLine = document.createElement("div");
        endLine.className = "console-line success";
        endLine.textContent = `[SUCCESS] Automated cleaning applied. Dataset dimension is preserved inside Session state.`;
        consoleOutput.appendChild(endLine);
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
        
        // Render success log details panel
        const successLogPanel = document.getElementById("cleaning-success-summary");
        const successLogList = document.getElementById("cleaning-success-audit-list");
        if (successLogPanel && successLogList) {
            successLogPanel.style.display = "block";
            let auditItems = "";
            data.history.forEach(item => {
                auditItems += `<li><i class="fa-solid fa-circle-check"></i> <b>[${item.column}]</b> ${item.action}: ${item.details}</li>`;
            });
            successLogList.innerHTML = auditItems;
        }

        renderCleanedPreviewTable(data.preview);
        document.getElementById("console-nav-actions").style.display = "flex";
        
    } catch (err) {
        clearInterval(timerInterval);
        const errLine = document.createElement("div");
        errLine.className = "console-line error";
        errLine.textContent = `[ERROR] Execution halted: ${err.message}`;
        consoleOutput.appendChild(errLine);
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function renderCleanedPreviewTable(preview) {
    const tablePanel = document.getElementById("cleaned-preview-panel");
    const previewTable = document.getElementById("cleaned-preview-table");
    
    if (preview && preview.length > 0) {
        tablePanel.style.display = "block";
        const cols = Object.keys(preview[0]);
        
        let headersHTML = "<tr>";
        cols.forEach(c => headersHTML += `<th>${escapeHTML(c)}</th>`);
        headersHTML += "</tr>";

        let bodyHTML = "";
        preview.forEach(row => {
            bodyHTML += "<tr>";
            cols.forEach(c => {
                const raw = row[c];
                const val = (raw === null || raw === undefined)
                    ? `<span style="color:var(--text-muted);font-style:italic;">Null</span>`
                    : escapeHTML(raw);
                bodyHTML += `<td>${val}</td>`;
            });
            bodyHTML += "</tr>";
        });
        
        previewTable.querySelector("thead").innerHTML = headersHTML;
        previewTable.querySelector("tbody").innerHTML = bodyHTML;
    }
}

// Validate Dataset (Module 7)
async function runValidationReport() {
    try {
        const response = await fetch(`${API_BASE}/validate`);
        if (!response.ok) {
            throw new Error("Could not load validation comparisons.");
        }
        
        const data = await response.json();
        const audit = data.validation.comparison;
        
        // Defensive checks for DOM nodes
        const setElVal = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };
        
        // Missing Values Audit
        setElVal("audit-missing-before", audit.missing_values.before);
        setElVal("audit-missing-after", audit.missing_values.after);
        setElVal("audit-missing-diff", `${audit.missing_values.resolved} Resolved`);
        
        // Outliers Audit
        setElVal("audit-outliers-before", audit.outliers.before);
        setElVal("audit-outliers-after", audit.outliers.after);
        setElVal("audit-outliers-diff", `${audit.outliers.resolved} Resolved`);
        
        // Duplicates Audit
        setElVal("audit-dup-before", audit.duplicates.before);
        setElVal("audit-dup-after", audit.duplicates.after);
        setElVal("audit-dup-diff", `${audit.duplicates.resolved} Resolved`);
        
        // Rows Audit
        setElVal("audit-rows-before", audit.rows.before);
        setElVal("audit-rows-after", audit.rows.after);
        
        const rowDiff = audit.rows.diff;
        const rowBadge = document.getElementById("audit-rows-diff");
        if (rowBadge) {
            if (rowDiff === 0) {
                rowBadge.textContent = "No Change";
                rowBadge.className = "audit-badge blue-badge";
            } else {
                rowBadge.textContent = `${Math.abs(rowDiff)} Dropped`;
                rowBadge.className = "audit-badge green-badge";
            }
        }
        
        // Dynamic Health Score Before vs After calculation based on quality metrics
        const beforeScore = calculateAndRenderHealthScore(STATE.profile, STATE.anomalies, false, audit.before_score);
        const afterScore = audit.after_score || 100;
        
        setElVal("export-score-before", `${beforeScore}%`);
        setElVal("export-score-after", `${afterScore}%`);
        
        // Render detailed audit table dynamically
        const detailedBody = document.getElementById("detailed-audit-body");
        if (detailedBody && audit.detailed) {
            const labelMap = {
                "missing_values": "Missing Values / Disguised Nulls",
                "duplicates": "Duplicate Rows",
                "invalid_dates": "Invalid / Unstandardized Dates",
                "email_issues": "Invalid Email Addresses",
                "phone_issues": "Invalid Phone Numbers",
                "casing_issues": "Categorical Casing Inconsistencies",
                "outliers": "Numeric Outliers (IQR)",
                "invalid_numeric": "Invalid Numeric Formats",
                "semantic_inconsistencies": "Semantic Category Mismatches"
            };
            
            let detailedHTML = "";
            for (const [key, metrics] of Object.entries(audit.detailed)) {
                const label = labelMap[key] || key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
                const resolvedBadge = metrics.resolved > 0 ? 
                    `<span class="rec-badge active"><i class="fa-solid fa-circle-check"></i> ${metrics.resolved} Resolved</span>` : 
                    `<span class="rec-badge" style="background:rgba(255,255,255,0.03); border-color:rgba(255,255,255,0.1); color:var(--text-muted);">${metrics.resolved} Resolved</span>`;
                const remainingBadge = metrics.remaining > 0 ? 
                    `<span class="rec-badge outliers"><i class="fa-solid fa-triangle-exclamation"></i> ${metrics.remaining} Remaining</span>` : 
                    `<span class="rec-badge active"><i class="fa-solid fa-circle-check"></i> Clean</span>`;
                detailedHTML += `
                    <tr>
                        <td><b>${escapeHTML(label)}</b></td>
                        <td>${metrics.before}</td>
                        <td>${metrics.after}</td>
                        <td>${resolvedBadge}</td>
                        <td>${remainingBadge}</td>
                    </tr>
                `;
            }
            detailedBody.innerHTML = detailedHTML;
        }
        
        // Render phone validation statistics summary
        const phoneCard = document.getElementById("phone-validation-summary-card");
        if (phoneCard) {
            const phoneVal = audit.phone_validation;
            if (phoneVal && phoneVal.has_phone) {
                phoneCard.style.display = "block";
                setElVal("phone-valid-before", phoneVal.before.valid);
                setElVal("phone-valid-after", phoneVal.after.valid);
                setElVal("phone-invalid-before", phoneVal.before.invalid);
                setElVal("phone-invalid-after", phoneVal.after.invalid);
            } else {
                phoneCard.style.display = "none";
            }
        }
        
        // Render Column-by-Column validation details
        const columnReport = data.validation.column_validation_report;
        const columnBody = document.getElementById("column-validation-body");
        if (columnBody && columnReport) {
            let columnHTML = "";
            for (const [colName, stats] of Object.entries(columnReport)) {
                const validBadge = stats.valid > 0 ? 
                    `<span class="rec-badge active"><i class="fa-solid fa-circle-check"></i> ${stats.valid} Valid</span>` :
                    `<span class="rec-badge" style="background:rgba(255,255,255,0.03); border-color:rgba(255,255,255,0.1); color:var(--text-muted);">${stats.valid} Valid</span>`;
                const invalidBadge = stats.invalid > 0 ? 
                    `<span class="rec-badge outliers"><i class="fa-solid fa-triangle-exclamation"></i> ${stats.invalid} Invalid</span>` :
                    `<span class="rec-badge active"><i class="fa-solid fa-circle-check"></i> Clean</span>`;
                const standardizedBadge = stats.standardized > 0 ?
                    `<span class="rec-badge active"><i class="fa-solid fa-wand-magic-sparkles"></i> ${stats.standardized} Standardized</span>` :
                    `<span class="rec-badge" style="background:rgba(255,255,255,0.03); border-color:rgba(255,255,255,0.1); color:var(--text-muted);">0 Standardized</span>`;
                const unresolvedBadge = stats.unresolved > 0 ? 
                    `<span class="rec-badge outliers"><i class="fa-solid fa-circle-info"></i> ${stats.unresolved} Unresolved</span>` :
                    `<span class="rec-badge active"><i class="fa-solid fa-check"></i> Resolved</span>`;
                    
                columnHTML += `
                    <tr>
                        <td><b>${escapeHTML(colName)}</b></td>
                        <td><span class="rec-badge info-badge" style="background:rgba(59,130,246,0.1); color:#60a5fa; border:1px solid rgba(59,130,246,0.2); font-size: 0.8rem; padding: 0.15rem 0.5rem; text-transform: none;">${escapeHTML(stats.semantic_type)}</span></td>
                        <td>${validBadge}</td>
                        <td>${invalidBadge}</td>
                        <td>${standardizedBadge}</td>
                        <td>${unresolvedBadge}</td>
                    </tr>
                `;
            }
            columnBody.innerHTML = columnHTML;
        }
        
    } catch (err) {
        alert(`Validation report failed: ${err.message}`);
    }
}

// =======================================================
// PHASE 2 — INTELLIGENT ML PIPELINE CONTROLLER
// =======================================================

STATE.ml = {
    summary: null,
    analysis: null,
    results: null,
    selectedTarget: "",
    selectedFeatures: [],
    problemType: ""
};

// Initialize Phase 2 Event Listeners
function initPhase2ML() {
    const btnContinue = document.getElementById("btn-continue-phase2");
    if (btnContinue) {
        btnContinue.addEventListener("click", () => {
            navigateToStep(6);
            startPhase2DatasetUnderstanding();
        });
    }

    const btnAnalyzeObj = document.getElementById("btn-analyze-objective");
    if (btnAnalyzeObj) {
        btnAnalyzeObj.addEventListener("click", analyzeMLObjective);
    }

    const btnMLBack1 = document.getElementById("btn-ml-back-1");
    if (btnMLBack1) {
        btnMLBack1.addEventListener("click", () => {
            showMLSubView(1);
            updateMLStepIndicator(1);
        });
    }

    const btnMLContinue2 = document.getElementById("btn-ml-continue-2");
    if (btnMLContinue2) {
        btnMLContinue2.addEventListener("click", executeMLPipelineTraining);
    }

    const btnMLBackToVal = document.getElementById("btn-ml-back-to-validation");
    if (btnMLBackToVal) {
        btnMLBackToVal.addEventListener("click", () => {
            navigateToStep(5);
        });
    }

    const btnMLReset = document.getElementById("btn-ml-reset");
    if (btnMLReset) {
        btnMLReset.addEventListener("click", () => {
            // Reset state
            document.getElementById("ml-suggested-questions").value = "";
            document.getElementById("ml-custom-question").value = "";
            showMLSubView(1);
            updateMLStepIndicator(1);
        });
    }
}

function showMLSubView(viewNum) {
    document.querySelectorAll(".ml-sub-view").forEach(v => {
        v.style.display = "none";
    });
    const subView = document.getElementById(`ml-sub-view-${viewNum}`);
    if (subView) subView.style.display = "flex";
}

function updateMLStepIndicator(stepNum) {
    document.querySelectorAll(".step-node-ml").forEach(node => {
        const num = parseInt(node.id.replace("ml-step-", ""));
        node.classList.remove("active", "completed");
        if (num < stepNum) {
            node.classList.add("completed");
        } else if (num === stepNum) {
            node.classList.add("active");
        }
    });
}

// 1. DATASET UNDERSTANDING
async function startPhase2DatasetUnderstanding() {
    showMLSubView(1);
    updateMLStepIndicator(1);

    const provider = document.getElementById("provider-select").value;
    const apiKey = document.getElementById("api-key-input").value;

    // Show loading indicators in summary labels
    document.getElementById("ml-summary-name").textContent = "Analyzing...";
    document.getElementById("ml-summary-rows").textContent = "Analyzing...";
    document.getElementById("ml-summary-cols").textContent = "Analyzing...";

    try {
        const response = await fetch(`${API_BASE}/ml/understand`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ provider, api_key: apiKey })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Phase 2 dataset initialization failed.");
        }

        const data = await response.json();
        STATE.ml.summary = data.summary;

        // Populate summary labels
        document.getElementById("ml-summary-name").textContent = data.summary.dataset_name;
        document.getElementById("ml-summary-rows").textContent = data.summary.rows;
        document.getElementById("ml-summary-cols").textContent = data.summary.columns;

        // Helper to render column badges
        const renderBadges = (elementId, list) => {
            const el = document.getElementById(elementId);
            if (!el) return;
            if (list.length === 0) {
                el.innerHTML = `<span style="font-size:0.8rem; color:var(--text-muted); font-style:italic;">None</span>`;
            } else {
                el.innerHTML = list.map(name => `<span class="sample-pill" style="border-color: rgba(99, 102, 241, 0.2); background: rgba(99, 102, 241, 0.05); color:#a5b4fc; font-weight:500;">${escapeHTML(name)}</span>`).join(" ");
            }
        };

        renderBadges("ml-cols-numeric", data.summary.numerical);
        renderBadges("ml-cols-categorical", data.summary.categorical);
        renderBadges("ml-cols-identifier", data.summary.identifier);

        // Populate suggested questions dropdown
        const selectEl = document.getElementById("ml-suggested-questions");
        selectEl.innerHTML = `<option value="">-- Select a suggested question --</option>`;
        data.questions.forEach((q, idx) => {
            const opt = document.createElement("option");
            opt.value = q;
            opt.textContent = `${idx + 1}. ${q}`;
            selectEl.appendChild(opt);
        });

    } catch (err) {
        alert(`Error starting Phase 2: ${err.message}`);
        navigateToStep(5); // Fall back to validation screen
    }
}

// 2. ANALYZE OBJECTIVE
async function analyzeMLObjective() {
    const suggestedSelect = document.getElementById("ml-suggested-questions");
    const customTextarea = document.getElementById("ml-custom-question");

    let question = suggestedSelect.value;
    if (!question && customTextarea.value.trim()) {
        question = customTextarea.value.trim();
    }

    if (!question) {
        alert("Please select a suggested question or describe your prediction objective.");
        return;
    }

    const provider = document.getElementById("provider-select").value;
    const apiKey = document.getElementById("api-key-input").value;

    const btn = document.getElementById("btn-analyze-objective");
    const objectiveError = document.getElementById("ml-objective-error");
    if (objectiveError) objectiveError.style.display = "none";
    const originalText = btn.innerHTML;
    btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Analyzing...`;
    btn.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/ml/analyze-objective`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, provider, api_key: apiKey })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Objective parsing failed.");
        }

        const data = await response.json();
        if (data.analysis.target_available === false || !data.analysis.target) {
            const message = data.analysis.unavailable_reason ||
                "The requested target is not available in this dataset.";
            if (objectiveError) {
                objectiveError.textContent = message;
                objectiveError.style.display = "block";
            }
            STATE.ml.analysis = null;
            return;
        }
        STATE.ml.analysis = data.analysis;
        STATE.ml.activeQuestion = question;

        // Render Sub-view 2
        document.getElementById("ml-intent-summary").textContent = data.analysis.objective_summary;
        document.getElementById("ml-intent-target").textContent = data.analysis.target;
        document.getElementById("ml-intent-problem").textContent = data.analysis.problem_type;

        const explanationEl = document.getElementById("ml-problem-explanation");
        const isForecasting = question.toLowerCase().includes("forecast");
        if (isForecasting) {
            explanationEl.innerHTML = `<strong>Regression (Trend Forecasting)</strong>: Cocoon will model the trend of the target using engineered date-time features (year, month, day, day of week) and regression algorithms. <br/><span style="color: var(--text-muted); font-size: 0.85rem;"><i class="fa-solid fa-circle-info" style="color: var(--color-primary); margin-right: 0.25rem;"></i> Note: Pure auto-regressive time-series forecasting (like ARIMA) is planned as a future extension.</span>`;
            explanationEl.style.borderLeftColor = "var(--color-primary)";
        } else if (data.analysis.problem_type.toLowerCase() === "regression") {
            explanationEl.textContent = "Regression: The target column is numerical/continuous, so Cocoon will train models to predict values.";
            explanationEl.style.borderLeftColor = "var(--color-secondary)";
        } else {
            explanationEl.textContent = "Classification: The target column is categorical, so Cocoon will train models to predict classes.";
            explanationEl.style.borderLeftColor = "var(--color-success)";
        }

        // Render selectors
        renderMLColumnsConfirmation(data.analysis);
        renderAlgorithmRecommendations(data.analysis.problem_type, data.analysis.candidate_algorithms, data.analysis.algorithm_reasons);

        showMLSubView(2);
        updateMLStepIndicator(2);

    } catch (err) {
        alert(`Error analyzing objective: ${err.message}`);
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}


function isProtectedPredictorColumn(colName) {
    const tokens = String(colName).toLowerCase().match(/[a-z0-9]+/g) || [];
    const glued = tokens.join("");
    const protectedTokens = new Set(["age", "gender", "sex"]);
    const idTokens = new Set(["id", "ids", "uuid", "guid", "pk", "key", "idx", "index", "roll", "regno", "serial"]);
    const piiTokens = new Set(["name", "fullname", "firstname", "lastname", "surname", "email", "mail", "phone", "mobile", "contact", "ssn", "address", "aadhaar", "passport", "dob"]);
    return tokens.some(t => protectedTokens.has(t) || idTokens.has(t) || piiTokens.has(t)) ||
        /(id|no|number|code|num)$/.test(glued) && glued.length > 3;
}

function renderMLColumnsConfirmation(analysis) {
    const targetSelect = document.getElementById("ml-target-select");
    const featuresContainer = document.getElementById("ml-features-container");
    const excludedContainer = document.getElementById("ml-excluded-container");

    targetSelect.innerHTML = "";
    featuresContainer.innerHTML = "";
    excludedContainer.innerHTML = "";

    const allColumns = STATE.ml.summary.numerical
        .concat(STATE.ml.summary.categorical)
        .concat(STATE.ml.summary.identifier)
        .concat(STATE.ml.summary.date)
        .concat(STATE.ml.summary.text);

    // 1. Populate Target dropdown
    allColumns.forEach(col => {
        const opt = document.createElement("option");
        opt.value = col;
        opt.textContent = col;
        if (col === analysis.target) {
            opt.selected = true;
        }
        targetSelect.appendChild(opt);
    });

    // Helper to render checkbox lists
    const renderCheckbox = (container, colName, isChecked, listType) => {
        const item = document.createElement("div");
        item.style.display = "flex";
        item.style.alignItems = "center";
        item.style.gap = "0.5rem";
        item.style.fontSize = "0.85rem";
        
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.value = colName;
        cb.checked = isChecked;
        cb.id = `cb-${listType}-${colName}`;
        cb.style.cursor = "pointer";

        const lbl = document.createElement("label");
        lbl.htmlFor = cb.id;
        lbl.textContent = colName;
        lbl.style.cursor = "pointer";

        item.appendChild(cb);
        item.appendChild(lbl);
        container.appendChild(item);

        // Bind mutual exclusivity
        cb.addEventListener("change", () => {
            if (cb.checked) {
                if (listType === "features") {
                    const other = document.getElementById(`cb-excluded-${colName}`);
                    if (other) other.checked = false;
                } else {
                    const other = document.getElementById(`cb-features-${colName}`);
                    if (other) other.checked = false;
                }
            } else {
                // If unchecking one, automatically check the other to ensure column is accounted for
                if (listType === "features") {
                    const other = document.getElementById(`cb-excluded-${colName}`);
                    if (other) other.checked = true;
                } else {
                    const other = document.getElementById(`cb-features-${colName}`);
                    if (other) other.checked = true;
                }
            }
        });
    };

    // 2. Render Checkboxes
    allColumns.forEach(col => {
        if (col === analysis.target) return; // Target cannot be feature or excluded

        const protectedPredictor = isProtectedPredictorColumn(col);
        const isFeature = !protectedPredictor && analysis.features.includes(col);
        renderCheckbox(featuresContainer, col, isFeature, "features");
        renderCheckbox(excludedContainer, col, protectedPredictor || !isFeature, "excluded");
        if (protectedPredictor) {
            const f = document.getElementById(`cb-features-${col}`);
            const e = document.getElementById(`cb-excluded-${col}`);
            if (f) { f.checked = false; f.disabled = true; }
            if (e) { e.checked = true; e.disabled = true; }
        }
    });

    // Listen to Target dropdown change to re-render checkboxes
    targetSelect.onchange = () => {
        const newTarget = targetSelect.value;
        
        // Build mock updated analysis with new target and move old target to features
        const oldTarget = analysis.target;
        let updatedFeatures = [...analysis.features];
        let updatedExcluded = [...analysis.excluded];

        // Remove new target from features or excluded list
        updatedFeatures = updatedFeatures.filter(x => x !== newTarget);
        updatedExcluded = updatedExcluded.filter(x => x !== newTarget);

        // Add old target to features
        updatedFeatures.push(oldTarget);

        analysis.target = newTarget;
        analysis.features = updatedFeatures;
        analysis.excluded = updatedExcluded;

        document.getElementById("ml-intent-target").textContent = newTarget;
        
        // Re-detect problem type heuristically for summary
        let isNum = STATE.ml.summary.numerical.includes(newTarget);
        analysis.problem_type = isNum ? "Regression" : "Classification";
        document.getElementById("ml-intent-problem").textContent = analysis.problem_type;

        renderMLColumnsConfirmation(analysis);
    };

    // Render the recommended 5 algorithms
    renderAlgorithmRecommendations(analysis.problem_type, analysis.candidate_algorithms, analysis.algorithm_reasons);
    
    // Set active style on the default selected radio (Automatic)
    setTimeout(() => {
        const activeRadio = document.querySelector("input[name='feature-selection-mode']:checked");
        const mode = activeRadio ? activeRadio.value : "automatic";
        window.toggleFeatureSelectionMode(mode);
    }, 50);
}

// 3. EXECUTE TRAINING
async function executeMLPipelineTraining() {
    const target = document.getElementById("ml-target-select").value;
    
    // Read features checked
    const features = [];
    document.querySelectorAll("#ml-features-container input[type='checkbox']").forEach(cb => {
        if (cb.checked) {
            features.push(cb.value);
        }
    });

    const excluded = [];
    document.querySelectorAll("#ml-excluded-container input[type='checkbox']").forEach(cb => {
        if (cb.checked) {
            excluded.push(cb.value);
        }
    });

    if (features.length === 0) {
        alert("Please select at least one predictor feature column.");
        return;
    }

    // Read the problem type from state rather than scraping it out of the DOM.
    // The backend validates it against the target and returns the authoritative
    // value in the response.
    const problemType = (STATE.ml.analysis && STATE.ml.analysis.problem_type) ||
        document.getElementById("ml-intent-problem").textContent;

    showMLSubView(3);
    updateMLStepIndicator(3);

    // Start progress checklist animation
    const setStageActive = (stageNum, text) => {
        const item = document.getElementById(`check-ml-${stageNum}`);
        if (item) {
            item.style.opacity = "1";
            item.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin" style="color: var(--color-secondary);"></i> ${text}`;
        }
    };

    const setStageCompleted = (stageNum, text) => {
        const item = document.getElementById(`check-ml-${stageNum}`);
        if (item) {
            item.style.opacity = "1";
            item.innerHTML = `<i class="fa-solid fa-circle-check" style="color: var(--color-success);"></i> ${text}`;
        }
    };

    // Animate stage checklist
    const checkStages = async () => {
        await sleep(400);
        setStageCompleted(1, "Preparing dataset");
        setStageActive(2, "Feature engineering (impute + scale + encode)");
        await sleep(500);
        setStageCompleted(2, "Feature engineering (impute + scale + encode)");
        setStageActive(3, "Feature selection");
        await sleep(400);
        setStageCompleted(3, "Feature selection");
        setStageActive(4, "Training candidate models");
    };

    // Run UI animations in background
    checkStages();

    try {
        const activeRadio = document.querySelector("input[name='feature-selection-mode']:checked");
        const selectionMode = activeRadio ? activeRadio.value : "automatic";

        const response = await fetch(`${API_BASE}/ml/train`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                target,
                features,
                excluded_features: excluded,
                problem_type: problemType,
                feature_selection_mode: selectionMode
            })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Training process failed.");
        }

        const data = await response.json();
        STATE.ml.results = data;

        // Finish animations
        setStageCompleted(4, "Training candidate models");
        setStageActive(5, "Evaluating models on test set");
        await sleep(400);
        setStageCompleted(5, "Evaluating models on test set");
        setStageActive(6, "Selecting best model");
        await sleep(450);
        setStageCompleted(6, "Selecting best model");
        await sleep(300);

        // Render Results Sub-view 4
        renderMLResults(data, target, features, problemType);

    } catch (err) {
        alert(`Error training models: ${err.message}`);
        showMLSubView(2);
        updateMLStepIndicator(2);
    }
}

// 4. RENDER RESULTS
// Formats a metric that may legitimately be null (e.g. ROC-AUC when the model
// exposes no probabilities, or adjusted R2 on too few test rows). Calling
// .toFixed() on those would throw and blank the whole results view.
function fmtMetric(value, digits) {
    if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
    const num = Number(value);
    if (!Number.isFinite(num)) return "n/a";
    return num.toFixed(digits === undefined ? 3 : digits);
}

function fmtQValue(value) {
    if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
    const num = Number(value);
    if (!Number.isFinite(num)) return "n/a";
    if (num === 0) return "0.000";
    if (num < 0.001) {
        return num.toFixed(6).replace(/(\.\d*?[1-9])0+$/, "$1");
    }
    if (num < 0.01) {
        return num.toFixed(4).replace(/(\.\d*?[1-9])0+$/, "$1");
    }
    return num.toFixed(4);
}

function renderMLResults(data, target, features, problemType) {
    // The backend decides the problem type; it may have overridden the request.
    const resolvedType = data.problem_type || problemType;
    const isRegression = resolvedType.toLowerCase() === "regression";
    problemType = resolvedType;

    const problemEl = document.getElementById("ml-intent-problem");
    if (problemEl) problemEl.textContent = resolvedType;
    
    // 1. Best Model Card
    const isReliable = data.best_model && data.best_model.beats_baseline !== false;
    const bestTitleEl = document.getElementById("ml-best-model-title");
    if (isReliable) {
        bestTitleEl.textContent = data.best_model.display_name || data.best_model.best_model;
        bestTitleEl.style.color = "var(--color-success)";
    } else {
        bestTitleEl.textContent = "No reliable model found";
        bestTitleEl.style.color = "var(--color-warning)";
    }
    document.getElementById("ml-best-reason-text").textContent = data.best_model.reason;

    const metricsGrid = document.getElementById("ml-best-metrics-grid");
    metricsGrid.innerHTML = "";
    
    const bestMetrics = data.best_model.metrics;
    
    const addMetricBadge = (label, val, trendIcon) => {
        const card = document.createElement("div");
        card.style.background = "rgba(255,255,255,0.03)";
        card.style.border = "1px solid rgba(255,255,255,0.05)";
        card.style.padding = "0.75rem 1.25rem";
        card.style.borderRadius = "8px";
        card.style.display = "flex";
        card.style.flexDirection = "column";
        card.style.minWidth = "120px";
        
        card.innerHTML = `
            <span style="font-size:0.7rem; text-transform:uppercase; color:var(--text-secondary);">${label} ${trendIcon}</span>
            <span style="font-size:1.5rem; font-weight:700; color:#fff; font-family:var(--font-mono); margin-top:0.25rem;">${val}</span>
        `;
        metricsGrid.appendChild(card);
    };

    if (isRegression) {
        addMetricBadge("MAE", fmtMetric(bestMetrics.mae), "<b>↓</b>");
        addMetricBadge("RMSE", fmtMetric(bestMetrics.rmse), "<b>↓</b>");
        addMetricBadge("R² Score", fmtMetric(bestMetrics.r2_score), "<b>↑</b>");
        addMetricBadge("Adjusted R²", fmtMetric(bestMetrics.adjusted_r2), "<b>↑</b>");
    } else {
        addMetricBadge("Accuracy", fmtMetric(bestMetrics.accuracy), "<b>↑</b>");
        addMetricBadge("Balanced Acc.", fmtMetric(bestMetrics.balanced_accuracy), "<b>↑</b>");
        addMetricBadge("F1 (macro)", fmtMetric(bestMetrics.f1_macro), "<b>↑</b>");
        addMetricBadge("Precision", fmtMetric(bestMetrics.precision), "<b>↑</b>");
        addMetricBadge("Recall", fmtMetric(bestMetrics.recall), "<b>↑</b>");
        addMetricBadge("ROC-AUC", fmtMetric(bestMetrics.roc_auc), "<b>↑</b>");
    }

    // 2. Feature Engineering transforms list
    const transList = document.getElementById("ml-transforms-list");
    transList.innerHTML = "";
    data.feature_engineering.transformations.forEach(t => {
        const li = document.createElement("li");
        li.innerHTML = `<i class="fa-solid fa-circle-check" style="color:var(--color-success); margin-right:0.5rem;"></i> ${t}`;
        transList.appendChild(li);
    });
    document.getElementById("ml-engineered-cols-count").textContent = data.feature_engineering.engineered_cols.length;

    // 3. Feature significance chart
    const significanceList = document.getElementById("ml-feature-importance-list");
    significanceList.innerHTML = "";
    document.getElementById("ml-selection-method-badge").textContent = data.feature_selection.method;

    const importances = data.feature_selection.importances;
    const details = data.feature_selection.details || {};

    // The backend owns the selection decision. This used to re-apply a
    // "score >= 0.05" rule here, so the panel could disagree with the features
    // the models were actually trained on.
    const selectedFeaturesList = data.feature_selection.selected || [];
    const removedFeaturesList = data.feature_selection.removed || [];
    const excludedByPolicy = data.feature_selection.excluded_by_policy || [];

    const methodTextEl = document.getElementById("ml-selection-method-text");
    if (methodTextEl) methodTextEl.textContent = data.feature_selection.method;
    const selTextEl = document.getElementById("ml-selection-selected-text");
    if (selTextEl) selTextEl.textContent = selectedFeaturesList.join(", ") || "None";
    const remTextEl = document.getElementById("ml-selection-removed-text");
    if (remTextEl) remTextEl.textContent = removedFeaturesList.join(", ") || "None";

    const criterionEl = document.getElementById("ml-selection-criterion-text");
    if (criterionEl) criterionEl.textContent = data.feature_selection.criterion || "";

    // Group 1: Selected Features
    const selectedGroupEl = document.getElementById("ml-selected-group-items");
    const selectedBadgeEl = document.getElementById("ml-selected-count-badge");
    if (selectedBadgeEl) selectedBadgeEl.textContent = selectedFeaturesList.length;
    if (selectedGroupEl) {
        if (selectedFeaturesList.length === 0) {
            selectedGroupEl.innerHTML = `<span style="color:var(--text-muted); font-style:italic;">None</span>`;
        } else {
            selectedGroupEl.innerHTML = selectedFeaturesList.map(f => {
                const info = details[f] || {};
                const isMI = info.selection_path === "mi_threshold" || (info.normalized_score !== undefined && info.normalized_score >= 0.3);
                const tagText = isMI ? "MI Threshold" : "Significance";
                const tagBg = isMI ? "rgba(16, 185, 129, 0.15)" : "rgba(59, 130, 246, 0.15)";
                const tagColor = isMI ? "var(--color-success)" : "#60a5fa";
                const tagBorder = isMI ? "rgba(16, 185, 129, 0.3)" : "rgba(59, 130, 246, 0.3)";
                return `<div class="ml-category-item" style="color:var(--color-success); display:flex; align-items:center; justify-content:space-between; gap:0.5rem; padding:0.2rem 0;">
                    <div style="display:flex; align-items:center; gap:0.5rem; min-width:0;">
                        <span style="font-weight:700; flex:0 0 auto; width:14px; text-align:center;">✓</span>
                        <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escapeHTML(f)}</span>
                    </div>
                    <span style="font-size:0.68rem; font-family:var(--font-sans); padding:0.1rem 0.4rem; border-radius:3px; background:${tagBg}; color:${tagColor}; border:1px solid ${tagBorder}; flex:0 0 auto; font-weight:600;">${tagText}</span>
                </div>`;
            }).join("");
        }
    }

    // Group 2: Removed Features
    const removedGroupEl = document.getElementById("ml-removed-group-items");
    const removedBadgeEl = document.getElementById("ml-removed-count-badge");
    if (removedBadgeEl) removedBadgeEl.textContent = removedFeaturesList.length;
    if (removedGroupEl) {
        if (removedFeaturesList.length === 0) {
            removedGroupEl.innerHTML = `<span style="color:var(--text-muted); font-style:italic;">None</span>`;
        } else {
            removedGroupEl.innerHTML = removedFeaturesList.map(f =>
                `<div class="ml-category-item" style="color:var(--color-danger); display:flex; align-items:center; gap:0.5rem;">
                    <span style="font-weight:700; flex:0 0 auto; width:14px; text-align:center;">✗</span>
                    <span style="flex:0 0 auto;">${escapeHTML(f)}</span>
                </div>`
            ).join("");
        }
    }

    // Group 3: Excluded by Policy / Identifier
    const excludedGroupEl = document.getElementById("ml-excluded-group-items");
    const excludedBadgeEl = document.getElementById("ml-excluded-count-badge");
    if (excludedBadgeEl) excludedBadgeEl.textContent = excludedByPolicy.length;
    if (excludedGroupEl) {
        if (excludedByPolicy.length === 0) {
            excludedGroupEl.innerHTML = `<span style="color:var(--text-muted); font-style:italic;">None</span>`;
        } else {
            excludedGroupEl.innerHTML = excludedByPolicy.map(item =>
                `<div style="display:flex; flex-direction:column; gap:0.15rem; padding:0.25rem 0; border-bottom:1px solid rgba(255,255,255,0.03);">
                    <div class="ml-category-item" style="color:var(--color-warning); font-family:var(--font-mono); font-weight:600; display:flex; align-items:center; gap:0.5rem;">
                        <span style="flex:0 0 auto; width:14px; text-align:center;">⊘</span>
                        <span style="flex:0 0 auto;">${escapeHTML(item.feature)}</span>
                    </div>
                    <div style="font-size:0.75rem; color:var(--text-muted); margin-left:1.35rem;">
                        Reason: <span style="color:var(--text-secondary);">${escapeHTML(item.reason)}</span>
                    </div>
                </div>`
            ).join("");
        }
    }

    // Validation Check: Verify every original non-target feature appears exactly once in either Selected or Removed,
    // while every intentionally excluded feature appears in Excluded by policy/identifier.
    const allOriginalColumns = data.feature_selection.all_columns ||
        (STATE.ml.summary && STATE.ml.summary.columns) ||
        [...STATE.ml.summary.numerical, ...STATE.ml.summary.categorical, ...STATE.ml.summary.identifier, ...STATE.ml.summary.date, ...STATE.ml.summary.text];

    const nonTargetOriginal = allOriginalColumns.filter(c => c !== target);
    const selectedSet = new Set(selectedFeaturesList);
    const removedSet = new Set(removedFeaturesList);
    const excludedMap = new Map();
    excludedByPolicy.forEach(item => excludedMap.set(item.feature, item.reason));

    const validationIssues = [];
    selectedFeaturesList.forEach(f => {
        if (removedSet.has(f)) validationIssues.push(`"${f}" appears in both Selected and Removed`);
        if (excludedMap.has(f)) validationIssues.push(`"${f}" appears in both Selected and Excluded`);
    });
    removedFeaturesList.forEach(f => {
        if (excludedMap.has(f)) validationIssues.push(`"${f}" appears in both Removed and Excluded`);
    });
    nonTargetOriginal.forEach(c => {
        const count = (selectedSet.has(c) ? 1 : 0) + (removedSet.has(c) ? 1 : 0) + (excludedMap.has(c) ? 1 : 0);
        if (count === 0) validationIssues.push(`"${c}" dropped silently`);
        if (count > 1) validationIssues.push(`"${c}" counted multiple times`);
    });

    const isAccountingValid = validationIssues.length === 0;
    const validationSummaryText = isAccountingValid
        ? `Validated: All ${nonTargetOriginal.length} original predictor columns accounted for (${selectedFeaturesList.length} Selected, ${removedFeaturesList.length} Removed, ${excludedByPolicy.length} Excluded by policy/identifier). Target: ${target}.`
        : `Traceability Warning: ${validationIssues.join("; ")}`;

    const valBox = document.getElementById("ml-feature-validation-box");
    const valMsg = document.getElementById("ml-feature-validation-msg");
    if (valBox && valMsg) {
        valMsg.textContent = validationSummaryText;
        if (isAccountingValid) {
            valBox.style.color = "var(--color-success)";
            valBox.style.borderColor = "rgba(16, 185, 129, 0.2)";
            valBox.style.background = "rgba(16, 185, 129, 0.08)";
        } else {
            valBox.style.color = "var(--color-warning)";
            valBox.style.borderColor = "rgba(245, 158, 11, 0.2)";
            valBox.style.background = "rgba(245, 158, 11, 0.08)";
        }
    }

    // Surface any feature that looks like a restatement of the target.
    const leakEl = document.getElementById("ml-leakage-warnings");
    if (leakEl) {
        const warnings = data.feature_selection.leakage_warnings || [];
        if (warnings.length) {
            leakEl.style.display = "block";
            leakEl.innerHTML = warnings.map(w =>
                `<div><i class="fa-solid fa-triangle-exclamation"></i> <b>${escapeHTML(w.feature)}</b> ${escapeHTML(w.reason)}</div>`
            ).join("");
        } else {
            leakEl.style.display = "none";
        }
    }
    if (Object.keys(importances).length === 0) {
        significanceList.innerHTML = `<div style="font-style:italic; color:var(--text-muted); font-size:0.85rem;">No features calculated.</div>`;
    } else {
        for (const [feat, val] of Object.entries(importances)) {
            const info = details[feat] || {};
            const isKept = selectedFeaturesList.includes(feat);
            const card = document.createElement("div");
            card.className = `ml-feature-card ${isKept ? "is-selected" : "is-removed"}`;
            const normalized = info.normalized_score !== undefined ? info.normalized_score : val;
            const mi = info.mutual_information !== undefined ? info.mutual_information : null;
            const q = info.q_value !== undefined ? info.q_value : null;
            // Distinguish *why* a feature was kept: a feature that only cleared
            // the FDR significance test (e.g. GPA, whose MI share is below the
            // 30% threshold) previously showed the same generic "Selected"
            // badge as one that cleared the MI-share rule, so the two looked
            // like they were kept for the same reason.
            // Follow the user-facing rule:
            // Case 1 — MI threshold passed: "Selected by MI Threshold"
            // Case 2 — MI threshold failed but FDR q < 0.05: "Selected by Significance"
            // Case 3 — Neither condition passed: "Removed"
            let selectionPath = info.selection_path;
            if (!selectionPath) {
                if (isKept) {
                    selectionPath = (info.meets_mi_floor || (normalized !== null && normalized >= 0.3)) ? "mi_threshold" : "significance";
                } else {
                    selectionPath = "not_selected";
                }
            }
            const pathLabels = {
                significance: "Selected by Significance",
                mi_threshold: "Selected by MI Threshold",
                minimum_set: "Selected (Minimum Set)",
            };
            const status = isKept ? (pathLabels[selectionPath] || "Selected by MI Threshold") : "Removed";
            const statusClass = isKept ? (selectionPath === "significance" ? "selected significance" : "selected") : "removed";
            const reason = info.reason || (isKept ? (selectionPath === "mi_threshold" ? "Passed the MI threshold." : "Selected by FDR statistical significance.") : "Did not meet the MI threshold and not statistically significant after FDR correction.");
            card.style.cssText = `
                background: ${isKept ? "rgba(16, 185, 129, 0.03)" : "rgba(239, 68, 68, 0.02)"};
                border: 1px solid ${isKept ? "rgba(16, 185, 129, 0.2)" : "rgba(239, 68, 68, 0.15)"};
                border-radius: 8px;
                padding: 1rem 1.15rem;
                display: flex;
                flex-direction: column;
                gap: 0.25rem;
            `;
            card.innerHTML = `
                <div class="ml-feature-card-head" style="display:flex; flex-direction:column; align-items:flex-start; gap:0.4rem; margin-bottom:0.75rem;">
                    <div class="ml-feature-badge-wrap">
                        <span class="ml-feature-status ${statusClass}" style="display:inline-flex; align-items:center; gap:0.4rem; padding:0.25rem 0.65rem; border-radius:4px; font-size:0.75rem; font-weight:700; letter-spacing:0.02em; line-height:1.2; background:${isKept ? (selectionPath === 'significance' ? 'rgba(59, 130, 246, 0.15)' : 'rgba(16, 185, 129, 0.15)') : 'rgba(239, 68, 68, 0.15)'}; color:${isKept ? (selectionPath === 'significance' ? '#60a5fa' : 'var(--color-success)') : 'var(--color-danger)'}; border:1px solid ${isKept ? (selectionPath === 'significance' ? 'rgba(59, 130, 246, 0.35)' : 'rgba(16, 185, 129, 0.35)') : 'rgba(239, 68, 68, 0.35)'};">
                            <i class="fa-solid ${isKept ? 'fa-check' : 'fa-xmark'}"></i> ${status}
                        </span>
                    </div>
                    <div class="ml-feature-card-name" style="font-size:1.05rem; font-weight:700; font-family:var(--font-mono); color:var(--text-primary); word-break:normal; overflow-wrap:break-word; line-height:1.4;" title="${escapeHTML(feat)}">${escapeHTML(feat)}</div>
                </div>
                <div class="ml-feature-metrics" style="display:grid; grid-template-columns:repeat(auto-fit, minmax(100px, 1fr)); gap:0.75rem; background:rgba(0,0,0,0.25); border:1px solid rgba(255,255,255,0.04); border-radius:6px; padding:0.65rem 0.85rem; margin-bottom:0.75rem;">
                    <div class="ml-metric-col" style="display:flex; flex-direction:column; gap:0.25rem;">
                        <span class="ml-metric-lbl" style="font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em; font-weight:600;">Normalized</span>
                        <strong class="ml-metric-val" style="font-size:0.95rem; font-family:var(--font-mono); font-weight:700; color:var(--text-primary);">${fmtMetric(normalized)}</strong>
                    </div>
                    <div class="ml-metric-col" style="display:flex; flex-direction:column; gap:0.25rem;">
                        <span class="ml-metric-lbl" style="font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em; font-weight:600;">MI Score</span>
                        <strong class="ml-metric-val" style="font-size:0.95rem; font-family:var(--font-mono); font-weight:700; color:var(--text-primary);">${mi === null ? 'n/a' : fmtMetric(mi)}</strong>
                    </div>
                    <div class="ml-metric-col" style="display:flex; flex-direction:column; gap:0.25rem;">
                        <span class="ml-metric-lbl" style="font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em; font-weight:600;">FDR q-value</span>
                        <strong class="ml-metric-val" style="font-size:0.95rem; font-family:var(--font-mono); font-weight:700; color:var(--text-primary);">${q === null ? 'n/a' : fmtQValue(q)}</strong>
                    </div>
                </div>
                <div class="ml-feature-reason" style="display:flex; flex-direction:column; gap:0.3rem; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.03); border-radius:6px; padding:0.6rem 0.85rem;">
                    <span class="ml-reason-lbl" style="font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em; font-weight:600;">Selection Rule</span>
                    <p class="ml-reason-text" style="margin:0; font-size:0.82rem; color:var(--text-secondary); line-height:1.5; word-break:normal; overflow-wrap:break-word;">${escapeHTML(reason)}</p>
                </div>
            `;
            significanceList.appendChild(card);
        }
    }

    // 4. Comparison Table
    const compTable = document.getElementById("ml-comparison-table");
    const thead = compTable.querySelector("thead");
    const tbody = document.getElementById("ml-comparison-tbody");
    const cvResults = data.cross_validation || {};

    tbody.innerHTML = "";

    // Both the table's CV column and the baseline row must read from the same
    // value the recommendation text uses (baseline.cv.mean, computed once in
    // the backend) - previously this cell was hardcoded to "n/a" for the
    // baseline row even though the backend already returns its CV score.
    const cvCellFromStats = (cv) => {
        if (!cv || cv.mean === undefined || cv.mean === null) return `<td style="text-align:right; font-family:var(--font-mono);">n/a</td>`;
        return `<td style="text-align:right; font-family:var(--font-mono);">${fmtMetric(cv.mean)} <span style="color:var(--text-muted); font-size:0.8em;">± ${fmtMetric(cv.std)}</span></td>`;
    };
    const cvCell = (name) => cvCellFromStats(cvResults[name]);

    const modelRow = (modelName, m, cols, opts) => {
        const options = opts || {};
        const tr = document.createElement("tr");
        if (options.isBest) {
            tr.style.background = "rgba(16, 185, 129, 0.06)";
            tr.style.fontWeight = "bold";
        }
        if (options.isBaseline) {
            tr.style.opacity = "0.75";
            tr.style.fontStyle = "italic";
        }
        const isReliableWinner = options.isBest && (data.best_model && data.best_model.beats_baseline !== false);
        const badge = isReliableWinner
            ? ' <span class="rec-badge active" style="font-size:0.6rem; margin-left:0.25rem;"><i class="fa-solid fa-trophy"></i> Best</span>'
            : (options.isBaseline
                ? ' <span class="rec-badge" style="font-size:0.6rem; margin-left:0.25rem; background:rgba(255,255,255,0.05); color:var(--text-muted);">reference</span>'
                : (options.isBest ? ' <span class="rec-badge" style="font-size:0.6rem; margin-left:0.25rem; background:rgba(245,158,11,0.15); color:var(--color-warning);">top nominal</span>' : ''));
        tr.innerHTML = `<td>${escapeHTML(modelName)}${badge}</td>` +
            (options.isBaseline ? cvCellFromStats(m.cv) : cvCell(modelName)) +
            cols.map(v => `<td style="text-align:right; font-family:var(--font-mono);">${v}</td>`).join("");
        return tr;
    };

    const cvHeader = `<th style="text-align: right;">CV score ↑</th>`;

    if (isRegression) {
        thead.innerHTML = `
            <tr>
                <th>Candidate Model</th>
                ${cvHeader}
                <th style="text-align: right;">MAE ↓</th>
                <th style="text-align: right;">MSE ↓</th>
                <th style="text-align: right;">RMSE ↓</th>
                <th style="text-align: right;">R² Score ↑</th>
                <th style="text-align: right;">Adjusted R² ↑</th>
            </tr>
        `;
        for (const [modelName, m] of Object.entries(data.evaluations)) {
            const isBest = modelName === data.best_model.best_model;
            tbody.appendChild(modelRow(modelName, m, [
                fmtMetric(m.mae), fmtMetric(m.mse), fmtMetric(m.rmse),
                fmtMetric(m.r2_score), fmtMetric(m.adjusted_r2)
            ], { isBest }));
        }
        if (data.baseline && Object.keys(data.baseline).length) {
            const b = data.baseline;
            tbody.appendChild(modelRow(`Baseline (${b.strategy || "mean"})`, b, [
                fmtMetric(b.mae), fmtMetric(b.mse), fmtMetric(b.rmse),
                fmtMetric(b.r2_score), fmtMetric(b.adjusted_r2)
            ], { isBaseline: true }));
        }
    } else {
        thead.innerHTML = `
            <tr>
                <th>Candidate Model</th>
                ${cvHeader}
                <th style="text-align: right;">Accuracy ↑</th>
                <th style="text-align: right;">Balanced Acc. ↑</th>
                <th style="text-align: right;">Precision (macro) ↑</th>
                <th style="text-align: right;">Recall (macro) ↑</th>
                <th style="text-align: right;">F1 (macro) ↑</th>
                <th style="text-align: right;">ROC-AUC ↑</th>
            </tr>
        `;
        for (const [modelName, m] of Object.entries(data.evaluations)) {
            const isBest = modelName === data.best_model.best_model;
            tbody.appendChild(modelRow(modelName, m, [
                fmtMetric(m.accuracy), fmtMetric(m.balanced_accuracy),
                fmtMetric(m.precision_macro), fmtMetric(m.recall_macro),
                fmtMetric(m.f1_macro), fmtMetric(m.roc_auc)
            ], { isBest }));
        }
        if (data.baseline && Object.keys(data.baseline).length) {
            const b = data.baseline;
            tbody.appendChild(modelRow(`Baseline (${b.strategy || "majority class"})`, b, [
                fmtMetric(b.accuracy), fmtMetric(b.balanced_accuracy),
                fmtMetric(b.precision_macro), fmtMetric(b.recall_macro),
                fmtMetric(b.f1_macro), fmtMetric(b.roc_auc)
            ], { isBaseline: true }));
        }
    }

    // 4.5. Render Model Performance Comparison Chart
    //
    // This used to show a "suitability rating" in stars derived from hardcoded
    // row-count thresholds in this file, with numFeatures actually holding the
    // dataset's total column count. Those stars had no connection to the data or
    // to the results, so they have been replaced with measured evidence: the
    // cross-validated score with its spread across folds.
    const chartList = document.getElementById("ml-model-comparison-chart");
    const chartDesc = document.getElementById("ml-chart-description");
    chartList.innerHTML = "";
    chartList.style.display = "grid";
    chartList.style.gridTemplateColumns = "repeat(auto-fit, minmax(220px, 1fr))";
    chartList.style.gap = "1rem";

    const cvMetricName = isRegression ? "R²" : "F1 (macro)";
    chartDesc.innerHTML = `Cross-validated <b>${cvMetricName}</b> on the training split (mean ± standard deviation across folds), with the held-out test score for comparison. Models whose intervals overlap are not distinguishable on this dataset.`;

    const tiedModels = (data.best_model && data.best_model.tied_models) || [];
    // Same value the recommendation text and comparison table use: the
    // baseline's cross-validated mean when available, falling back to the
    // held-out score only if CV wasn't possible. Previously this always read
    // the held-out score, so the visualizer showed a different number than
    // the recommendation (which compares on CV scores whenever it can).
    const baselineCv = data.baseline ? data.baseline.cv : null;
    const baselineScore = (baselineCv && baselineCv.mean !== undefined && baselineCv.mean !== null)
        ? baselineCv.mean
        : (data.baseline ? (isRegression ? data.baseline.r2_score : data.baseline.f1_macro) : null);

    Object.keys(data.evaluations).forEach(algo => {
        const m = data.evaluations[algo];
        const cv = cvResults[algo];
        const isBest = algo === data.best_model.best_model;
        const isTied = tiedModels.indexOf(algo) !== -1 && !isBest;

        const holdout = isRegression ? m.r2_score : m.f1_macro;
        const primary = (cv && cv.mean !== undefined && cv.mean !== null) ? cv.mean : holdout;
        const pct = (primary === null || primary === undefined)
            ? 0 : Math.max(0, Math.min(100, Math.round(primary * 100)));

        const card = document.createElement("div");
        card.style.background = isBest ? "rgba(16, 185, 129, 0.05)" : "rgba(0,0,0,0.2)";
        card.style.border = isBest ? "1px solid var(--color-success)" : "1px solid rgba(255,255,255,0.05)";
        card.style.borderRadius = "8px";
        card.style.padding = "0.75rem 1rem";
        card.style.display = "flex";
        card.style.flexDirection = "column";
        card.style.gap = "0.25rem";
        card.style.position = "relative";

        if (isBest && (data.best_model && data.best_model.beats_baseline !== false)) {
            const bestBadge = document.createElement("div");
            bestBadge.style.cssText = "position:absolute; top:-10px; right:10px; background:var(--color-success); color:#000; font-size:0.6rem; font-weight:800; padding:0.15rem 0.5rem; border-radius:4px; text-transform:uppercase; letter-spacing:0.05em; display:flex; align-items:center; gap:0.25rem;";
            bestBadge.innerHTML = `<i class="fa-solid fa-trophy"></i> Recommended`;
            card.appendChild(bestBadge);
        } else if (isBest) {
            const warnBadge = document.createElement("div");
            warnBadge.style.cssText = "position:absolute; top:-10px; right:10px; background:var(--color-warning); color:#000; font-size:0.6rem; font-weight:800; padding:0.15rem 0.5rem; border-radius:4px; text-transform:uppercase; letter-spacing:0.05em; display:flex; align-items:center; gap:0.25rem;";
            warnBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Baseline Not Beaten`;
            card.appendChild(warnBadge);
        }

        const spread = (cv && cv.std !== undefined && cv.std !== null)
            ? `<span style="font-size:0.7rem; color:var(--text-muted);">± ${fmtMetric(cv.std)} over ${cv.folds} folds</span>`
            : `<span style="font-size:0.7rem; color:var(--text-muted);">single split (no cross-validation)</span>`;

        const content = document.createElement("div");
        content.style.cssText = "display:flex; flex-direction:column; gap:0.25rem;";
        content.innerHTML = `
            <span style="font-weight:700; font-size:0.95rem; color:${isBest ? 'var(--color-success)' : 'var(--text-primary)'};">${escapeHTML(algo)}</span>
            ${isTied ? '<span style="font-size:0.65rem; color:var(--color-warning);">CV score range overlaps with recommendation</span>' : ''}
            <div style="display:flex; align-items:baseline; gap:0.35rem; margin-top:0.25rem; border-top:1px solid rgba(255,255,255,0.03); padding-top:0.5rem;">
                <span style="font-weight:700; font-size:1.25rem; font-family:var(--font-mono); color:${isBest ? 'var(--color-success)' : 'var(--text-primary)'};">${fmtMetric(primary)}</span>
                <span style="font-size:0.7rem; color:var(--text-muted);">${cvMetricName}</span>
            </div>
            ${spread}
            <div class="ml-feature-bar-container" style="margin-top:0.25rem; height:4px; background:rgba(255,255,255,0.05); border-radius:2px;">
                <div class="ml-feature-bar-fill" style="width:${pct}%; height:100%; border-radius:2px; background:${isBest ? 'linear-gradient(90deg, var(--color-success), #34d399)' : 'linear-gradient(90deg, var(--color-secondary), var(--color-primary))'}"></div>
            </div>
            <span style="font-size:0.68rem; color:var(--text-muted);">held-out test: ${fmtMetric(holdout)}</span>
        `;
        card.appendChild(content);
        chartList.appendChild(card);
    });

    // Baseline and honesty notices
    const noticeEl = document.getElementById("ml-result-notices");
    if (noticeEl) {
        const notices = [];
        if (baselineScore !== null && baselineScore !== undefined) {
            notices.push(`Trivial baseline (${escapeHTML(data.baseline.strategy || "reference")}) scores <b>${fmtMetric(baselineScore)}</b> ${cvMetricName}. Any model must clearly beat this to be useful.`);
        }
        if (data.best_model && data.best_model.beats_baseline === false) {
            notices.push(`<b>The recommended model does not meaningfully beat that baseline.</b> The selected features carry little usable signal about this target.`);
        }
        (data.best_model && data.best_model.warnings || []).forEach(w => notices.push(escapeHTML(w)));
        ((data.training_info && data.training_info.notes) || []).forEach(w => notices.push(escapeHTML(w)));
        if (data.split) {
            notices.push(`Split: ${data.split.train_rows} training rows / ${data.split.test_rows} test rows${data.split.stratified ? ", stratified by class" : ""}. Preprocessing was fitted on the training rows only.`);
        }
        if (data.problem_type_reason) notices.push(escapeHTML(data.problem_type_reason));

        if (notices.length) {
            noticeEl.style.display = "block";
            noticeEl.innerHTML = notices.map(n => `<div><i class="fa-solid fa-circle-info"></i> ${n}</div>`).join("");
        } else {
            noticeEl.style.display = "none";
        }
    }

    // 5. Confusion Matrix (Classification only)
    const cmCard = document.getElementById("ml-confusion-matrix-card");
    if (!isRegression && bestMetrics.confusion_matrix && bestMetrics.confusion_matrix.tp !== undefined) {
        cmCard.style.display = "block";
        const cm = bestMetrics.confusion_matrix;
        document.querySelector("#cm-tp .cm-val").textContent = cm.tp;
        document.querySelector("#cm-tn .cm-val").textContent = cm.tn;
        document.querySelector("#cm-fp .cm-val").textContent = cm.fp;
        document.querySelector("#cm-fn .cm-val").textContent = cm.fn;
        // Name the classes: "positive" is the minority class chosen by the backend,
        // not whichever label happened to sort second.
        const posEl = document.getElementById("cm-positive-label");
        const negEl = document.getElementById("cm-negative-label");
        if (posEl) posEl.textContent = cm.positive_class || "positive";
        if (negEl) negEl.textContent = cm.negative_class || "negative";
    } else {
        cmCard.style.display = "none";
    }

    // 6. Summary Report Code block
    document.getElementById("summary-report-dataset").textContent = STATE.ml.summary.dataset_name;
    document.getElementById("summary-report-objective").textContent = STATE.ml.activeQuestion;
    document.getElementById("summary-report-problem").textContent = problemType;
    document.getElementById("summary-report-target").textContent = target;
    document.getElementById("summary-report-features").textContent = selectedFeaturesList.join(", ") || "None";

    const summaryRemovedEl = document.getElementById("summary-report-removed");
    if (summaryRemovedEl) {
        summaryRemovedEl.textContent = removedFeaturesList.join(", ") || "None";
    }

    const summaryExcludedEl = document.getElementById("summary-report-excluded");
    if (summaryExcludedEl) {
        if (excludedByPolicy.length === 0) {
            summaryExcludedEl.textContent = "None";
        } else {
            summaryExcludedEl.innerHTML = excludedByPolicy.map(item =>
                `<div><b>${escapeHTML(item.feature)}</b> &mdash; <span style="color:var(--text-secondary);">${escapeHTML(item.reason)}</span></div>`
            ).join("");
        }
    }

    const summaryValidationEl = document.getElementById("summary-report-validation");
    if (summaryValidationEl) {
        summaryValidationEl.textContent = validationSummaryText;
    }

    document.getElementById("summary-report-models").textContent = data.models_trained.join(", ");
    const isModelReliable = data.best_model && data.best_model.beats_baseline !== false;
    const finalBest = isModelReliable
        ? (data.best_model.best_model || "-")
        : "No reliable model found";

    document.getElementById("summary-report-best").textContent = finalBest;
    document.getElementById("summary-report-recommendation").textContent = data.best_model.reason ||
        `Recommended Model: ${finalBest}.`;
    const basisEl = document.getElementById("summary-report-basis");
    if (basisEl) {
        basisEl.textContent = data.best_model.ranking_basis
            ? `Ranked on ${data.best_model.ranking_basis}` + (!isModelReliable ? " - does NOT beat trivial baseline" : " - successfully outperforms baseline")
            : "-";
    }

    showMLSubView(4);
    updateMLStepIndicator(8);
}

// Lists the candidate estimators that will be trained.
function renderAlgorithmRecommendations(problemType, candidateAlgorithms, algorithmReasons) {
    const listContainer = document.getElementById("ml-algo-recommendations-list");
    const panel = document.getElementById("ml-recommendations-panel");
    if (!listContainer || !panel) return;

    listContainer.innerHTML = "";
    panel.style.display = "block";

    const isRegression = String(problemType).toLowerCase() === "regression";
    
    // Default fallback descriptions if not provided by backend
    const defaultBlurbs = isRegression
        ? {
            "Linear Regression": "Ordinary Least Squares baseline; directly interpretable coefficient slopes.",
            "Ridge Regression": "L2 regularisation; stabilizes collinear predictors and prevents coefficient explosion.",
            "Lasso Regression": "L1 regularisation; promotes feature sparsity by zeroing out irrelevant weights.",
            "Decision Tree": "Step-wise rule splits without assuming linearity in target relationships.",
            "Random Forest": "High predictive stability by averaging de-correlated decision trees.",
            "Gradient Boosting": "Sequential gradient boosting that iteratively minimizes prediction residuals.",
            "SVR": "Margin-based regression with epsilon-insensitive loss, robust to outlier targets.",
            "KNN": "Non-parametric instance-based regression baseline averaging local neighbors."
          }
        : {
            "Logistic Regression": "Interpretable linear baseline with balanced class weighting to prevent majority-class bias.",
            "Decision Tree": "Transparent rule-based partitioning and invariance to monotonic feature scaling.",
            "Random Forest": "Robust ensemble averaging across non-linear interactions and resistance to overfitting.",
            "Gradient Boosting": "Sequential boosting to minimize residual classification errors on structured tabular data.",
            "SVM": "Maximum-margin separation with kernel projection, well-suited to small-to-medium sample sizes.",
            "KNN": "Instance-based non-parametric baseline sensitive to local proximity.",
            "Gaussian Naive Bayes": "Fast probabilistic baseline with low variance, effective for smaller datasets.",
            "Extra Trees": "Extremely randomized forest ensemble to reduce variance on noisy or correlated features."
          };

    const algos = (candidateAlgorithms && candidateAlgorithms.length > 0)
        ? candidateAlgorithms
        : (isRegression
            ? ["Linear Regression", "Ridge Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
            : ["Logistic Regression", "Decision Tree", "Random Forest", "KNN", "SVM"]);

    const reasons = algorithmReasons || defaultBlurbs;

    algos.forEach(algo => {
        const blurb = reasons[algo] || defaultBlurbs[algo] || "Suitability measured via cross-validation.";
        const card = document.createElement("div");
        card.style.cssText = "background:rgba(0,0,0,0.25); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:0.75rem 1rem; display:flex; flex-direction:column; gap:0.3rem;";
        card.innerHTML = `
            <span style="font-weight:600; font-size:0.85rem; color:var(--text-primary);">${escapeHTML(algo)}</span>
            <span style="font-size:0.72rem; color:var(--text-muted); line-height:1.35;">${escapeHTML(blurb)}</span>
        `;
        listContainer.appendChild(card);
    });
}

// Window global helper for switching Feature Selection Mode
window.toggleFeatureSelectionMode = function(mode) {
    const autoLabel = document.getElementById("label-selection-auto");
    const manualLabel = document.getElementById("label-selection-manual");
    const featuresContainer = document.getElementById("ml-features-container");
    const excludedContainer = document.getElementById("ml-excluded-container");
    
    if (!autoLabel || !manualLabel || !featuresContainer || !excludedContainer) return;
    
    if (mode === "automatic") {
        autoLabel.style.border = "1px solid var(--color-primary)";
        autoLabel.style.background = "rgba(0,0,0,0.25)";
        manualLabel.style.border = "1px solid rgba(255,255,255,0.05)";
        manualLabel.style.background = "rgba(0,0,0,0.1)";
        
        featuresContainer.querySelectorAll("input[type='checkbox']").forEach(cb => cb.disabled = true);
        excludedContainer.querySelectorAll("input[type='checkbox']").forEach(cb => cb.disabled = true);
    } else {
        manualLabel.style.border = "1px solid var(--color-primary)";
        manualLabel.style.background = "rgba(0,0,0,0.25)";
        autoLabel.style.border = "1px solid rgba(255,255,255,0.05)";
        autoLabel.style.background = "rgba(0,0,0,0.1)";
        
        featuresContainer.querySelectorAll("input[type='checkbox']").forEach(cb => cb.disabled = false);
        excludedContainer.querySelectorAll("input[type='checkbox']").forEach(cb => cb.disabled = false);
    }
};
