const form = document.querySelector("#prediction-form");
const fieldGroups = document.querySelector("#field-groups");
const submitButton = document.querySelector("#submit-button");
const formError = document.querySelector("#form-error");
const toast = document.querySelector("#toast");
const resultState = document.querySelector("#result-state");
const inputSummary = document.querySelector("#input-summary");
let dashboardInfo = null;
let toastTimeout = null;

const GROUPS = [
  { title: "PATIENT PROFILE", fields: ["age", "sex", "smoking"] },
  { title: "MEDICAL HISTORY", fields: ["anaemia", "diabetes", "high_blood_pressure"] },
  { title: "CLINICAL MEASUREMENTS", fields: ["creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium", "time"] },
];

const formatNumber = (value, digits = 1) =>
  Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => toast.classList.remove("show"), 5500);
}

function setFormError(message) {
  formError.textContent = message;
  formError.hidden = !message;
}

function getFeature(name) {
  return dashboardInfo?.features?.find((item) => item.name === name);
}

function createField(feature) {
  const field = document.createElement("div");
  field.className = "field";
  const label = document.createElement("label");
  label.htmlFor = `field-${feature.name}`;
  label.append(document.createTextNode(feature.label));
  const unit = document.createElement("small");
  unit.textContent = feature.unit;
  label.append(unit);
  field.append(label);

  let input;
  if (feature.binary) {
    input = document.createElement("select");
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select…";
    input.append(placeholder);
    for (const option of feature.options) {
      const item = document.createElement("option");
      item.value = String(option.value);
      item.textContent = option.label;
      input.append(item);
    }
  } else {
    input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.min = String(feature.min);
    input.max = String(feature.max);
    input.placeholder = `${feature.min} – ${feature.max}`;
    input.inputMode = "decimal";
  }
  input.id = `field-${feature.name}`;
  input.name = feature.name;
  input.required = true;
  input.setAttribute("aria-describedby", `hint-${feature.name}`);
  input.title = feature.description;
  field.append(input);

  const hint = document.createElement("small");
  hint.className = "field-hint";
  hint.id = `hint-${feature.name}`;
  hint.textContent = `Observed training range: ${feature.min}–${feature.max}`;
  field.append(hint);
  return field;
}

function renderFields() {
  for (const group of GROUPS) {
    const section = document.createElement("section");
    section.className = "field-group";
    const title = document.createElement("h3");
    title.className = "field-group-heading";
    title.textContent = group.title;
    section.append(title);
    const grid = document.createElement("div");
    grid.className = "field-grid";
    for (const name of group.fields) {
      const feature = getFeature(name);
      if (feature) grid.append(createField(feature));
    }
    section.append(grid);
    fieldGroups.append(section);
  }
}

function readInputs() {
  const values = {};
  for (const feature of dashboardInfo.features) {
    const input = form.elements.namedItem(feature.name);
    input.setAttribute("aria-invalid", "false");
    const raw = input.value.trim();
    if (!raw) {
      input.setAttribute("aria-invalid", "true");
      input.focus();
      throw new Error(`${feature.label} is required.`);
    }
    const value = Number(raw);
    if (!Number.isFinite(value)) {
      input.setAttribute("aria-invalid", "true");
      input.focus();
      throw new Error(`${feature.label} must be a valid number.`);
    }
    if (value < feature.min || value > feature.max) {
      input.setAttribute("aria-invalid", "true");
      input.focus();
      throw new Error(`${feature.label} must be within the observed training range (${feature.min}–${feature.max} ${feature.unit}).`);
    }
    if (feature.binary && value !== 0 && value !== 1) {
      input.setAttribute("aria-invalid", "true");
      input.focus();
      throw new Error(`${feature.label} must be 0 or 1.`);
    }
    values[feature.name] = value;
  }
  return values;
}

function renderSummary(values) {
  inputSummary.replaceChildren();
  inputSummary.classList.remove("empty-summary");
  for (const feature of dashboardInfo.features) {
    const item = document.createElement("div");
    item.className = "summary-value";
    const label = document.createElement("span");
    label.textContent = feature.label;
    const value = document.createElement("b");
    value.textContent = `${values[feature.name]} ${feature.unit}`;
    item.append(label, value);
    inputSummary.append(item);
  }
}

function renderLocalSensitivity(items) {
  const container = document.querySelector("#local-signals");
  container.replaceChildren();
  if (!items?.length) {
    container.innerHTML = '<p class="muted">No local sensitivity values available.</p>';
    return;
  }
  const maximum = Math.max(...items.map((item) => Math.abs(item.delta)), 0.0001);
  for (const item of items.slice(0, 6)) {
    const row = document.createElement("div");
    row.className = "signal-row";
    const label = document.createElement("span");
    label.className = "signal-label";
    label.textContent = item.feature;
    const track = document.createElement("span");
    track.className = "signal-track";
    const bar = document.createElement("span");
    bar.className = `signal-bar ${item.delta >= 0 ? "positive" : ""}`;
    bar.style.display = "block";
    bar.style.width = `${Math.max(3, Math.abs(item.delta) / maximum * 100)}%`;
    track.append(bar);
    const value = document.createElement("span");
    value.className = "signal-value";
    value.textContent = `${item.delta >= 0 ? "+" : ""}${(item.delta * 100).toFixed(1)} pp`;
    row.append(label, track, value);
    container.append(row);
  }
}

function renderPrediction(result) {
  document.querySelector("#empty-result").hidden = true;
  document.querySelector("#prediction-result").hidden = false;
  const positive = result.positive_probability;
  const negative = result.negative_probability;
  const predictedPositive = result.predicted_class === 1;
  const pill = document.querySelector("#class-pill");
  pill.textContent = `CLASS ${result.predicted_class}`;
  pill.classList.toggle("positive", predictedPositive);
  document.querySelector("#class-label").textContent = result.class_label;
  document.querySelector("#positive-probability").textContent = `${(positive * 100).toFixed(1)}%`;
  document.querySelector("#negative-probability").textContent = `${(negative * 100).toFixed(1)}%`;
  document.querySelector("#gauge-percent").textContent = `${(positive * 100).toFixed(1)}%`;
  const gauge = document.querySelector("#gauge-value");
  gauge.style.strokeDasharray = `${(positive * 100).toFixed(2)} 100`;
  gauge.classList.toggle("positive", predictedPositive);
  document.querySelector("#result-threshold").textContent = result.threshold.toFixed(2);
  document.querySelector("#target-definition").textContent = result.target_definition;
  resultState.textContent = "PREDICTION COMPLETE";
  resultState.classList.add("complete");
  renderSummary(result.features);
  renderLocalSensitivity(result.local_sensitivity);
}

function renderGlobalImportance(items) {
  const container = document.querySelector("#global-signals");
  container.replaceChildren();
  if (!items?.length) {
    container.innerHTML = '<p class="muted">Global feature importance is unavailable for this model.</p>';
    return;
  }
  const maximum = Math.max(...items.map((item) => item.importance), 0.0001);
  for (const item of items.slice(0, 6)) {
    const row = document.createElement("div");
    row.className = "signal-row";
    const label = document.createElement("span");
    label.className = "signal-label";
    label.textContent = item.feature;
    const track = document.createElement("span");
    track.className = "signal-track";
    const bar = document.createElement("span");
    bar.className = "signal-bar";
    bar.style.display = "block";
    bar.style.width = `${Math.max(3, item.importance / maximum * 100)}%`;
    track.append(bar);
    const value = document.createElement("span");
    value.className = "signal-value";
    value.textContent = item.importance.toFixed(3);
    row.append(label, track, value);
    container.append(row);
  }
}

function renderMetrics(metadata) {
  const metrics = metadata.test_metrics || {};
  const cards = [
    ["ACCURACY", metrics.accuracy],
    ["PRECISION", metrics.precision],
    ["RECALL", metrics.recall],
    ["F1-SCORE", metrics.f1],
    ["ROC-AUC", metrics.roc_auc],
    ["PR-AUC", metrics.pr_auc],
  ];
  const grid = document.querySelector("#metrics-grid");
  grid.replaceChildren();
  for (const [label, raw] of cards) {
    const card = document.createElement("article");
    card.className = "metric-card";
    const name = document.createElement("span");
    name.textContent = label;
    const value = document.createElement("strong");
    value.textContent = Number.isFinite(Number(raw)) ? Number(raw).toFixed(3) : "—";
    card.append(name, value);
    grid.append(card);
  }
  document.querySelector("#roc-value").textContent = `AUC ${Number(metrics.roc_auc).toFixed(3)}`;
  document.querySelector("#pr-value").textContent = `PR-AUC ${Number(metrics.pr_auc).toFixed(3)}`;
  const matrix = metrics.confusion_matrix;
  const matrixContainer = document.querySelector("#confusion-matrix");
  if (Array.isArray(matrix) && matrix.length === 2) {
    const cells = [
      ["Actual / predicted", "header"], ["Predicted 0", "header"], ["Predicted 1", "header"],
      ["Actual 0", "header"], [matrix[0][0], "TN"], [matrix[0][1], "FP"],
      ["Actual 1", "header"], [matrix[1][0], "FN"], [matrix[1][1], "TP"],
    ];
    matrixContainer.replaceChildren();
    for (const [value, kind] of cells) {
      const cell = document.createElement("div");
      cell.className = `matrix-cell ${kind === "header" ? "header" : ""}`;
      if (kind === "header") cell.textContent = value;
      else {
        const number = document.createElement("b");
        number.textContent = value;
        const label = document.createElement("span");
        label.textContent = kind;
        cell.append(number, label);
      }
      matrixContainer.append(cell);
    }
  }
  const tbody = document.querySelector("#comparison-table");
  tbody.replaceChildren();
  for (const entry of metadata.model_comparison || []) {
    const row = document.createElement("tr");
    const values = [
      entry.model,
      entry.metrics.accuracy,
      entry.metrics.precision,
      entry.metrics.recall,
      entry.metrics.f1,
      entry.metrics.roc_auc,
      entry.metrics.pr_auc,
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = typeof value === "number" ? value.toFixed(3) : value;
      row.append(cell);
    }
    tbody.append(row);
  }
}

function renderAbout(metadata) {
  const target = metadata.target_definition?.description || "DEATH_EVENT = 1 means an event was recorded; 0 means no event was recorded.";
  const crossValidation = metadata.cross_validation || {};
  const metrics = metadata.test_metrics || {};
  const details = [
    ["Dataset", metadata.dataset_source],
    ["Target", target],
    ["Training samples", metadata.training_samples],
    ["Validation samples", metadata.validation_samples],
    ["Test samples", metadata.test_samples],
    ["Feature count", metadata.num_features],
    ["Training methodology", metadata.training_methodology],
    ["Candidate selection", `${crossValidation.strategy || "Validation"} · selected using validation ROC-AUC`],
    ["Selected model", metadata.selected_model],
    ["Validation ROC-AUC", crossValidation.validation_auc],
    ["Test metrics", `Accuracy ${metrics.accuracy ?? "—"} · ROC-AUC ${metrics.roc_auc ?? "—"} · PR-AUC ${metrics.pr_auc ?? "—"}`],
    ["Classification threshold", metadata.classification_threshold],
    ["Model version", metadata.model_version],
    ["Training date (UTC)", metadata.training_date_utc],
  ];
  const list = document.querySelector("#model-details");
  list.replaceChildren();
  for (const [label, value] of details) {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    term.textContent = label;
    const description = document.createElement("dd");
    description.textContent = value == null ? "Not available" : String(value);
    row.append(term, description);
    list.append(row);
  }
  document.querySelector("#about-notice").textContent = metadata.educational_notice || "This model is not clinically validated and must not be used for diagnosis or treatment.";
}

async function loadDashboard() {
  try {
    const response = await fetch("/api/predict", { headers: { Accept: "application/json" } });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || `Server returned ${response.status}`);
    dashboardInfo = body;
    renderFields();
    const metadata = body.metadata || {};
    const metrics = metadata.test_metrics || {};
    document.querySelector("#summary-model").textContent = metadata.selected_model || "Trained model";
    document.querySelector("#summary-version").textContent = `Version ${metadata.model_version || "not specified"}`;
    document.querySelector("#summary-accuracy").textContent = Number.isFinite(Number(metrics.accuracy)) ? `${(Number(metrics.accuracy) * 100).toFixed(1)}%` : "—";
    document.querySelector("#summary-test-samples").textContent = metadata.test_samples ?? "—";
    document.querySelector("#summary-threshold").textContent = Number(metadata.classification_threshold ?? 0.5).toFixed(2);
    renderGlobalImportance(body.global_importance);
    renderMetrics(metadata);
    renderAbout(metadata);
  } catch (error) {
    const message = `Unable to load model information. ${error.message}. Confirm the Vercel Python function and model files are deployed.`;
    showToast(message);
    document.querySelector("#global-signals").innerHTML = '<p class="muted">Model information could not be loaded.</p>';
    document.querySelector("#metrics-grid").innerHTML = '<p class="muted">Model metrics could not be loaded.</p>';
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setFormError("");
  if (!dashboardInfo) {
    setFormError("Model information is still loading. Please wait a moment and try again.");
    return;
  }
  let features;
  try {
    features = readInputs();
  } catch (error) {
    setFormError(error.message);
    return;
  }
  submitButton.disabled = true;
  submitButton.querySelector(".button-label").textContent = "Evaluating model…";
  resultState.textContent = "RUNNING MODEL";
  resultState.classList.remove("complete");
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ features }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || `Server returned ${response.status}`);
    renderPrediction(body);
    setFormError("");
  } catch (error) {
    resultState.textContent = "PREDICTION FAILED";
    setFormError(error.message);
    showToast(`Prediction was not completed: ${error.message}`);
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector(".button-label").textContent = "Run model prediction";
  }
});

document.querySelectorAll("[data-view]").forEach((button) => {
  button.addEventListener("click", () => {
    const viewName = button.dataset.view;
    document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
    document.querySelector(`#${viewName}-view`)?.classList.add("active");
    document.querySelectorAll(".nav-link").forEach((link) => link.classList.toggle("active", link.dataset.view === viewName));
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
});

loadDashboard();
