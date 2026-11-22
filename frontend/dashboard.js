"use strict";

const byId = (id) => document.getElementById(id);
const labelText = (label) => label.replaceAll("_", " ");
const percent = (value) => (100 * value).toFixed(1) + "%";
const predictionForm = byId("prediction-form");
const uploadForm = byId("upload-form");
const retrainButton = byId("retrain-button");
const uploadButton = byId("upload-button");
let hasEnoughImages = false;
let activeJob = null;
let startingJob = false;
let previewUrl;

function message(id, text, error = false) {
  const target = byId(id);
  target.textContent = text;
  target.classList.toggle("error", error);
}

async function request(path, options) {
  const response = await fetch(path, { ...options, redirect: "manual" });
  if (response.type === "opaqueredirect") {
    const error = new Error("Sign in again to continue.");
    error.name = "SignInRequired";
    throw error;
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data?.detail;
    const description = Array.isArray(detail) ? detail.map((item) => item.msg).join(". ") : detail;
    throw new Error(description || "Request failed (" + response.status + ").");
  }
  if (!data) throw new Error("The API did not return JSON.");
  return data;
}

function formatUptime(seconds) {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor(seconds / 3600) % 24;
  const minutes = Math.floor(seconds / 60) % 60;
  if (days) return days + "d " + hours + "h";
  if (hours) return hours + "h " + minutes + "m";
  if (minutes) return minutes + "m " + Math.floor(seconds % 60) + "s";
  return Math.floor(seconds) + "s";
}

async function refreshHealth() {
  try {
    const health = await request("/health");
    byId("health-status").textContent = health.model_loaded ? "Online" : "Starting";
    byId("uptime").textContent = formatUptime(health.uptime_seconds);
    byId("model-version").textContent = health.model_version;
    byId("sign-in-again").hidden = true;
  } catch (error) {
    const signInRequired = error.name === "SignInRequired";
    byId("health-status").textContent = signInRequired ? "Sign-in required" : "Unavailable";
    byId("sign-in-again").hidden = !signInRequired;
    byId("uptime").textContent = "—";
    byId("model-version").textContent = "—";
  }
}

function pair(name, value) {
  const row = document.createElement("div");
  const label = document.createElement("dt");
  const count = document.createElement("dd");
  label.textContent = name;
  count.textContent = value;
  row.append(label, count);
  return row;
}

function updateRetrainButton() {
  retrainButton.disabled = !hasEnoughImages || startingJob || Boolean(activeJob);
}

async function refreshDataset() {
  try {
    const summary = await request("/datasets/summary");
    const select = byId("upload-label");
    if (select.options.length === 1) {
      for (const label of summary.class_names) select.add(new Option(labelText(label), label));
    }
    byId("class-counts").replaceChildren(...summary.class_names.map(
      (name) => pair(labelText(name), summary.class_counts[name])
    ));
    hasEnoughImages = summary.total_images >= summary.minimum_images_for_retraining &&
      summary.class_names.every((name) => summary.class_counts[name] >= summary.minimum_images_per_class);
    message("training-requirements", hasEnoughImages
      ? summary.total_images + " images ready for retraining."
      : "Retraining needs " + summary.minimum_images_for_retraining + " images total and at least " +
        summary.minimum_images_per_class + " per class.");
    uploadButton.disabled = false;
  } catch (error) {
    hasEnoughImages = false;
    message("training-requirements", "Could not read upload totals. " + error.message, true);
  }
  updateRetrainButton();
}

byId("prediction-file").addEventListener("change", (event) => {
  const preview = byId("preview");
  const file = event.target.files[0];
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  preview.hidden = !file;
  if (file) {
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
  } else {
    preview.removeAttribute("src");
  }
  byId("prediction-result").hidden = true;
  message("prediction-status", "");
});

predictionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = predictionForm.querySelector("button");
  const file = byId("prediction-file");
  const body = new FormData(predictionForm);
  button.disabled = file.disabled = true;
  byId("prediction-result").hidden = true;
  message("prediction-status", "Classifying image…");
  try {
    const result = await request("/predict", { method: "POST", body });
    byId("predicted-class").textContent = result.display_class;
    byId("prediction-detail").textContent = percent(result.confidence) + " confidence · " +
      result.inference_ms + " ms inference";
    const probabilities = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
    byId("probabilities").replaceChildren(...probabilities.map(([name, value]) => {
      const row = pair(labelText(name), percent(value));
      const meter = document.createElement("meter");
      meter.max = 1;
      meter.value = value;
      meter.setAttribute("aria-label", labelText(name));
      row.lastChild.prepend(meter);
      return row;
    }));
    byId("prediction-result").hidden = false;
    message("prediction-status", "Prediction complete.");
  } catch (error) {
    message("prediction-status", error.message, true);
  } finally {
    button.disabled = file.disabled = false;
  }
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const body = new FormData(uploadForm);
  const inputs = Array.from(uploadForm.elements);
  inputs.forEach((input) => { input.disabled = true; });
  message("upload-status", "Saving images…");
  try {
    const result = await request("/datasets/upload", { method: "POST", body });
    uploadForm.reset();
    message("upload-status", result.stored + " saved · " + result.duplicates + " duplicates");
  } catch (error) {
    message("upload-status", error.message, true);
  } finally {
    inputs.forEach((input) => { input.disabled = false; });
    await refreshDataset();
  }
});

async function pollJob() {
  try {
    const job = await request("/retrain/" + encodeURIComponent(activeJob));
    if (job.status === "queued" || job.status === "running") {
      message("retrain-status", job.status === "queued"
        ? "Retraining queued."
        : "Training with " + job.dataset_images + " images for " + job.epochs + " epochs.");
      setTimeout(pollJob, 2000);
      return;
    }
    if (job.status === "completed") {
      const accuracy = job.metrics?.accuracy;
      message("retrain-status", "Model " + job.model_version + " is active." +
        (typeof accuracy === "number" ? " Training accuracy " + percent(accuracy) + "." : ""));
      void refreshHealth();
    } else {
      message("retrain-status", job.error || "Retraining failed.", true);
    }
  } catch (error) {
    message("retrain-status", "Could not read job status. Retrying. " + error.message, true);
    setTimeout(pollJob, 5000);
    return;
  }
  activeJob = null;
  updateRetrainButton();
}

retrainButton.addEventListener("click", async () => {
  startingJob = true;
  updateRetrainButton();
  message("retrain-status", "Starting retraining…");
  try {
    const job = await request("/retrain", { method: "POST" });
    activeJob = job.job_id;
    void pollJob();
  } catch (error) {
    message("retrain-status", error.message, true);
  } finally {
    startingJob = false;
    await refreshDataset();
  }
});

void refreshHealth();
void refreshDataset();
setInterval(refreshHealth, 10000);
