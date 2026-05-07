// ===== STYLE PROMPTS =====
const STYLE_PROMPTS = {
  descriptive: "Descriptive",
  poetic: "Poetic",
  technical: "Technical",
  social: "Social"
};

// ===== LOADING STATUS MESSAGES =====
const STATUS_STEPS = [
  "Encoding image features via CNN...",
  "Extracting spatial feature maps...",
  "Passing context vector to RNN decoder...",
  "Generating token sequence...",
  "Finalizing natural language output..."
];

// ===== STATE =====
let uploadedFile = null;
let selectedStyle = "descriptive";

// ===== DOM REFERENCES =====
const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");

const previewSection = document.getElementById("previewSection");
const previewImg = document.getElementById("previewImg");
const previewMeta = document.getElementById("previewMeta");

const changeBtn = document.getElementById("changeBtn");

const generateBtn = document.getElementById("generateBtn");

const loadingBar = document.getElementById("loadingBar");
const loadingFill = document.getElementById("loadingFill");

const statusText = document.getElementById("statusText");

const resultSection = document.getElementById("resultSection");

const caption1El = document.getElementById("caption1");
const type1El = document.getElementById("type1");
const tags1El = document.getElementById("tags1");

const copyBtn1 = document.getElementById("copyBtn1");

// ===== STYLE CHIP SELECTION =====
document.querySelectorAll(".option-chip").forEach(chip => {

  chip.addEventListener("click", () => {

    document
      .querySelectorAll(".option-chip")
      .forEach(c => c.classList.remove("active"));

    chip.classList.add("active");

    selectedStyle = chip.dataset.style;
  });
});

// ===== UPLOAD — CLICK =====
uploadZone.addEventListener("click", () => {
  fileInput.click();
});

// ===== DRAG & DROP =====
uploadZone.addEventListener("dragover", e => {
  e.preventDefault();
  uploadZone.classList.add("dragging");
});

uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("dragging");
});

uploadZone.addEventListener("drop", e => {

  e.preventDefault();

  uploadZone.classList.remove("dragging");

  const file = e.dataTransfer.files[0];

  if (file && file.type.startsWith("image/")) {
    handleFile(file);
  }
});

// ===== FILE INPUT =====
fileInput.addEventListener("change", e => {

  if (e.target.files[0]) {
    handleFile(e.target.files[0]);
  }
});

// ===== CHANGE IMAGE =====
changeBtn.addEventListener("click", () => {

  uploadedFile = null;

  previewSection.style.display = "none";

  uploadZone.style.display = "block";

  resultSection.style.display = "none";

  fileInput.value = "";
});

// ===== HANDLE FILE =====
function handleFile(file) {

  uploadedFile = file;

  const reader = new FileReader();

  reader.onload = ev => {

    previewImg.src = ev.target.result;

    previewMeta.textContent =
      `${file.name} — ${(file.size / 1024).toFixed(0)} KB`;

    previewSection.style.display = "block";

    uploadZone.style.display = "none";

    resultSection.style.display = "none";
  };

  reader.readAsDataURL(file);
}

// ===== GENERATE CAPTION =====
generateBtn.addEventListener("click", generateCaption);

async function generateCaption() {

  if (!uploadedFile) {

    shakeUploadZone();

    return;
  }

  setLoading(true);

  const styleLabel =
    selectedStyle.charAt(0).toUpperCase() +
    selectedStyle.slice(1);

  try {

    // Create form data
    const formData = new FormData();

    formData.append("image", uploadedFile);

    formData.append("style", selectedStyle);

    // Send image to Flask backend
    const response = await fetch(
      "http://127.0.0.1:5000/generate-caption",
      {
        method: "POST",
        body: formData
      }
    );

    if (!response.ok) {

      throw new Error("Backend server error");
    }

    const data = await response.json();

    showResult({
      caption: data.caption,
      tags: data.tags || []
    }, styleLabel);

  } catch (err) {

    showError(err.message);

  } finally {

    setLoading(false);
  }
}

// ===== SHOW RESULT =====
function showResult(parsed, styleLabel) {

  type1El.textContent = styleLabel;

  caption1El.textContent =
    parsed.caption || "No caption generated.";

  tags1El.innerHTML = "";

  (parsed.tags || []).forEach(tag => {

    const span = document.createElement("span");

    span.className = "tag";

    span.textContent =
      "#" + tag.replace(/\s+/g, "");

    tags1El.appendChild(span);
  });

  resultSection.style.display = "block";

  resultSection.scrollIntoView({
    behavior: "smooth",
    block: "nearest"
  });
}

// ===== SHOW ERROR =====
function showError(msg) {

  type1El.textContent = "Error";

  caption1El.textContent =
    "Something went wrong: " + msg;

  tags1El.innerHTML = "";

  resultSection.style.display = "block";
}

// ===== COPY BUTTON =====
copyBtn1.addEventListener("click", () => {

  const text = caption1El.textContent;

  navigator.clipboard.writeText(text)
    .then(() => {

      copyBtn1.classList.add("copied");

      copyBtn1.innerHTML =
        '<i class="ti ti-check"></i>';

      setTimeout(() => {

        copyBtn1.classList.remove("copied");

        copyBtn1.innerHTML =
          '<i class="ti ti-copy"></i>';

      }, 2000);
    })
    .catch(() => {

      const ta = document.createElement("textarea");

      ta.value = text;

      document.body.appendChild(ta);

      ta.select();

      document.execCommand("copy");

      document.body.removeChild(ta);
    });
});

// ===== LOADING STATE =====
let statusInterval = null;

let progressValue = 0;

function setLoading(on) {

  if (on) {

    generateBtn.disabled = true;

    generateBtn.innerHTML =
      '<i class="ti ti-loader spinning"></i> Processing...';

    loadingBar.style.display = "block";

    statusText.style.display = "block";

    resultSection.style.display = "none";

    progressValue = 0;

    loadingFill.style.width = "0%";

    let step = 0;

    statusText.textContent = STATUS_STEPS[0];

    statusInterval = setInterval(() => {

      step = Math.min(
        step + 1,
        STATUS_STEPS.length - 1
      );

      statusText.textContent =
        STATUS_STEPS[step];

      progressValue =
        Math.min(progressValue + 18, 88);

      loadingFill.style.width =
        progressValue + "%";

    }, 700);

  } else {

    clearInterval(statusInterval);

    loadingFill.style.width = "100%";

    setTimeout(() => {

      loadingBar.style.display = "none";

      statusText.style.display = "none";

      loadingFill.style.width = "0%";

    }, 500);

    generateBtn.disabled = false;

    generateBtn.innerHTML =
      '<i class="ti ti-bolt"></i> Generate Caption';
  }
}

// ===== SHAKE EFFECT =====
function shakeUploadZone() {

  uploadZone.style.borderColor =
    "rgba(255,80,80,0.7)";

  uploadZone.style.background =
    "rgba(255,80,80,0.06)";

  uploadZone.style.display = "block";

  setTimeout(() => {

    uploadZone.style.borderColor =
      "rgba(108,63,255,0.4)";

    uploadZone.style.background =
      "rgba(108,63,255,0.04)";

  }, 1200);
}