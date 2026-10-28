const video = document.getElementById("video");
const canvas = document.getElementById("captureCanvas");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const clearButton = document.getElementById("clearButton");
const statusText = document.getElementById("status");
const sentenceText = document.getElementById("sentenceText");
const predictedWord = document.getElementById("predictedWord");
const confidenceValue = document.getElementById("confidenceValue");
const modelState = document.getElementById("modelState");

const probRows = new Map();
document.querySelectorAll(".prob-row").forEach((row) => {
  probRows.set(row.dataset.word, row);
});

let stream = null;
let pollTimer = null;
let requestInFlight = false;

function setStatus(message) {
  statusText.textContent = message;
}

function updateButtons(isRunning) {
  startButton.disabled = isRunning;
  stopButton.disabled = !isRunning;
}

function stopStream() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  video.srcObject = null;
  requestInFlight = false;
  updateButtons(false);
  setStatus("Camera stopped.");
}

async function sendFrame() {
  if (!stream || requestInFlight || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return;
  }

  requestInFlight = true;
  const width = video.videoWidth || 640;
  const height = video.videoHeight || 480;
  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, width, height);

  const blob = await new Promise((resolve) => {
    canvas.toBlob(resolve, "image/jpeg", 0.82);
  });

  if (!blob) {
    requestInFlight = false;
    return;
  }

  const body = new FormData();
  body.append("frame", blob, "frame.jpg");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Prediction request failed");
    }
    applyPrediction(data);
  } catch (error) {
    setStatus(error.message);
  } finally {
    requestInFlight = false;
  }
}

function applyPrediction(data) {
  sentenceText.textContent = data.sentence_text;
  predictedWord.textContent = data.predicted_word;
  confidenceValue.textContent = Number(data.confidence).toFixed(2);
  modelState.textContent = data.ready ? "Listening" : "Collecting 20 frames";

  Object.entries(data.probabilities).forEach(([word, score]) => {
    const row = probRows.get(word);
    if (!row) {
      return;
    }
    row.querySelector(".prob-value").textContent = Number(score).toFixed(2);
    row.querySelector(".prob-fill").style.width = `${Math.max(0, Math.min(100, score * 100))}%`;
  });

  if (data.ready) {
    setStatus("Prediction stream is running.");
  } else {
    setStatus("Camera is on. Collecting enough frames for the first prediction.");
  }
}

async function startCamera() {
  try {
    setStatus("Requesting camera permission...");
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user"
      },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
    updateButtons(true);
    setStatus("Camera started. Show one of the trained signs.");
    pollTimer = setInterval(sendFrame, 250);
  } catch (error) {
    setStatus(`Camera permission failed: ${error.message}`);
    stopStream();
  }
}

async function clearSentence() {
  await fetch("/reset", { method: "POST" });
  sentenceText.textContent = "Perform a sign";
  predictedWord.textContent = "Waiting...";
  confidenceValue.textContent = "0.00";
  modelState.textContent = "Warming up";
  probRows.forEach((row) => {
    row.querySelector(".prob-value").textContent = "0.00";
    row.querySelector(".prob-fill").style.width = "0%";
  });
  setStatus("Sentence cleared.");
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopStream);
clearButton.addEventListener("click", clearSentence);

window.addEventListener("beforeunload", stopStream);
