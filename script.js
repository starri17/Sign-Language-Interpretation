import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/vision_bundle.mjs";

const MEDIAPIPE_VERSION = "0.10.22-rc.20250304";
const WASM_PATH = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const HAND_MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const POSE_ARM_CONNECTIONS = [
  [11, 12], // 양쪽 어깨
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
];
const POSE_ARM_LANDMARKS = new Set([11, 12, 13, 14, 15, 16]);
const MIN_POSE_VISIBILITY = 0.35;

const elements = {
  video: document.querySelector("#webcam"),
  canvas: document.querySelector("#outputCanvas"),
  cameraStage: document.querySelector("#cameraStage"),
  startButton: document.querySelector("#startButton"),
  stopButton: document.querySelector("#stopButton"),
  statusDot: document.querySelector("#statusDot"),
  statusText: document.querySelector("#statusText"),
  handCount: document.querySelector("#handCount"),
  fpsValue: document.querySelector("#fpsValue"),
  errorMessage: document.querySelector("#errorMessage"),
  errorText: document.querySelector("#errorText"),
};

const context = elements.canvas.getContext("2d");

let handLandmarker = null;
let poseLandmarker = null;
let mediaStream = null;
let animationFrameId = null;
let lastVideoTime = -1;
let isRunning = false;
let frameCounter = 0;
let fpsStartedAt = performance.now();

/**
 * MediaPipe 모델을 초기화합니다.
 * 향후 FaceLandmarker를 추가할 때 이 초기화 계층에 모델을 연결하면 됩니다.
 */
async function initializeTrackers() {
  setStatus("AI 모델 준비 중", "loading");
  hideError();

  try {
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);

    try {
      [handLandmarker, poseLandmarker] = await createLandmarkers(vision, "GPU");
    } catch (gpuError) {
      console.warn("GPU 초기화 실패, CPU 모드로 재시도합니다:", gpuError);
      [handLandmarker, poseLandmarker] = await createLandmarkers(vision, "CPU");
    }

    setStatus("카메라 대기 중", "idle");
    elements.startButton.disabled = false;
  } catch (error) {
    console.error("MediaPipe 초기화 오류:", error);
    setStatus("모델 로드 실패", "error");
    showError(
      "AI 모델을 불러오지 못했습니다. 인터넷 연결을 확인한 뒤 페이지를 새로고침해 주세요.",
    );
  }
}

async function createLandmarkers(vision, delegate) {
  let handTracker = null;
  let poseTracker = null;

  try {
    handTracker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: HAND_MODEL_PATH,
        delegate,
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    poseTracker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: POSE_MODEL_PATH,
        delegate,
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      outputSegmentationMasks: false,
    });

    return [handTracker, poseTracker];
  } catch (error) {
    handTracker?.close();
    poseTracker?.close();
    throw error;
  }
}

async function startCamera() {
  if (isRunning || !handLandmarker || !poseLandmarker) return;

  hideError();
  elements.startButton.disabled = true;
  setStatus("카메라 연결 중", "loading");

  if (!navigator.mediaDevices?.getUserMedia) {
    handleCameraError(new Error("UNSUPPORTED_BROWSER"));
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user",
      },
      audio: false,
    });

    elements.video.srcObject = mediaStream;
    await elements.video.play();
    await waitForVideoMetadata();

    resizeCanvas();
    isRunning = true;
    lastVideoTime = -1;
    frameCounter = 0;
    fpsStartedAt = performance.now();

    elements.cameraStage.classList.add("is-live");
    elements.stopButton.disabled = false;
    setStatus("실시간 추적 중", "live");
    animationFrameId = requestAnimationFrame(predictFrame);
  } catch (error) {
    handleCameraError(error);
  }
}

function stopCamera() {
  isRunning = false;

  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  elements.video.srcObject = null;
  elements.cameraStage.classList.remove("is-live");
  elements.startButton.disabled = !handLandmarker || !poseLandmarker;
  elements.stopButton.disabled = true;
  elements.handCount.textContent = "0";
  elements.fpsValue.textContent = "0";
  clearCanvas();
  setStatus("카메라 대기 중", "idle");
}

function predictFrame() {
  if (!isRunning) return;

  if (
    elements.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    elements.video.currentTime !== lastVideoTime
  ) {
    const timestamp = performance.now();

    try {
      const poseResult = poseLandmarker.detectForVideo(elements.video, timestamp);
      const handResult = handLandmarker.detectForVideo(elements.video, timestamp);

      drawTrackingResults(poseResult, handResult);
      updateStats(handResult.landmarks?.length ?? 0, timestamp);
      publishTrackingFrame(poseResult, handResult, timestamp);
      lastVideoTime = elements.video.currentTime;
    } catch (error) {
      console.error("프레임 분석 오류:", error);
      showError("영상 분석 중 오류가 발생했습니다. 카메라를 종료한 뒤 다시 시작해 주세요.");
      stopCamera();
      setStatus("추적 오류", "error");
      return;
    }
  }

  animationFrameId = requestAnimationFrame(predictFrame);
}

function drawTrackingResults(poseResult, handResult) {
  clearCanvas();

  const pose = poseResult.landmarks?.[0];
  if (pose) drawUpperBody(pose);

  const hands = handResult.landmarks ?? [];
  hands.forEach((landmarks, index) => {
    const categoryName = handResult.handedness?.[index]?.[0]?.categoryName ?? "Unknown";
    drawHand(landmarks, categoryName);
  });
}

function drawUpperBody(landmarks) {
  context.save();
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = "#b7ff3c";
  context.lineWidth = Math.max(3, elements.canvas.width * 0.0045);
  context.shadowColor = "rgba(183, 255, 60, 0.45)";
  context.shadowBlur = 10;

  for (const [startIndex, endIndex] of POSE_ARM_CONNECTIONS) {
    const start = landmarks[startIndex];
    const end = landmarks[endIndex];
    if (!isVisible(start) || !isVisible(end)) continue;
    drawLine(start, end);
  }

  context.shadowBlur = 7;
  for (const index of POSE_ARM_LANDMARKS) {
    const landmark = landmarks[index];
    if (!isVisible(landmark)) continue;
    drawPoint(landmark, "#b7ff3c", Math.max(4, elements.canvas.width * 0.006));
  }

  context.restore();
}

function drawHand(landmarks, categoryName) {
  // MediaPipe의 handedness는 카메라 원본 기준이며 화면은 거울처럼 반전됩니다.
  const color = categoryName === "Left" ? "#56e6ff" : "#ff7f66";

  context.save();
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = color;
  context.lineWidth = Math.max(2, elements.canvas.width * 0.0028);
  context.shadowColor = color;
  context.shadowBlur = 7;

  for (const connection of HandLandmarker.HAND_CONNECTIONS) {
    const startIndex = connection.start ?? connection[0];
    const endIndex = connection.end ?? connection[1];
    drawLine(landmarks[startIndex], landmarks[endIndex]);
  }

  for (const landmark of landmarks) {
    drawPoint(landmark, color, Math.max(2.5, elements.canvas.width * 0.0038));
  }

  context.restore();
}

function drawLine(start, end) {
  if (!start || !end) return;
  context.beginPath();
  context.moveTo(start.x * elements.canvas.width, start.y * elements.canvas.height);
  context.lineTo(end.x * elements.canvas.width, end.y * elements.canvas.height);
  context.stroke();
}

function drawPoint(landmark, color, radius) {
  context.beginPath();
  context.arc(
    landmark.x * elements.canvas.width,
    landmark.y * elements.canvas.height,
    radius,
    0,
    Math.PI * 2,
  );
  context.fillStyle = color;
  context.fill();
  context.lineWidth = Math.max(1, radius * 0.28);
  context.strokeStyle = "rgba(6, 10, 10, 0.8)";
  context.stroke();
}

function isVisible(landmark) {
  return landmark && (landmark.visibility ?? 1) >= MIN_POSE_VISIBILITY;
}

function resizeCanvas() {
  const width = elements.video.videoWidth || 1280;
  const height = elements.video.videoHeight || 720;
  elements.canvas.width = width;
  elements.canvas.height = height;
  elements.cameraStage.style.aspectRatio = `${width} / ${height}`;
}

function clearCanvas() {
  context.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
}

function updateStats(handCount, now) {
  elements.handCount.textContent = String(handCount);
  frameCounter += 1;

  const elapsed = now - fpsStartedAt;
  if (elapsed >= 1000) {
    elements.fpsValue.textContent = String(Math.round((frameCounter * 1000) / elapsed));
    frameCounter = 0;
    fpsStartedAt = now;
  }
}

/**
 * 후속 얼굴 표정/동적 수어 분류기가 원본 추적 결과를 구독할 수 있는 확장 지점입니다.
 * window.addEventListener("tracking:frame", (event) => { ... });
 */
function publishTrackingFrame(poseResult, handResult, timestamp) {
  window.dispatchEvent(
    new CustomEvent("tracking:frame", {
      detail: { poseResult, handResult, timestamp },
    }),
  );
}

function waitForVideoMetadata() {
  if (elements.video.readyState >= HTMLMediaElement.HAVE_METADATA) {
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    elements.video.addEventListener("loadedmetadata", resolve, { once: true });
  });
}

function handleCameraError(error) {
  console.error("카메라 실행 오류:", error);

  const messages = {
    NotAllowedError: "카메라 권한이 거부되었습니다. 브라우저 주소창의 카메라 권한을 허용해 주세요.",
    SecurityError: "보안 설정 때문에 카메라를 사용할 수 없습니다. localhost 또는 HTTPS로 실행해 주세요.",
    NotFoundError: "사용 가능한 카메라를 찾지 못했습니다. 카메라 연결 상태를 확인해 주세요.",
    NotReadableError: "다른 프로그램이 카메라를 사용 중입니다. 해당 프로그램을 종료한 뒤 다시 시도해 주세요.",
    OverconstrainedError: "카메라가 요청한 영상 설정을 지원하지 않습니다.",
    AbortError: "카메라 연결이 중단되었습니다. 잠시 후 다시 시도해 주세요.",
    UNSUPPORTED_BROWSER: "이 브라우저는 웹캠 기능을 지원하지 않습니다. 최신 Chrome 또는 Edge를 사용해 주세요.",
  };

  const message = messages[error.name] || messages[error.message] ||
    "카메라를 시작할 수 없습니다. 연결 및 브라우저 권한을 확인해 주세요.";

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  elements.video.srcObject = null;
  elements.startButton.disabled = false;
  elements.stopButton.disabled = true;
  elements.cameraStage.classList.remove("is-live");
  setStatus("카메라 연결 실패", "error");
  showError(message);
}

function setStatus(message, type) {
  elements.statusText.textContent = message;
  elements.statusDot.className = "status-dot";

  if (type === "loading") elements.statusDot.classList.add("is-loading");
  if (type === "live") elements.statusDot.classList.add("is-live");
  if (type === "error") elements.statusDot.classList.add("is-error");
}

function showError(message) {
  elements.errorText.textContent = message;
  elements.errorMessage.hidden = false;
}

function hideError() {
  elements.errorMessage.hidden = true;
  elements.errorText.textContent = "";
}

elements.startButton.addEventListener("click", startCamera);
elements.stopButton.addEventListener("click", stopCamera);
window.addEventListener("resize", () => {
  if (isRunning) resizeCanvas();
});
window.addEventListener("beforeunload", stopCamera);

initializeTrackers();
