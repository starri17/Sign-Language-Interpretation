const MEDIAPIPE_VERSION = "0.10.22-rc.20250304";
const MEDIAPIPE_MODULE_URL =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/vision_bundle.mjs`;
const WASM_PATH = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const HAND_MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const POSE_UPPER_BODY_CONNECTIONS = [
  [0, 11], // 코-왼쪽 어깨
  [0, 12], // 코-오른쪽 어깨
  [11, 12], // 양쪽 어깨
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23], // 상체 왼쪽
  [12, 24], // 상체 오른쪽
  [23, 24], // 골반
];
const POSE_UPPER_BODY_LANDMARKS = new Set([0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]);
const MIN_POSE_VISIBILITY = 0.25;
const MIN_INFERENCE_INTERVAL_MS = 1000 / 15;
const VIDEO_HEALTH_CHECK_INTERVAL_MS = 1500;
const BLACK_FRAME_LUMA_THRESHOLD = 5;
const BLACK_FRAME_LIT_PIXEL_RATIO = 0.01;
const APP_BUILD = "20261116-2";
const LOG_PREFIX = "[KSL Tracker]";
const VIRTUAL_CAMERA_PATTERN = /virtual|eshare|obs|manycam|snap camera|droidcam|iriun/i;

const elements = {
  video: document.querySelector("#webcam"),
  canvas: document.querySelector("#outputCanvas"),
  cameraStage: document.querySelector("#cameraStage"),
  startButton: document.querySelector("#startButton"),
  stopButton: document.querySelector("#stopButton"),
  cameraSelect: document.querySelector("#cameraSelect"),
  refreshCamerasButton: document.querySelector("#refreshCamerasButton"),
  cameraListHint: document.querySelector("#cameraListHint"),
  videoDebugMode: document.querySelector("#videoDebugMode"),
  statusDot: document.querySelector("#statusDot"),
  statusText: document.querySelector("#statusText"),
  poseCount: document.querySelector("#poseCount"),
  handCount: document.querySelector("#handCount"),
  fpsValue: document.querySelector("#fpsValue"),
  errorMessage: document.querySelector("#errorMessage"),
  errorText: document.querySelector("#errorText"),
  cameraStatusCard: document.querySelector("#cameraStatusCard"),
  cameraStatusValue: document.querySelector("#cameraStatusValue"),
  cameraStatusDetail: document.querySelector("#cameraStatusDetail"),
  videoStatusCard: document.querySelector("#videoStatusCard"),
  videoStatusValue: document.querySelector("#videoStatusValue"),
  videoStatusDetail: document.querySelector("#videoStatusDetail"),
  modelStatusCard: document.querySelector("#modelStatusCard"),
  modelStatusValue: document.querySelector("#modelStatusValue"),
  modelStatusDetail: document.querySelector("#modelStatusDetail"),
};

const context = elements.canvas.getContext("2d");

let FilesetResolver = null;
let HandLandmarker = null;
let PoseLandmarker = null;
let handLandmarker = null;
let poseLandmarker = null;
let mediaStream = null;
let animationFrameId = null;
let predictionStartTimerId = null;
let trackerStartTimerId = null;
let videoHealthTimerId = null;
let lastVideoTime = -1;
let lastInferenceAt = 0;
let isRunning = false;
let isCheckingVideoSignal = false;
let isVideoSignalVerified = false;
let cameraState = "idle";
let videoState = "idle";
let trackerState = "idle";
let trackerInitializationPromise = null;
let lastCameraErrorMessage = "";
let lastVideoSignal = null;
let lastPoseCount = 0;
let lastHandCount = 0;
let frameCounter = 0;
let fpsStartedAt = performance.now();

/**
 * MediaPipe 모델을 초기화합니다.
 * 향후 FaceLandmarker를 추가할 때 이 초기화 계층에 모델을 연결하면 됩니다.
 */
async function initializeTrackers() {
  trackerState = "loading";
  updateAppStatus();
  console.info(`${LOG_PREFIX} MediaPipe 라이브러리 로딩 시작`, {
    moduleUrl: MEDIAPIPE_MODULE_URL,
    wasmPath: WASM_PATH,
  });

  try {
    const visionModule = await import(MEDIAPIPE_MODULE_URL);
    ({ FilesetResolver, HandLandmarker, PoseLandmarker } = visionModule);
    console.info(`${LOG_PREFIX} MediaPipe 라이브러리 로딩 완료`);

    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    console.info(`${LOG_PREFIX} MediaPipe WASM 준비 완료`);

    try {
      [handLandmarker, poseLandmarker] = await createLandmarkers(vision, "CPU");
      console.info(`${LOG_PREFIX} 손·포즈 모델 준비 완료 (CPU)`);
    } catch (cpuError) {
      console.warn(`${LOG_PREFIX} CPU 초기화 실패, GPU 모드로 재시도합니다.`, cpuError);
      [handLandmarker, poseLandmarker] = await createLandmarkers(vision, "GPU");
      console.info(`${LOG_PREFIX} 손·포즈 모델 준비 완료 (GPU)`);
    }

    trackerState = "ready";
    lastVideoTime = -1;
    if (cameraState !== "error" && videoState !== "black") hideError();
    updateAppStatus();
  } catch (error) {
    trackerState = "error";
    console.error(`${LOG_PREFIX} MediaPipe 초기화 오류`, error);
    updateAppStatus();
    if (cameraState !== "error") {
      showError(
        "AI 모델을 불러오지 못했습니다. 카메라 영상은 사용할 수 있지만 관절 추적은 동작하지 않습니다. 인터넷 연결과 브라우저 콘솔을 확인해 주세요.",
        "warning",
      );
    }
  }
}

function ensureTrackersInitialized() {
  if (elements.videoDebugMode.checked || videoState !== "playing") {
    return Promise.resolve();
  }
  if (trackerState === "ready" || trackerState === "error") {
    return Promise.resolve();
  }
  if (trackerInitializationPromise) return trackerInitializationPromise;

  trackerInitializationPromise = initializeTrackers().finally(() => {
    trackerInitializationPromise = null;
  });
  return trackerInitializationPromise;
}

function scheduleTrackerInitialization() {
  if (trackerStartTimerId !== null) window.clearTimeout(trackerStartTimerId);
  trackerStartTimerId = window.setTimeout(() => {
    trackerStartTimerId = null;
    ensureTrackersInitialized();
  }, 250);
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
      minHandDetectionConfidence: 0.35,
      minHandPresenceConfidence: 0.35,
      minTrackingConfidence: 0.4,
    });

    poseTracker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: POSE_MODEL_PATH,
        delegate,
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.35,
      minPosePresenceConfidence: 0.35,
      minTrackingConfidence: 0.4,
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
  console.info(`${LOG_PREFIX} 카메라 시작 버튼 클릭`, {
    isSecureContext: window.isSecureContext,
    cameraState,
    trackerState,
  });

  if (isRunning || cameraState === "connecting") {
    console.info(`${LOG_PREFIX} 카메라가 이미 실행 중이거나 연결 중입니다.`);
    return;
  }

  hideError();
  elements.startButton.disabled = true;
  elements.cameraSelect.disabled = true;
  cameraState = "connecting";
  videoState = "loading";
  isVideoSignalVerified = false;
  lastCameraErrorMessage = "";
  lastVideoSignal = null;
  const hadExplicitDeviceSelection = Boolean(elements.cameraSelect.value);
  updateAppStatus();

  if (!navigator.mediaDevices?.getUserMedia) {
    const reason = window.isSecureContext ? "UNSUPPORTED_BROWSER" : "INSECURE_CONTEXT";
    handleCameraError(new Error(reason));
    return;
  }

  try {
    await logCameraPermissionState();

    const constraints = createCameraConstraints();

    mediaStream = await acquireCameraStream(constraints);
    console.info(`${LOG_PREFIX} 카메라 스트림 획득 성공`, getStreamDiagnostics(mediaStream));

    let videoTracks = mediaStream.getVideoTracks();
    if (!mediaStream.active || videoTracks.length === 0 || videoTracks[0].readyState !== "live") {
      const error = new Error("MEDIA_STREAM_INACTIVE");
      error.name = "MediaStreamInactiveError";
      throw error;
    }

    const activeDeviceId = videoTracks[0]?.getSettings?.().deviceId ?? "";
    const videoInputs = await refreshCameraDevices(activeDeviceId);

    if (!hadExplicitDeviceSelection && isVirtualCameraLabel(videoTracks[0]?.label)) {
      const physicalCamera = videoInputs.find((device) => (
        device.deviceId !== activeDeviceId &&
        device.label &&
        !isVirtualCameraLabel(device.label)
      ));

      if (physicalCamera) {
        console.info(`${LOG_PREFIX} 가상 카메라 대신 실제 카메라를 우선 선택합니다.`, {
          from: videoTracks[0].label,
          to: physicalCamera.label,
        });

        try {
          const physicalStream = await navigator.mediaDevices.getUserMedia({
            video: {
              deviceId: { exact: physicalCamera.deviceId },
              width: { ideal: 1280 },
              height: { ideal: 720 },
              frameRate: { ideal: 30 },
            },
            audio: false,
          });
          const physicalTracks = physicalStream.getVideoTracks();
          if (physicalStream.active && physicalTracks[0]?.readyState === "live") {
            mediaStream.getTracks().forEach((track) => track.stop());
            mediaStream = physicalStream;
            videoTracks = physicalTracks;
            elements.cameraSelect.value = physicalCamera.deviceId;
            await refreshCameraDevices(physicalCamera.deviceId);
          } else {
            physicalStream.getTracks().forEach((track) => track.stop());
          }
        } catch (physicalCameraError) {
          console.warn(`${LOG_PREFIX} 실제 카메라 자동 선택 실패`, physicalCameraError);
          elements.cameraSelect.value = activeDeviceId;
        }
      }
    }

    attachTrackDiagnostics(mediaStream);
    cameraState = "connected";
    updateAppStatus();

    elements.video.muted = true;
    elements.video.autoplay = true;
    elements.video.playsInline = true;
    elements.video.srcObject = mediaStream;
    if (elements.video.srcObject !== mediaStream) {
      const error = new Error("VIDEO_SOURCE_OBJECT_MISMATCH");
      error.name = "VideoSourceObjectError";
      throw error;
    }
    console.info(`${LOG_PREFIX} video.srcObject 연결 완료`);

    const metadataPromise = waitForVideoMetadata();
    const playPromise = elements.video.play();
    console.info(`${LOG_PREFIX} video.play() 실행 요청`);
    await Promise.all([metadataPromise, playPromise]);
    console.info(`${LOG_PREFIX} 비디오 메타데이터 준비 완료`, {
      videoWidth: elements.video.videoWidth,
      videoHeight: elements.video.videoHeight,
      readyState: elements.video.readyState,
    });
    console.info(`${LOG_PREFIX} video.play() 실행 성공`, {
      paused: elements.video.paused,
      readyState: elements.video.readyState,
    });

    elements.cameraStage.classList.add("is-live");
    updateAppStatus();

    const firstFrameMetadata = await waitForFirstVideoFrame();
    console.info(`${LOG_PREFIX} 첫 카메라 프레임 표시 준비 완료`, firstFrameMetadata);

    if (
      elements.video.videoWidth <= 0 ||
      elements.video.videoHeight <= 0 ||
      elements.video.paused
    ) {
      const error = new Error("VIDEO_NOT_PLAYING");
      error.name = "VideoNotPlayingError";
      throw error;
    }

    resizeCanvas();
    isRunning = true;
    cameraState = "live";
    videoState = "checking";
    lastVideoTime = -1;
    lastInferenceAt = 0;
    frameCounter = 0;
    fpsStartedAt = performance.now();

    elements.stopButton.disabled = false;

    const initialSignal = await inspectVideoSignal(4, 140);
    lastVideoSignal = initialSignal;
    isVideoSignalVerified = true;
    videoState = initialSignal.isProbablyBlack ? "black" : "playing";
    elements.cameraSelect.disabled = elements.cameraSelect.options.length <= 1;
    console.info(`${LOG_PREFIX} 카메라 프레임 밝기 진단`, initialSignal);
    console.info(`${LOG_PREFIX} 화면 표시와 MediaPipe 입력 확인`, {
      displayVideoElementId: elements.video.id,
      mediaPipeInputElementId: elements.video.id,
      usesSameVideoElement: true,
      srcObjectMatchesStream: elements.video.srcObject === mediaStream,
      canvasBackground: getComputedStyle(elements.canvas).backgroundColor,
      videoDisplay: getComputedStyle(elements.video).display,
      videoVisibility: getComputedStyle(elements.video).visibility,
      videoOpacity: getComputedStyle(elements.video).opacity,
      videoZIndex: getComputedStyle(elements.video).zIndex,
      canvasZIndex: getComputedStyle(elements.canvas).zIndex,
    });
    updateAppStatus();

    if (initialSignal.isProbablyBlack) {
      resetDetectionStats();
      elements.fpsValue.textContent = "0";
      clearCanvas();
      showError(
        "카메라는 연결되었지만 실제 프레임이 거의 검은색입니다. 위의 카메라 목록에서 다른 웹캠을 선택하거나 렌즈 덮개·조명·Windows 카메라 사용 상태를 확인해 주세요.",
        "warning",
      );
    } else if (!elements.videoDebugMode.checked) {
      scheduleTrackerInitialization();
    }

    if (trackerState === "error") {
      showError(
        "카메라 영상은 정상적으로 실행 중이지만 AI 모델을 불러오지 못해 관절 추적은 사용할 수 없습니다.",
        "warning",
      );
    }

    predictionStartTimerId = window.setTimeout(() => {
      predictionStartTimerId = null;
      if (isRunning) animationFrameId = requestAnimationFrame(predictFrame);
    }, 100);
    startVideoHealthMonitor();
  } catch (error) {
    handleCameraError(error);
  }
}

function stopCamera() {
  console.info(`${LOG_PREFIX} 카메라 종료`);
  isRunning = false;
  cameraState = "idle";
  stopVideoHealthMonitor();

  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }

  if (predictionStartTimerId !== null) {
    window.clearTimeout(predictionStartTimerId);
    predictionStartTimerId = null;
  }

  if (trackerStartTimerId !== null) {
    window.clearTimeout(trackerStartTimerId);
    trackerStartTimerId = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  elements.video.srcObject = null;
  elements.cameraStage.classList.remove("is-live");
  videoState = "idle";
  isVideoSignalVerified = false;
  lastVideoSignal = null;
  lastVideoTime = -1;
  lastInferenceAt = 0;
  elements.startButton.disabled = !navigator.mediaDevices?.getUserMedia;
  elements.cameraSelect.disabled = elements.cameraSelect.options.length <= 1;
  elements.stopButton.disabled = true;
  resetDetectionStats();
  elements.fpsValue.textContent = "0";
  clearCanvas();
  updateAppStatus();
}

function predictFrame() {
  if (!isRunning) return;

  const timestamp = performance.now();

  if (
    trackerState === "ready" &&
    handLandmarker &&
    poseLandmarker &&
    videoState === "playing" &&
    isVideoSignalVerified &&
    !elements.videoDebugMode.checked &&
    elements.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    elements.video.currentTime !== lastVideoTime &&
    timestamp - lastInferenceAt >= MIN_INFERENCE_INTERVAL_MS
  ) {
    lastInferenceAt = timestamp;

    try {
      const poseResult = poseLandmarker.detectForVideo(elements.video, timestamp);
      const handResult = handLandmarker.detectForVideo(elements.video, timestamp);

      const poseCount = poseResult.landmarks?.length ?? 0;
      const handCount = handResult.landmarks?.length ?? 0;

      drawTrackingResults(poseResult, handResult);
      updateStats(poseCount, handCount, timestamp);
      publishTrackingFrame(poseResult, handResult, timestamp);
      lastVideoTime = elements.video.currentTime;
    } catch (error) {
      trackerState = "error";
      console.error(`${LOG_PREFIX} 프레임 분석 오류`, error);
      resetDetectionStats();
      elements.fpsValue.textContent = "0";
      clearCanvas();
      showError(
        "관절 분석 중 오류가 발생했습니다. 카메라 영상은 계속 표시되지만 추적은 중지되었습니다. 브라우저 콘솔을 확인해 주세요.",
        "warning",
      );
      updateAppStatus();
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

  for (const [startIndex, endIndex] of POSE_UPPER_BODY_CONNECTIONS) {
    const start = landmarks[startIndex];
    const end = landmarks[endIndex];
    if (!isVisible(start) || !isVisible(end)) continue;
    drawLine(start, end);
  }

  context.shadowBlur = 7;
  for (const index of POSE_UPPER_BODY_LANDMARKS) {
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

function updateStats(poseCount, handCount, now) {
  const detectionChanged = poseCount !== lastPoseCount || handCount !== lastHandCount;
  lastPoseCount = poseCount;
  lastHandCount = handCount;
  elements.poseCount.textContent = String(poseCount);
  elements.handCount.textContent = String(handCount);
  frameCounter += 1;

  if (detectionChanged) updateAppStatus();

  const elapsed = now - fpsStartedAt;
  if (elapsed >= 1000) {
    elements.fpsValue.textContent = String(Math.round((frameCounter * 1000) / elapsed));
    frameCounter = 0;
    fpsStartedAt = now;
  }
}

function resetDetectionStats() {
  lastPoseCount = 0;
  lastHandCount = 0;
  elements.poseCount.textContent = "0";
  elements.handCount.textContent = "0";
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

function waitForVideoMetadata(timeoutMs = 10000) {
  if (elements.video.readyState >= HTMLMediaElement.HAVE_METADATA) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const timeoutId = window.setTimeout(() => {
      cleanup();
      const error = new Error("VIDEO_METADATA_TIMEOUT");
      error.name = "VideoMetadataTimeoutError";
      reject(error);
    }, timeoutMs);

    const handleLoadedMetadata = () => {
      cleanup();
      resolve();
    };

    const handleVideoError = () => {
      cleanup();
      const error = new Error("VIDEO_ELEMENT_ERROR");
      error.name = "VideoElementError";
      reject(error);
    };

    const cleanup = () => {
      window.clearTimeout(timeoutId);
      elements.video.removeEventListener("loadedmetadata", handleLoadedMetadata);
      elements.video.removeEventListener("error", handleVideoError);
    };

    elements.video.addEventListener("loadedmetadata", handleLoadedMetadata);
    elements.video.addEventListener("error", handleVideoError);
  });
}

function waitForFirstVideoFrame(timeoutMs = 10000) {
  return new Promise((resolve, reject) => {
    let frameCallbackId = null;
    let fallbackAnimationId = null;

    const timeoutId = window.setTimeout(() => {
      cleanup();
      const error = new Error("VIDEO_FRAME_TIMEOUT");
      error.name = "VideoFrameTimeoutError";
      reject(error);
    }, timeoutMs);

    const finish = (metadata = {}) => {
      cleanup();
      resolve({
        ...metadata,
        videoWidth: elements.video.videoWidth,
        videoHeight: elements.video.videoHeight,
        readyState: elements.video.readyState,
        paused: elements.video.paused,
        currentTime: elements.video.currentTime,
      });
    };

    const checkFallbackFrame = () => {
      if (
        elements.video.videoWidth > 0 &&
        elements.video.videoHeight > 0 &&
        elements.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
        !elements.video.paused
      ) {
        finish();
        return;
      }

      fallbackAnimationId = requestAnimationFrame(checkFallbackFrame);
    };

    const cleanup = () => {
      window.clearTimeout(timeoutId);
      if (
        frameCallbackId !== null &&
        typeof elements.video.cancelVideoFrameCallback === "function"
      ) {
        elements.video.cancelVideoFrameCallback(frameCallbackId);
      }
      if (fallbackAnimationId !== null) cancelAnimationFrame(fallbackAnimationId);
    };

    if (typeof elements.video.requestVideoFrameCallback === "function") {
      frameCallbackId = elements.video.requestVideoFrameCallback((_, metadata) => {
        finish(metadata);
      });
    } else {
      fallbackAnimationId = requestAnimationFrame(checkFallbackFrame);
    }
  });
}

function createCameraConstraints() {
  const selectedDeviceId = elements.cameraSelect.value;
  const videoConstraints = {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    frameRate: { ideal: 30 },
  };

  if (selectedDeviceId) {
    videoConstraints.deviceId = { exact: selectedDeviceId };
  } else {
    videoConstraints.facingMode = { ideal: "user" };
  }

  return { video: videoConstraints, audio: false };
}

function isVirtualCameraLabel(label = "") {
  return VIRTUAL_CAMERA_PATTERN.test(label);
}

async function acquireCameraStream(preferredConstraints) {
  console.info(`${LOG_PREFIX} getUserMedia() 호출`, preferredConstraints);

  try {
    return await navigator.mediaDevices.getUserMedia(preferredConstraints);
  } catch (error) {
    const canRetryWithBasicConstraints =
      error?.name === "OverconstrainedError" ||
      (error?.name === "NotFoundError" && Boolean(elements.cameraSelect.value));

    if (!canRetryWithBasicConstraints) throw error;

    console.warn(
      `${LOG_PREFIX} 선택한 장치 또는 영상 조건을 사용할 수 없어 기본 카메라로 다시 시도합니다.`,
      { name: error.name, message: error.message, constraint: error.constraint },
    );
    elements.cameraSelect.value = "";
    elements.cameraListHint.textContent =
      "선택한 설정을 사용할 수 없어 기본 카메라로 다시 연결하고 있습니다.";

    const fallbackConstraints = { video: true, audio: false };
    console.info(`${LOG_PREFIX} getUserMedia() 기본 조건 재호출`, fallbackConstraints);
    return navigator.mediaDevices.getUserMedia(fallbackConstraints);
  }
}

async function refreshCameraDevices(preferredDeviceId = elements.cameraSelect.value) {
  if (!navigator.mediaDevices?.enumerateDevices) {
    elements.cameraSelect.disabled = true;
    elements.cameraListHint.textContent = "이 브라우저는 카메라 목록 조회를 지원하지 않습니다.";
    return [];
  }

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter((device) => device.kind === "videoinput");
    const activeDeviceId = mediaStream?.getVideoTracks()[0]?.getSettings?.().deviceId ?? "";
    let targetDeviceId = preferredDeviceId || activeDeviceId;

    if (!targetDeviceId && videoInputs.some((device) => isVirtualCameraLabel(device.label))) {
      targetDeviceId = videoInputs.find((device) => (
        device.label && !isVirtualCameraLabel(device.label)
      ))?.deviceId ?? "";
    }

    elements.cameraSelect.replaceChildren();
    const defaultOption = new Option("기본 카메라", "");
    elements.cameraSelect.add(defaultOption);

    videoInputs.forEach((device, index) => {
      const label = device.label || `카메라 ${index + 1}`;
      elements.cameraSelect.add(new Option(label, device.deviceId));
    });

    if (
      targetDeviceId &&
      videoInputs.some((device) => device.deviceId === targetDeviceId)
    ) {
      elements.cameraSelect.value = targetDeviceId;
    }

    elements.cameraSelect.disabled =
      videoInputs.length === 0 || cameraState === "connecting" || cameraState === "connected";
    elements.cameraListHint.textContent = videoInputs.length > 0
      ? `${videoInputs.length}개의 카메라를 찾았습니다. 검은 화면이면 다른 장치를 선택해 보세요.`
      : "사용 가능한 카메라를 찾지 못했습니다.";

    console.info(`${LOG_PREFIX} 사용 가능한 카메라 목록`, videoInputs.map((device, index) => ({
      label: device.label || `카메라 ${index + 1}`,
      deviceId: device.deviceId,
      groupId: device.groupId,
    })));

    return videoInputs;
  } catch (error) {
    console.error(`${LOG_PREFIX} 카메라 목록 조회 오류`, error);
    elements.cameraListHint.textContent = "카메라 목록을 가져오지 못했습니다. 권한을 확인해 주세요.";
    return [];
  }
}

async function inspectVideoSignal(sampleCount = 1, delayMs = 0) {
  const samples = [];

  for (let index = 0; index < sampleCount; index += 1) {
    const sample = inspectCurrentVideoFrame();
    if (sample.available) samples.push(sample);
    if (index < sampleCount - 1 && delayMs > 0) await delay(delayMs);
  }

  if (samples.length === 0) {
    return {
      available: false,
      isProbablyBlack: false,
      meanLuma: null,
      litPixelRatio: null,
      reason: "영상 픽셀을 검사할 수 없음",
    };
  }

  const brightestSample = samples.reduce((brightest, sample) => (
    sample.meanLuma > brightest.meanLuma ? sample : brightest
  ));

  return {
    ...brightestSample,
    sampleCount: samples.length,
    isProbablyBlack: samples.every((sample) => sample.isProbablyBlack),
  };
}

function inspectCurrentVideoFrame() {
  if (
    elements.video.videoWidth <= 0 ||
    elements.video.videoHeight <= 0 ||
    elements.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
  ) {
    return { available: false, isProbablyBlack: false };
  }

  try {
    const sampleWidth = 48;
    const sampleHeight = Math.max(
      1,
      Math.round(sampleWidth * (elements.video.videoHeight / elements.video.videoWidth)),
    );
    const sampleCanvas = document.createElement("canvas");
    sampleCanvas.width = sampleWidth;
    sampleCanvas.height = sampleHeight;
    const sampleContext = sampleCanvas.getContext("2d", { willReadFrequently: true });
    sampleContext.drawImage(elements.video, 0, 0, sampleWidth, sampleHeight);

    const pixels = sampleContext.getImageData(0, 0, sampleWidth, sampleHeight).data;
    let lumaTotal = 0;
    let litPixelCount = 0;
    const pixelCount = pixels.length / 4;

    for (let offset = 0; offset < pixels.length; offset += 4) {
      const luma = (
        (0.2126 * pixels[offset]) +
        (0.7152 * pixels[offset + 1]) +
        (0.0722 * pixels[offset + 2])
      );
      lumaTotal += luma;
      if (luma >= 16) litPixelCount += 1;
    }

    const meanLuma = lumaTotal / pixelCount;
    const litPixelRatio = litPixelCount / pixelCount;

    return {
      available: true,
      meanLuma,
      litPixelRatio,
      isProbablyBlack:
        meanLuma < BLACK_FRAME_LUMA_THRESHOLD &&
        litPixelRatio < BLACK_FRAME_LIT_PIXEL_RATIO,
    };
  } catch (error) {
    console.warn(`${LOG_PREFIX} 영상 픽셀 진단을 수행하지 못했습니다.`, error);
    return { available: false, isProbablyBlack: false };
  }
}

function startVideoHealthMonitor() {
  stopVideoHealthMonitor();

  videoHealthTimerId = window.setInterval(async () => {
    if (!isRunning || isCheckingVideoSignal) return;

    isCheckingVideoSignal = true;
    try {
      const signal = await inspectVideoSignal(2, 100);
      if (!signal.available) return;

      const wasBlack = videoState === "black";
      lastVideoSignal = signal;

      if (signal.isProbablyBlack) {
        videoState = "black";
        resetDetectionStats();
        elements.fpsValue.textContent = "0";
        clearCanvas();

        if (!wasBlack) {
          showError(
            "카메라에서 검은 프레임만 감지되었습니다. 다른 카메라를 선택하거나 렌즈 덮개·조명·다른 프로그램의 카메라 사용 여부를 확인해 주세요.",
            "warning",
          );
        }
      } else {
        videoState = "playing";
        if (wasBlack && trackerState !== "error") hideError();
        if (trackerState === "idle" && !elements.videoDebugMode.checked) {
          scheduleTrackerInitialization();
        }
      }

      updateAppStatus();
    } finally {
      isCheckingVideoSignal = false;
    }
  }, VIDEO_HEALTH_CHECK_INTERVAL_MS);
}

function stopVideoHealthMonitor() {
  if (videoHealthTimerId !== null) {
    window.clearInterval(videoHealthTimerId);
    videoHealthTimerId = null;
  }
  isCheckingVideoSignal = false;
}

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function logCameraPermissionState() {
  if (!navigator.permissions?.query) {
    console.info(`${LOG_PREFIX} Permissions API를 지원하지 않아 카메라 권한 상태를 미리 확인할 수 없습니다.`);
    return;
  }

  try {
    const permission = await navigator.permissions.query({ name: "camera" });
    console.info(`${LOG_PREFIX} 현재 카메라 권한 상태: ${permission.state}`);
  } catch (error) {
    console.info(`${LOG_PREFIX} 이 브라우저에서는 카메라 권한 사전 조회를 지원하지 않습니다.`, error);
  }
}

function getStreamDiagnostics(stream) {
  return {
    id: stream.id,
    active: stream.active,
    videoTracks: stream.getVideoTracks().map((track) => ({
      id: track.id,
      label: track.label,
      readyState: track.readyState,
      enabled: track.enabled,
      muted: track.muted,
      settings: track.getSettings?.(),
    })),
  };
}

function attachTrackDiagnostics(stream) {
  stream.getVideoTracks().forEach((track) => {
    track.addEventListener("mute", () => {
      console.warn(`${LOG_PREFIX} 카메라 트랙의 영상 공급이 일시 중단되었습니다.`, track.label);
      if (mediaStream === stream) {
        videoState = "loading";
        updateAppStatus();
      }
    });
    track.addEventListener("unmute", () => {
      console.info(`${LOG_PREFIX} 카메라 트랙의 영상 공급이 재개되었습니다.`, track.label);
      if (mediaStream === stream && !elements.video.paused) {
        videoState = isVideoSignalVerified
          ? (lastVideoSignal?.isProbablyBlack ? "black" : "playing")
          : "checking";
        updateAppStatus();
      }
    });
    track.addEventListener("ended", () => {
      console.warn(`${LOG_PREFIX} 카메라 트랙이 종료되었습니다.`, track.label);
      if (!isRunning) return;

      const error = new Error("CAMERA_DISCONNECTED");
      error.name = "CameraDisconnectedError";
      handleCameraError(error);
    });
  });
}

function handleCameraError(error) {
  console.error(`${LOG_PREFIX} 카메라 실행 오류`, {
    name: error?.name,
    message: error?.message,
    error,
  });

  const messages = {
    NotAllowedError: "카메라 권한이 거부되었습니다. 브라우저 주소창의 카메라 권한과 Windows 설정 > 개인정보 및 보안 > 카메라의 접근 권한을 허용해 주세요.",
    SecurityError: "보안 설정 때문에 카메라를 사용할 수 없습니다. localhost 또는 HTTPS로 실행해 주세요.",
    NotFoundError: "사용 가능한 카메라를 찾지 못했습니다. 카메라 연결 상태를 확인해 주세요.",
    NotReadableError: "다른 프로그램이 카메라를 사용 중입니다. 해당 프로그램을 종료한 뒤 다시 시도해 주세요.",
    OverconstrainedError: "카메라가 요청한 영상 설정을 지원하지 않습니다.",
    AbortError: "카메라 연결이 중단되었습니다. 잠시 후 다시 시도해 주세요.",
    CameraDisconnectedError: "카메라 연결이 끊어졌습니다. 장치를 다시 연결한 뒤 카메라 시작 버튼을 눌러 주세요.",
    MediaStreamInactiveError: "카메라 스트림은 생성되었지만 활성 비디오 트랙이 없습니다. 다른 카메라를 선택하거나 다른 프로그램을 종료해 주세요.",
    VideoSourceObjectError: "카메라 스트림을 비디오 요소에 연결하지 못했습니다. 페이지를 새로고침한 뒤 다시 시도해 주세요.",
    VideoMetadataTimeoutError: "카메라 영상 정보를 가져오는 데 시간이 너무 오래 걸립니다. 다른 프로그램에서 카메라를 사용 중인지 확인해 주세요.",
    VideoFrameTimeoutError: "카메라 연결은 성공했지만 실제 영상 프레임이 도착하지 않았습니다. 다른 프로그램의 카메라 사용 여부와 브라우저 권한을 확인해 주세요.",
    VideoNotPlayingError: "카메라 영상의 크기가 0이거나 재생이 멈춰 있습니다. 브라우저를 새로고침한 뒤 다시 시도해 주세요.",
    VideoElementError: "카메라 영상을 비디오 화면에 연결하는 중 오류가 발생했습니다.",
    INSECURE_CONTEXT: "카메라는 localhost 또는 HTTPS 환경에서만 사용할 수 있습니다. VS Code Live Server로 다시 실행해 주세요.",
    UNSUPPORTED_BROWSER: "이 브라우저는 웹캠 기능을 지원하지 않습니다. 최신 Chrome 또는 Edge를 사용해 주세요.",
  };

  const message = messages[error.name] || messages[error.message] ||
    "카메라를 시작할 수 없습니다. 연결 및 브라우저 권한을 확인해 주세요.";

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  isRunning = false;
  isVideoSignalVerified = false;
  stopVideoHealthMonitor();
  cameraState = "error";
  videoState = "error";
  lastCameraErrorMessage = message;

  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
  if (predictionStartTimerId !== null) {
    window.clearTimeout(predictionStartTimerId);
    predictionStartTimerId = null;
  }
  if (trackerStartTimerId !== null) {
    window.clearTimeout(trackerStartTimerId);
    trackerStartTimerId = null;
  }

  elements.video.srcObject = null;
  elements.startButton.disabled = !navigator.mediaDevices?.getUserMedia;
  elements.cameraSelect.disabled = elements.cameraSelect.options.length <= 1;
  elements.stopButton.disabled = true;
  elements.cameraStage.classList.remove("is-live");
  resetDetectionStats();
  clearCanvas();
  updateAppStatus();
  showError(message);
}

function updateAppStatus() {
  updateRuntimeStatus();

  if (cameraState === "error") {
    setStatus("카메라 연결 실패", "error");
    return;
  }

  if (cameraState === "connecting") {
    setStatus("카메라 연결 중", "loading");
    return;
  }

  if (cameraState === "connected") {
    setStatus("카메라 연결됨 · 첫 영상 프레임 확인 중", "loading");
    return;
  }

  if (cameraState === "live") {
    if (videoState === "black") {
      setStatus("카메라 연결됨 · 검은 영상 감지", "warning");
    } else if (videoState !== "playing") {
      setStatus("카메라 연결됨 · 영상 재생 대기 중", "loading");
    } else if (elements.videoDebugMode.checked) {
      setStatus("영상 디버그 모드 · 관절 추적 일시 중지", "warning");
    } else if (trackerState === "idle") {
      setStatus("카메라 영상 확인 완료 · AI 모델 시작 대기", "loading");
    } else if (trackerState === "ready") {
      if (lastPoseCount > 0 || lastHandCount > 0) {
        setStatus(`사람 ${lastPoseCount}명 · 손 ${lastHandCount}개 실시간 추적 중`, "live");
      } else {
        setStatus("사람을 찾는 중 · 상반신과 손을 화면 안에 보여 주세요", "warning");
      }
    } else if (trackerState === "loading") {
      setStatus("카메라 실행 중 · AI 모델 준비 중", "loading");
    } else {
      setStatus("카메라 실행 중 · 관절 추적 불가", "warning");
    }
    return;
  }

  if (trackerState === "ready") {
    setStatus("카메라 대기 중", "idle");
  } else if (trackerState === "idle") {
    setStatus("카메라 대기 중 · AI 모델 미실행", "idle");
  } else if (trackerState === "loading") {
    setStatus("카메라 사용 가능 · AI 모델 준비 중", "loading");
  } else {
    setStatus("카메라 사용 가능 · 모델 로드 실패", "warning");
  }
}

function updateRuntimeStatus() {
  const cameraStates = {
    idle: ["대기", "장치 연결 전", "idle"],
    connecting: ["연결 중", "권한 및 장치 확인 중", "loading"],
    connected: ["연결됨", getCameraStatusDetail(), "live"],
    live: ["연결됨", getCameraStatusDetail(), "live"],
    error: ["오류", lastCameraErrorMessage || "카메라 연결 실패", "error"],
  };

  const videoStates = {
    idle: ["대기", getVideoStatusDetail(), "idle"],
    loading: ["재생 준비 중", getVideoStatusDetail(), "loading"],
    checking: ["프레임 확인 중", getVideoStatusDetail(), "loading"],
    playing: ["재생 중", getVideoStatusDetail(), "live"],
    black: ["검은 프레임 감지", getVideoStatusDetail(), "warning"],
    error: ["재생 실패", getVideoStatusDetail(), "error"],
  };

  const modelStates = {
    idle: ["대기", "정상 카메라 프레임 확인 후 시작", "idle"],
    loading: ["준비 중", "라이브러리와 모델 로딩 중", "loading"],
    ready: [
      "준비 완료",
      isRunning
        ? `사람 ${lastPoseCount}명 · 손 ${lastHandCount}개 감지`
        : "입력 #webcam · 손·포즈 추적 가능",
      "live",
    ],
    error: ["로드 실패", "카메라 영상만 사용 가능", "error"],
  };

  if (elements.videoDebugMode.checked && trackerState === "ready") {
    modelStates.ready = ["일시 중지", "영상 디버그 모드 사용 중", "warning"];
  }

  setRuntimeStatus(
    elements.cameraStatusCard,
    elements.cameraStatusValue,
    elements.cameraStatusDetail,
    cameraStates[cameraState] ?? cameraStates.idle,
  );
  setRuntimeStatus(
    elements.videoStatusCard,
    elements.videoStatusValue,
    elements.videoStatusDetail,
    videoStates[videoState] ?? videoStates.idle,
  );
  setRuntimeStatus(
    elements.modelStatusCard,
    elements.modelStatusValue,
    elements.modelStatusDetail,
    modelStates[trackerState] ?? modelStates.loading,
  );
}

function setRuntimeStatus(card, valueElement, detailElement, state) {
  const [value, detail, stateName] = state;
  card.dataset.state = stateName;
  valueElement.textContent = value;
  detailElement.textContent = detail;
  detailElement.title = detail;
}

function getCameraStatusDetail() {
  const track = mediaStream?.getVideoTracks()[0];
  if (!track) return "활성 비디오 트랙 없음";

  const settings = track.getSettings?.() ?? {};
  const size = settings.width && settings.height
    ? `${settings.width}×${settings.height}`
    : "해상도 확인 중";
  const supplyState = track.muted ? "muted" : "unmuted";
  return `${track.label || "카메라"} · ${size} · ${track.readyState} · ${supplyState}`;
}

function getVideoStatusDetail() {
  const playState = elements.video.paused ? "paused" : "playing";
  const brightness = lastVideoSignal?.available
    ? ` · 밝기 ${lastVideoSignal.meanLuma.toFixed(1)}`
    : "";
  return `${elements.video.videoWidth}×${elements.video.videoHeight} · readyState ${elements.video.readyState} · ${playState}${brightness}`;
}

function setStatus(message, type) {
  elements.statusText.textContent = message;
  elements.statusDot.className = "status-dot";

  if (type === "loading") elements.statusDot.classList.add("is-loading");
  if (type === "live") elements.statusDot.classList.add("is-live");
  if (type === "warning") elements.statusDot.classList.add("is-warning");
  if (type === "error") elements.statusDot.classList.add("is-error");
}

function showError(message, type = "error") {
  elements.errorText.textContent = message;
  elements.errorMessage.classList.toggle("is-warning", type === "warning");
  elements.errorMessage.hidden = false;
}

function hideError() {
  elements.errorMessage.hidden = true;
  elements.errorMessage.classList.remove("is-warning");
  elements.errorText.textContent = "";
}

elements.startButton.addEventListener("click", startCamera);
elements.stopButton.addEventListener("click", stopCamera);
elements.videoDebugMode.addEventListener("change", () => {
  const enabled = elements.videoDebugMode.checked;
  elements.cameraStage.classList.toggle("is-video-debug", enabled);
  clearCanvas();
  resetDetectionStats();
  elements.fpsValue.textContent = "0";
  lastVideoTime = -1;
  updateAppStatus();

  if (!enabled && isRunning && videoState === "playing" && trackerState === "idle") {
    scheduleTrackerInitialization();
  }

  const videoStyle = getComputedStyle(elements.video);
  const canvasStyle = getComputedStyle(elements.canvas);
  console.info(`${LOG_PREFIX} 영상 디버그 모드 ${enabled ? "켜짐" : "꺼짐"}`, {
    video: {
      display: videoStyle.display,
      visibility: videoStyle.visibility,
      opacity: videoStyle.opacity,
      zIndex: videoStyle.zIndex,
      filter: videoStyle.filter,
      transform: videoStyle.transform,
    },
    canvas: {
      display: canvasStyle.display,
      zIndex: canvasStyle.zIndex,
      backgroundColor: canvasStyle.backgroundColor,
    },
  });
});
elements.refreshCamerasButton.addEventListener("click", async () => {
  elements.refreshCamerasButton.disabled = true;
  try {
    await refreshCameraDevices();
  } finally {
    elements.refreshCamerasButton.disabled = false;
  }
});
elements.cameraSelect.addEventListener("change", async () => {
  const selectedOption = elements.cameraSelect.selectedOptions[0];
  console.info(`${LOG_PREFIX} 카메라 선택 변경`, {
    label: selectedOption?.textContent,
    deviceId: elements.cameraSelect.value,
  });

  if (mediaStream || cameraState === "connecting" || cameraState === "connected" || cameraState === "live") {
    stopCamera();
    await startCamera();
  }
});
elements.video.addEventListener("loadedmetadata", () => {
  console.info(`${LOG_PREFIX} video loadedmetadata 이벤트`, {
    videoWidth: elements.video.videoWidth,
    videoHeight: elements.video.videoHeight,
  });
  updateRuntimeStatus();
});
elements.video.addEventListener("playing", () => {
  console.info(`${LOG_PREFIX} video playing 이벤트`);
  if (mediaStream) {
    videoState = isVideoSignalVerified
      ? (lastVideoSignal?.isProbablyBlack ? "black" : "playing")
      : "checking";
    updateAppStatus();
  }
});
elements.video.addEventListener("waiting", () => {
  console.warn(`${LOG_PREFIX} video waiting 이벤트`);
  if (mediaStream) {
    videoState = "loading";
    updateAppStatus();
  }
});
elements.video.addEventListener("stalled", () => {
  console.warn(`${LOG_PREFIX} video stalled 이벤트`);
  if (mediaStream) {
    videoState = "loading";
    updateAppStatus();
  }
});
elements.video.addEventListener("pause", () => {
  console.info(`${LOG_PREFIX} video pause 이벤트`);
  if (isRunning) {
    videoState = "loading";
    updateAppStatus();
  }
});
elements.video.addEventListener("resize", () => {
  console.info(`${LOG_PREFIX} video resize 이벤트`, {
    videoWidth: elements.video.videoWidth,
    videoHeight: elements.video.videoHeight,
  });
  if (isRunning) resizeCanvas();
  updateRuntimeStatus();
});
elements.video.addEventListener("error", () => {
  console.error(`${LOG_PREFIX} video 요소 오류`, elements.video.error);
  videoState = "error";
  updateAppStatus();
});
window.addEventListener("resize", () => {
  if (isRunning) resizeCanvas();
});
window.addEventListener("beforeunload", stopCamera);

console.info(`${LOG_PREFIX} 애플리케이션 초기화`, {
  build: APP_BUILD,
  url: window.location.href,
  isSecureContext: window.isSecureContext,
  hasMediaDevices: Boolean(navigator.mediaDevices),
  hasGetUserMedia: Boolean(navigator.mediaDevices?.getUserMedia),
});

if (!navigator.mediaDevices?.getUserMedia) {
  const reason = window.isSecureContext ? "UNSUPPORTED_BROWSER" : "INSECURE_CONTEXT";
  handleCameraError(new Error(reason));
} else {
  elements.startButton.disabled = false;
  updateAppStatus();
  refreshCameraDevices();
  navigator.mediaDevices.addEventListener?.("devicechange", () => {
    console.info(`${LOG_PREFIX} 카메라 장치 변경 감지`);
    refreshCameraDevices();
  });
}
