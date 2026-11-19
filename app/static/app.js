const byId = id => document.getElementById(id);
const form = byId("upload"), file = byId("file");
const camera = byId("camera"), result = byId("result"), live = byId("live");
const empty = byId("empty"), list = byId("plates"), noPlates = byId("no-plates");
const status = message => { byId("status").textContent = message; };
const imageUrl = data => "data:image/jpeg;base64," + data;
const cards = new Map();
let cameraSession = null;
const previewCanvas = byId("preview-canvas"), playback = byId("playback");
const highlight = byId("highlight"), screen = previewCanvas.parentElement;
let uploadEndpoint = null, viewPaused = false, previewAnimation = null, selectedPlate = null;

function updatePlayback() {
  const paused = result.hidden ? viewPaused : result.paused;
  playback.classList.toggle("paused", paused);
  playback.classList.toggle("completed", !result.hidden);
  playback.setAttribute("aria-label", paused ? "Tiếp tục video" : "Tạm dừng video");
  byId("playback-label").textContent = paused ? "Tiếp tục" : "Tạm dừng";
}

function drawPreview() {
  if (!viewPaused && live.naturalWidth && !previewCanvas.hidden) {
    if (previewCanvas.width !== live.naturalWidth || previewCanvas.height !== live.naturalHeight) {
      previewCanvas.width = live.naturalWidth;
      previewCanvas.height = live.naturalHeight;
    }
    previewCanvas.getContext("2d").drawImage(live, 0, 0);
    playback.hidden = false;
  }
  previewAnimation = requestAnimationFrame(drawPreview);
}

function drawHighlight() {
  highlight.hidden = true;
  if (!selectedPlate || result.hidden || !result.paused || !result.videoWidth ||
      Math.abs(result.currentTime - selectedPlate.seek_seconds) > .02) return;
  const bounds = screen.getBoundingClientRect();
  const scale = Math.min(bounds.width / result.videoWidth, bounds.height / result.videoHeight);
  const width = result.videoWidth * scale, height = result.videoHeight * scale;
  const [x1, y1, x2, y2] = selectedPlate.bbox;
  highlight.style.left = `${(bounds.width - width) / 2 + x1 * width}px`;
  highlight.style.top = `${(bounds.height - height) / 2 + y1 * height}px`;
  highlight.style.width = `${(x2 - x1) * width}px`;
  highlight.style.height = `${(y2 - y1) * height}px`;
  highlight.hidden = false;
}

playback.onclick = async () => {
  if (!result.hidden) {
    if (result.paused) {
      try { await result.play(); } catch (error) { status(error.message); }
    } else result.pause();
    return;
  }
  if (!uploadEndpoint) return;
  playback.disabled = true;
  try {
    const paused = !viewPaused;
    await json(await fetch(uploadEndpoint + "/playback", {
      method: "PATCH", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({paused}),
    }));
    viewPaused = paused;
    updatePlayback();
  } catch (error) { status(error.message); }
  finally { playback.disabled = false; }
};

for (const event of ["loadedmetadata", "seeked", "timeupdate", "play", "pause", "ended"]) {
  result.addEventListener(event, () => { updatePlayback(); drawHighlight(); });
}
new ResizeObserver(drawHighlight).observe(screen);

function reset() {
  cancelAnimationFrame(previewAnimation);
  previewAnimation = null;
  uploadEndpoint = selectedPlate = null;
  viewPaused = false;
  previewCanvas.hidden = playback.hidden = highlight.hidden = true;
  result.pause();
  cards.clear();
  list.replaceChildren();
  noPlates.hidden = false;
  result.removeAttribute("src");
  result.load();
  result.hidden = live.hidden = true;
  live.removeAttribute("src");
  empty.hidden = false;
}

function showPlates(plates, seekable) {
  for (const plate of plates) {
    const existing = cards.get(plate.id);
    if (existing?.signature === `${plate.plate}:${plate.seconds}:${seekable}`) continue;
    const row = document.createElement("li");
    const image = new Image();
    image.src = imageUrl(plate.crop);
    image.alt = `Ảnh crop biển số ${plate.plate}`;
    const info = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = plate.plate;
    const detail = document.createElement("small");
    detail.textContent = `${plate.status} · ID ${plate.id} · ${plate.seconds.toFixed(2)} s`;
    info.append(name, detail);
    if (seekable) {
      const button = document.createElement("button");
      button.className = "seek";
      button.textContent = "Tua đến vị trí";
      button.onclick = () => {
        result.pause();
        selectedPlate = plate.bbox ? {...plate, seek_seconds: plate.seek_seconds ?? plate.seconds} : null;
        for (const [id, card] of cards) card.row.classList.toggle("selected", id === plate.id);
        result.currentTime = plate.seek_seconds ?? plate.seconds;
        drawHighlight();
      };
      info.append(button);
    }
    row.append(image, info);
    if (existing) existing.row.replaceWith(row);
    else list.prepend(row);
    cards.set(plate.id, {row, signature: `${plate.plate}:${plate.seconds}:${seekable}`});
    noPlates.hidden = true;
  }
}

async function json(response) {
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Yêu cầu thất bại.");
  return data;
}

form.onsubmit = event => event.preventDefault();
file.onchange = async () => {
  const video = file.files[0];
  if (!video || cameraSession) return;
  if (video.size > 100 * 1024 ** 2) {
    status("Video vượt quá 100 MB.");
    return;
  }
  reset();
  camera.disabled = file.disabled = true;
  status("Đang tải video lên…");
  try {
    const job = await json(await fetch("/api/videos", {
      method: "POST", body: video, headers: {"Content-Type": "application/octet-stream"},
    }));
    const endpoint = "/api/videos/" + job.id;
    uploadEndpoint = endpoint;
    live.src = endpoint + "/preview";
    previewCanvas.hidden = false;
    updatePlayback();
    drawPreview();
    empty.hidden = true;
    for (;;) {
      const data = await json(await fetch(endpoint, {cache: "no-store"}));
      if (data.status === "failed") throw new Error(data.error);
      const ready = data.status === "ready";
      showPlates(data.plates, ready && !viewPaused);
      if (ready && !viewPaused) {
        result.src = endpoint + "/result";
        result.hidden = false;
        updatePlayback();
        empty.hidden = true;
        status(`Đã xử lý ${data.frames} khung hình.`);
        break;
      }
      const total = data.total ? ` / ${data.total}` : "";
      status(`${viewPaused ? "Đã tạm dừng" : "Đang xử lý video"} · ${data.frames}${total} khung hình`);
      await new Promise(resolve => setTimeout(resolve, 800));
    }
  } catch (error) {
    status(error.message);
  } finally {
    cancelAnimationFrame(previewAnimation);
    previewCanvas.hidden = true;
    live.hidden = true;
    live.removeAttribute("src");
    empty.hidden = !result.hidden;
    uploadEndpoint = null;
    playback.hidden = result.hidden;
    camera.disabled = file.disabled = false;
    file.value = "";
  }
};

function stopCamera(message) {
  const session = cameraSession;
  if (!session) return;
  cameraSession = null;
  session.stream?.getTracks().forEach(track => track.stop());
  session.socket?.close();
  session.video.srcObject = null;
  camera.textContent = "Mở camera";
  file.disabled = false;
  if (message) status(message);
}

camera.onclick = async () => {
  if (cameraSession) {
    stopCamera("Đã dừng camera.");
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    status("Mở ứng dụng bằng HTTPS hoặc localhost để dùng camera.");
    return;
  }
  reset();
  const session = {video: document.createElement("video")};
  cameraSession = session;
  camera.textContent = "Dừng camera";
  file.disabled = true;
  status("Đang xin quyền camera…");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {width: {ideal: 960}, height: {ideal: 540}}, audio: false,
    });
    if (cameraSession !== session) {
      stream.getTracks().forEach(track => track.stop());
      return;
    }
    session.stream = stream;
    const video = session.video;
    video.muted = video.playsInline = true;
    video.srcObject = stream;
    await video.play();
    if (cameraSession !== session) return;
    const canvas = document.createElement("canvas");
    const scale = Math.min(1, 960 / Math.max(video.videoWidth, video.videoHeight));
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    const context = canvas.getContext("2d");
    const socket = session.socket = new WebSocket(
      `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/api/camera`
    );
    function send() {
      if (cameraSession !== session || socket.readyState !== WebSocket.OPEN) return;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(blob => {
        if (cameraSession === session && blob && socket.readyState === WebSocket.OPEN) socket.send(blob);
      }, "image/jpeg", .85);
    }
    socket.onopen = send;
    socket.onmessage = ({data}) => {
      if (cameraSession !== session) return;
      const response = JSON.parse(data);
      if (response.error) {
        stopCamera(response.error);
        return;
      }
      live.src = imageUrl(response.frame);
      live.hidden = false;
      empty.hidden = true;
      showPlates(response.plates, false);
      status("Camera đang nhận diện.");
      requestAnimationFrame(send);
    };
    socket.onclose = () => {
      if (cameraSession === session) stopCamera("Đã ngắt kết nối camera.");
    };
    socket.onerror = () => {
      if (cameraSession === session) stopCamera("Không kết nối được camera với máy chủ.");
    };
    stream.getVideoTracks()[0].onended = () => {
      if (cameraSession === session) stopCamera("Camera đã ngừng cấp hình.");
    };
  } catch (error) {
    if (cameraSession === session) stopCamera(`Không mở được camera. ${error.message}`);
  }
};

window.addEventListener("pagehide", () => {
  stopCamera();
  if (uploadEndpoint && viewPaused) {
    fetch(uploadEndpoint + "/playback", {
      method: "PATCH", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({paused: false}), keepalive: true,
    }).catch(() => {});
  }
});
