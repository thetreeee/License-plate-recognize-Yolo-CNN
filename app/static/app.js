const byId = id => document.getElementById(id);
const form = byId("upload"), file = byId("file"), submit = byId("submit");
const camera = byId("camera"), result = byId("result"), live = byId("live");
const empty = byId("empty"), list = byId("plates"), noPlates = byId("no-plates");
const status = message => { byId("status").textContent = message; };
const imageUrl = data => "data:image/jpeg;base64," + data;
const cards = new Map();
let cameraSession = null;

function reset() {
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
      button.onclick = () => { result.currentTime = plate.seconds; };
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

form.onsubmit = async event => {
  event.preventDefault();
  const video = file.files[0];
  if (!video || cameraSession) return;
  if (video.size > 100 * 1024 ** 2) {
    status("Video vượt quá 100 MB.");
    return;
  }
  reset();
  submit.disabled = camera.disabled = file.disabled = true;
  status("Đang tải video lên…");
  try {
    const job = await json(await fetch("/api/videos", {
      method: "POST", body: video, headers: {"Content-Type": "application/octet-stream"},
    }));
    const endpoint = "/api/videos/" + job.id;
    live.src = endpoint + "/preview";
    live.hidden = false;
    empty.hidden = true;
    for (;;) {
      const data = await json(await fetch(endpoint, {cache: "no-store"}));
      if (data.status === "failed") throw new Error(data.error);
      const ready = data.status === "ready";
      showPlates(data.plates, ready);
      if (ready) {
        result.src = endpoint + "/result";
        result.hidden = false;
        empty.hidden = true;
        status(`Đã xử lý ${data.frames} khung hình.`);
        break;
      }
      const total = data.total ? ` / ${data.total}` : "";
      status(`Đang xử lý video · ${data.frames}${total} khung hình`);
      await new Promise(resolve => setTimeout(resolve, 800));
    }
  } catch (error) {
    status(error.message);
  } finally {
    live.hidden = true;
    live.removeAttribute("src");
    empty.hidden = !result.hidden;
    submit.disabled = camera.disabled = file.disabled = false;
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
  submit.disabled = file.disabled = false;
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
  submit.disabled = file.disabled = true;
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

window.addEventListener("pagehide", () => stopCamera());
