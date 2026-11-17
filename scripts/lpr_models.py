"""Checkpoint-compatible OCR models and preprocessing for the VNLP notebooks.

Supports both v1 checkpoints (crnn_best.pt, layout_best.pt) and
v2 checkpoints (crnn_best_v2.pt, layout_best_v2.pt).

Input crops are RGB uint8. No filename labels, ground-truth layout, or cached
OCR images are used by PlateReader at inference time.
"""
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


# 31-character alphabet used in LPRNet v5 and CRNN v2 (excluding I, O, J, Q, W)
CRNN_V2_CHARSET = "".join(list("0123456789") + list("ABCDEFGHKLMNPSTUVXYZ") + ["R"])
# 36-character standard alphanumeric alphabet used in CRNN v1
CRNN_V1_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def check_weight_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing weights: {path}")
    with path.open("rb") as handle:
        if handle.read(48).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"{path} is a Git LFS pointer. Run: git lfs pull")
    return path


def load_checkpoint(path):
    # These are the user's trusted training checkpoints, not arbitrary downloads.
    return torch.load(check_weight_file(path), map_location="cpu", weights_only=False)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class CRNN(nn.Module):
    def __init__(self, num_classes, hidden=256):
        super().__init__()
        def block(cin, cout):
            return [
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            ]
        self.cnn = nn.Sequential(
            *block(3, 64), nn.MaxPool2d(2, 2),
            *block(64, 128), nn.MaxPool2d(2, 2),
            *block(128, 256), *block(256, 256), nn.MaxPool2d((2, 1), (2, 1)),
            *block(256, 512), *block(512, 512), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, (2, 1), bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
        )
        self.rnn = nn.LSTM(512, hidden, num_layers=2, batch_first=True,
                           bidirectional=True, dropout=0.1)
        self.head = nn.Linear(hidden * 2, num_classes)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        f = self.cnn(x)
        assert f.shape[2] == 1, f"Feature height must be 1: {f.shape}"
        seq = f.squeeze(2).transpose(1, 2)
        seq, _ = self.rnn(seq)
        return self.head(seq).transpose(0, 1)  # logits [T, B, C]


def row_separator(crop):
    """Tìm một đường phân cách có thể nghiêng; không giả định khe ngang ở h/2."""
    h, w = crop.shape[:2]
    if h < 12 or w < 12:
        return np.full(w, h / 2, dtype=np.float32), False, "too_small", np.zeros(h), np.zeros((h, w), np.uint8)
    sh = 96
    sw = min(256, max(64, round(w * sh / h)))
    gray = cv2.cvtColor(cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 31, 7)
    ink = (mask > 0).astype(np.float32)
    # Bỏ hai mép dọc có viền/ốc vít khi chấm điểm; giữ nguyên ảnh RGB khi OCR.
    xs = np.arange(round(0.12 * sw), round(0.88 * sw))
    centers = np.arange(round(0.30 * sh), round(0.70 * sh) + 1)
    drifts = np.linspace(-0.35 * sh, 0.35 * sh, 15)
    cc, dd = np.meshgrid(centers, drifts, indexing="ij")
    cc = cc.ravel()
    dd = dd.ravel()
    ys = cc[:, None] + dd[:, None] * (xs[None, :] / max(1, sw - 1) - 0.5)

    def line_ink(y_values):
        values = []
        for band in (-1, 0, 1):
            yy = np.clip(np.rint(y_values + band).astype(int), 0, sh - 1)
            values.append(ink[yy, xs[None, :]].mean(axis=1))
        return np.mean(values, axis=0)

    gap_ink = line_ink(ys)
    # Nhẹ nhàng ưu tiên vùng giữa và ít nghiêng khi nhiều khe có cùng mức mực.
    scores = gap_ink + 0.04 * np.abs(cc / sh - 0.5) + 0.003 * np.abs(dd) / (0.35 * sh)
    best = int(np.argmin(scores))
    # Hai phía của khe phải có nét chữ; độ tin cậy chỉ là heuristic để audit.
    top = max(float(line_ink(ys[best:best + 1] - offset * sh)[0]) for offset in (0.12, 0.18, 0.24))
    bottom = max(float(line_ink(ys[best:best + 1] + offset * sh)[0]) for offset in (0.12, 0.18, 0.24))
    ratio = float(gap_ink[best]) / max(1e-6, min(top, bottom))
    ok = min(top, bottom) > 0.03 and ratio < 0.65
    cut = (cc[best] + dd[best] * (np.arange(w) / max(1, w - 1) - 0.5)) * (h / sh)
    cut = np.clip(cut, 2, h - 2).astype(np.float32)
    projection = ink.mean(axis=1)
    projection = cv2.resize(projection[:, None], (1, h)).ravel()
    debug_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return cut, bool(ok), ("sloped_gap" if ok else "low_confidence_candidate"), projection, debug_mask


def row_strip(crop, y_top, y_bot, out_w=94, out_h=24):
    """Nắn dải giữa 2 đường y_top, y_bot thành dải thẳng, rồi resize về out_w x out_h (chuẩn LPRNet v5)."""
    w = crop.shape[1]
    out_height = max(2, int(round(float(np.mean(y_bot - y_top)))))
    t = np.linspace(0, 1, out_height, dtype=np.float32)[:, None]
    map_y = (y_top[None, :] + t * (y_bot - y_top)[None, :]).astype(np.float32)
    map_x = np.tile(np.arange(w, dtype=np.float32), (out_height, 1))
    strip = cv2.remap(crop, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    interp = cv2.INTER_AREA if (strip.shape[1] > out_w or strip.shape[0] > out_h) else cv2.INTER_LINEAR
    return cv2.resize(strip, (out_w, out_h), interpolation=interp)


def split_two_rows_v2(crop, cut, overlap=0.03, out_w=94, out_h=24):
    """Tách hai dòng theo chuẩn LPRNet v5 / CRNN v2: nắn thẳng từng hàng về 94x24 rồi ghép ngang thành 188x24."""
    h, w = crop.shape[:2]
    boundary = np.full(w, float(cut), np.float32) if np.ndim(cut) == 0 else np.asarray(cut, np.float32)
    if boundary.shape != (w,) or not np.isfinite(boundary).all() or not ((boundary > 1) & (boundary < h - 1)).all():
        raise ValueError("Đường phân cách không hợp lệ")
    ov = overlap * h
    top = row_strip(crop, np.zeros(w, np.float32), np.minimum(boundary + ov, h - 1), out_w, out_h)
    bot = row_strip(crop, np.maximum(boundary - ov, 0), np.full(w, h - 1, np.float32), out_w, out_h)
    return np.concatenate((top, bot), axis=1)


def merge_rows(crop, cut, height):
    """Phương pháp ghép hàng của checkpoint CRNN v1 (giữ khoảng cách gap 127)."""
    h, w = crop.shape[:2]
    boundary = np.full(w, float(cut), np.float32) if np.ndim(cut) == 0 else np.asarray(cut, np.float32)
    if boundary.shape != (w,) or not np.isfinite(boundary).all() or not ((boundary > 1) & (boundary < h - 1)).all():
        raise ValueError("Đường phân cách không hợp lệ")
    xx = np.tile(np.arange(w, dtype=np.float32), (height, 1))
    t = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    upper = t * (boundary[None, :] - 1)
    lower = boundary[None, :] + t * (h - 1 - boundary[None, :])
    lines = []
    for yy, mean_h in ((upper, boundary.mean()), (lower, h - boundary.mean())):
        strip = cv2.remap(crop, xx, yy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        new_w = max(1, round(w * height / mean_h))
        lines.append(cv2.resize(strip, (new_w, height), interpolation=cv2.INTER_LINEAR))
    gap = np.full((height, max(2, height // 8), 3), 127, np.uint8)
    return np.concatenate([lines[0], gap, lines[1]], axis=1)


def letterbox(line, width, height):
    h, w = line.shape[:2]
    scale = min(height / h, width / w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(line, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    out = np.full((height, width, 3), 127, np.uint8)
    y = (height - nh) // 2
    out[y:y + nh, :nw] = resized
    return out, nw, nh


@dataclass(frozen=True)
class OCRConfig:
    height: int
    width: int
    margin: Union[Tuple[float, float], float]  # (pad_x, pad_y) or float
    charset: str
    layout_size: int
    lpr_width: int = 94
    lpr_height: int = 24
    split_overlap: float = 0.03
    version: str = "v2"

    @classmethod
    def from_checkpoints(cls, crnn_state, layout_state):
        is_v1_crnn = (
            isinstance(crnn_state.get("identity"), dict)
            and "preprocess" in crnn_state.get("identity", {})
        )

        if is_v1_crnn:
            p = crnn_state["identity"]["preprocess"]
            if p.get("normalize") != "RGB [-1,1]" or p.get("bbox_mode") != "xyxy":
                raise ValueError("Unexpected OCR normalization/bbox convention in v1 CRNN")
            height = int(p.get("height", 32))
            width = int(p.get("width", 256))
            charset = crnn_state.get("charset") or p.get("charset") or CRNN_V1_CHARSET
            margin = float(p.get("margin", 0.03))
            crnn_version = "v1"
        else:
            # v2 CRNN (trained with LPRNet v5 split and 31-character set)
            num_classes = crnn_state["model"]["head.weight"].shape[0]
            height = 32
            width = 256
            if num_classes == 32:
                charset = CRNN_V2_CHARSET
            elif num_classes == 37:
                charset = CRNN_V1_CHARSET
            else:
                charset = crnn_state.get("charset", CRNN_V2_CHARSET)
            margin = (0.03, 0.06)  # PAD_X = 0.03, PAD_Y = 0.06
            crnn_version = "v2"

        if isinstance(layout_state.get("identity"), dict):
            layout_size = int(layout_state["identity"].get("size", 128))
        else:
            layout_size = 128

        return cls(
            height=height,
            width=width,
            margin=margin,
            charset=charset,
            layout_size=layout_size,
            lpr_width=94,
            lpr_height=24,
            split_overlap=0.03,
            version=crnn_version,
        )


def crop_from_box(rgb, bbox, margin):
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(f"Invalid clipped detection bbox: {bbox}")
    if isinstance(margin, (tuple, list)):
        mx = round((x2 - x1) * float(margin[0]))
        my = round((y2 - y1) * float(margin[1]))
    else:
        mx = round((x2 - x1) * float(margin))
        my = round((y2 - y1) * float(margin))
    box = (max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my))
    return rgb[box[1]:box[3], box[0]:box[2]].copy(), box


@dataclass
class OCRReading:
    text: str = ""
    layout: str = ""
    layout_confidence: Optional[float] = None
    split_ok: bool = False
    split_method: str = ""
    error: str = ""
    debug_images: dict = field(default_factory=dict, repr=False)

    def as_dict(self):
        return {key: getattr(self, key) for key in (
            "text", "layout", "layout_confidence", "split_ok", "split_method", "error"
        )}


class PlateReader:
    classes = ("one_row", "two_rows")

    def __init__(self, crnn_path, layout_path, device):
        self.device = torch.device(device)
        crnn_state = load_checkpoint(crnn_path)
        layout_state = load_checkpoint(layout_path)
        self.config = OCRConfig.from_checkpoints(crnn_state, layout_state)
        hidden = int(crnn_state["model"]["rnn.weight_hh_l0"].shape[1])
        num_classes = crnn_state["model"]["head.weight"].shape[0]
        self.crnn = CRNN(num_classes, hidden=hidden)
        self.crnn.load_state_dict(crnn_state["model"], strict=True)
        self.crnn.to(self.device).float().eval()

        self.layout_model = mobilenet_v3_small(weights=None)
        self.layout_model.classifier[-1] = nn.Linear(self.layout_model.classifier[-1].in_features, 2)
        self.layout_model.load_state_dict(layout_state["model"], strict=True)
        self.layout_model.to(self.device).float().eval()

        self.idx_to_char = {i + 1: c for i, c in enumerate(self.config.charset)}
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        self.epochs = {"crnn": crnn_state.get("epoch"), "layout": layout_state.get("epoch")}

    def layout_tensor(self, crop):
        image = Image.fromarray(crop).convert("RGB")
        w, h = image.size
        size = self.config.layout_size
        scale = min(size / w, size / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        image = image.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (size, size), (127, 127, 127))
        canvas.paste(image, ((size - nw) // 2, (size - nh) // 2))
        tensor = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.0
        return (tensor - self.mean) / self.std

    def ocr_tensor(self, crop, layout):
        if self.config.version == "v2":
            if layout == "two_rows":
                cut, ok, method, _, _ = row_separator(crop)
                line = split_two_rows_v2(
                    crop, cut,
                    overlap=self.config.split_overlap,
                    out_w=self.config.lpr_width,
                    out_h=self.config.lpr_height
                )
            elif layout == "one_row":
                interp = (
                    cv2.INTER_AREA
                    if (crop.shape[1] > self.config.lpr_width or crop.shape[0] > self.config.lpr_height)
                    else cv2.INTER_LINEAR
                )
                line = cv2.resize(crop, (self.config.lpr_width, self.config.lpr_height), interpolation=interp)
                ok, method = True, "one_row"
            else:
                raise ValueError(f"Unsupported layout: {layout}")

            image, _, _ = letterbox(line, self.config.width, self.config.height)
            # v2 CRNN was trained with BGR inputs from cv2.imread
            bgr_image = image[:, :, ::-1] if image.shape[2] == 3 else image
            tensor = torch.from_numpy(np.ascontiguousarray(bgr_image)).permute(2, 0, 1).float() / 127.5 - 1.0
            return tensor, bool(ok), method, line, image
        else:
            # v1 CRNN
            if layout == "two_rows":
                cut, ok, method, _, _ = row_separator(crop)
                line = merge_rows(crop, cut, self.config.height)
            elif layout == "one_row":
                line, ok, method = crop, True, "one_row"
            else:
                raise ValueError(f"Unsupported layout: {layout}")

            image, _, _ = letterbox(line, self.config.width, self.config.height)
            tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 127.5 - 1.0
            return tensor, bool(ok), method, line, image

    def decode(self, logits):
        predictions = []
        for sequence in logits.argmax(2).transpose(0, 1).cpu().tolist():
            chars, previous = [], None
            for index in sequence:
                if index != 0 and index != previous:
                    chars.append(self.idx_to_char[index])
                previous = index
            predictions.append("".join(chars))
        return predictions

    @torch.inference_mode()
    def read_many(self, crops, debug=False):
        readings = [OCRReading() for _ in crops]
        times = dict(layout_ms=0.0, preprocess_ms=0.0, crnn_decode_ms=0.0)
        if not crops:
            return readings, times
        self.layout_model.eval()
        self.crnn.eval()
        synchronize(self.device)
        start = time.perf_counter()
        inputs = torch.stack([self.layout_tensor(crop) for crop in crops]).to(self.device)
        logits = self.layout_model(inputs)
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("MobileNet produced NaN/Inf")
        confidence, indices = logits.softmax(1).max(1)
        for reading, index, score in zip(readings, indices.cpu().tolist(), confidence.cpu().tolist()):
            reading.layout = self.classes[index]
            reading.layout_confidence = float(score)
        synchronize(self.device)
        after_layout = time.perf_counter()

        jobs = []
        for i, (crop, reading) in enumerate(zip(crops, readings)):
            try:
                tensor, ok, method, line, image = self.ocr_tensor(crop, reading.layout)
                reading.split_ok, reading.split_method = ok, method
                jobs.append((i, tensor))
                if debug:
                    reading.debug_images = {"crop_rgb": crop, "line_rgb": line, "ocr_input_rgb": image}
            except (ValueError, cv2.error) as exc:
                reading.error = str(exc)
        ocr_inputs = torch.stack([tensor for _, tensor in jobs]).to(self.device) if jobs else None
        synchronize(self.device)
        after_preprocess = time.perf_counter()

        if jobs:
            ocr_logits = self.crnn(ocr_inputs)
            if not bool(torch.isfinite(ocr_logits).all()):
                raise RuntimeError("CRNN produced NaN/Inf")
            for (i, _), text in zip(jobs, self.decode(ocr_logits)):
                readings[i].text = text
        synchronize(self.device)
        end = time.perf_counter()
        times.update(
            layout_ms=(after_layout - start) * 1000,
            preprocess_ms=(after_preprocess - after_layout) * 1000,
            crnn_decode_ms=(end - after_preprocess) * 1000,
        )
        return readings, times
