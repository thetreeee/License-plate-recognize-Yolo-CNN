"""Suy luận LPRNet v5 (biển số VN) độc lập, không cần notebook.

Ví dụ:
    python lprnet_v5_infer.py --weights lprnet_v5/best_lprnet_vn_v5.pth --image xe.jpg --bbox 120 340 260 420
    python lprnet_v5_infer.py --weights lprnet_v5/best_lprnet_vn_v5.pth --image bien_da_crop.jpg

Tiền xử lý, kiến trúc và giải mã giống hệt notebook VN_ALPR_LPRNet_v5_TrainReview.ipynb (chế độ đánh giá).
"""
import argparse
import functools
import json
import math
import os
import re
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Cấu hình (phải giống lúc train) ---
CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M',
         'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z', 'R']   # 31 ký tự + blank = 32 lớp
BLANK_IDX = len(CHARS)
NUM_CLASSES = len(CHARS) + 1
IMG_W, IMG_H = 94, 24          # một dòng
IMG_W2 = 2 * IMG_W             # biển 2 dòng: dòng trên | dòng dưới -> 188x24
PAD_X, PAD_Y = 0.03, 0.06      # đệm quanh bbox (tỉ lệ theo kích thước bbox)
SPLIT_OVERLAP = 0.03           # mỗi dòng lấn qua đường tách 3% chiều cao
TWO_ROW_RATIO_THR = 1.50       # rộng/cao bbox < ngưỡng -> biển 2 dòng (Bước 4.3 của notebook, đúng 95.97% mẫu)
BEAM_WIDTH = 10

VN_PLATE_RE = re.compile(r"^[0-9]{2}[A-Z]{1,2}[0-9]{4,6}$")
TO_DIGIT = {"B": "8", "D": "0", "S": "5", "Z": "2", "G": "6", "A": "4"}
TO_LETTER = {v: k for k, v in TO_DIGIT.items()}


# ============================ Kiến trúc ============================
class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class WidthBatchNorm2d(nn.BatchNorm2d):
    """Thống kê chạy riêng cho ảnh rộng 94 (1 dòng) và 188 (2 dòng), gamma/beta dùng chung."""
    def __init__(self, num_features):
        super().__init__(num_features)
        self.register_buffer("running_mean_wide", torch.zeros(num_features))
        self.register_buffer("running_var_wide", torch.ones(num_features))
        self.wide = False

    def forward(self, x):
        if not self.wide:
            return super().forward(x)
        base_mean, base_var = self.running_mean, self.running_var
        self.running_mean, self.running_var = self.running_mean_wide, self.running_var_wide
        try:
            return super().forward(x)
        finally:
            self.running_mean_wide, self.running_var_wide = self.running_mean, self.running_var
            self.running_mean, self.running_var = base_mean, base_var


class LPRNet(nn.Module):
    KEEP_IDX = (2, 6, 13, 22)

    def __init__(self, class_num=NUM_CLASSES, dropout_rate=0.5):
        super().__init__()
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            WidthBatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(64, 128),
            WidthBatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_basic_block(64, 256),
            WidthBatchNorm2d(256),
            nn.ReLU(),
            small_basic_block(256, 256),
            WidthBatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(64, 256, kernel_size=(1, 4), stride=1),
            WidthBatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, class_num, kernel_size=(13, 1), stride=1),
            WidthBatchNorm2d(class_num),
            nn.ReLU(),
        )
        self.container = nn.Sequential(
            nn.Conv2d(448 + class_num, class_num, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, x):
        wide = x.shape[-1] != IMG_W
        for m in self.modules():
            if isinstance(m, WidthBatchNorm2d):
                m.wide = wide
        keep = []
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in self.KEEP_IDX:
                keep.append(x)
        out_hw = keep[-1].shape[-2:]
        context = []
        for i, f in enumerate(keep):
            if i in (0, 1):
                f = F.avg_pool2d(f, kernel_size=5, stride=5)
            elif i == 2:
                f = F.avg_pool2d(f, kernel_size=(4, 10), stride=(4, 2))
            if f.shape[-2:] != out_hw:
                f = F.interpolate(f, size=out_hw, mode="bilinear", align_corners=False)
            f = f / f.pow(2).mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            context.append(f)
        x = self.container(torch.cat(context, 1))
        logits = x.mean(dim=2)            # [B, C, T]
        return logits.permute(2, 0, 1)    # [T, B, C]


def load_model(path, device="cpu"):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and isinstance(sd.get("model"), dict):   # checkpoint "last" có optimizer, history...
        sd = sd["model"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    n_cls = sd["container.0.weight"].shape[0]
    assert n_cls == NUM_CLASSES, f"Checkpoint có {n_cls} lớp, v5 cần {NUM_CLASSES} (checkpoint v2–v4 có 31 lớp)"
    net = LPRNet()
    net.load_state_dict(sd)
    return net.to(device).eval()


# ============================ Tiền xử lý ============================
def resize_to(img, size):
    w, h = size
    interp = cv2.INTER_AREA if (img.shape[1] > w or img.shape[0] > h) else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


def crop_plate(img, bbox):
    xmin, ymin, xmax, ymax = bbox
    bw, bh = xmax - xmin, ymax - ymin
    H, W = img.shape[:2]
    x0, y0 = max(0, int(round(xmin - PAD_X * bw))), max(0, int(round(ymin - PAD_Y * bh)))
    x1, y1 = min(W, int(round(xmax + PAD_X * bw))), min(H, int(round(ymax + PAD_Y * bh)))
    crop = img[y0:y1, x0:x1]
    return crop if crop.size > 0 else img


def find_row_gap(crop):
    """Đường thẳng (có thể nghiêng) giữa 2 dòng chữ, cắt qua ít nét mực nhất: (y mép trái, y mép phải)."""
    h, w = crop.shape[:2]
    if h < 12 or w < 12:
        return h / 2, h / 2
    sh = 96
    sw = min(256, max(64, round(w * sh / h)))
    gray = cv2.cvtColor(cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    ink = (cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7) > 0).astype(np.float32)
    xs = np.arange(round(0.12 * sw), round(0.88 * sw))
    cc, dd = np.meshgrid(np.arange(round(0.30 * sh), round(0.70 * sh) + 1),
                         np.linspace(-0.35 * sh, 0.35 * sh, 15), indexing="ij")
    cc, dd = cc.ravel().astype(np.float32), dd.ravel().astype(np.float32)
    ys = cc[:, None] + dd[:, None] * (xs[None, :] / max(1, sw - 1) - 0.5)
    gap = np.mean([ink[np.clip(np.rint(ys + band).astype(int), 0, sh - 1), xs[None, :]].mean(axis=1)
                   for band in (-1, 0, 1)], axis=0)
    score = gap + 0.04 * np.abs(cc / sh - 0.5) + 0.003 * np.abs(dd) / (0.35 * sh)
    b = int(np.argmin(score))
    y_left, y_right = (cc[b] - dd[b] / 2) * h / sh, (cc[b] + dd[b] / 2) * h / sh
    return float(np.clip(y_left, 2, h - 2)), float(np.clip(y_right, 2, h - 2))


def row_strip(crop, y_top, y_bot):
    """Nắn dải giữa 2 đường y_top, y_bot thành dải thẳng rồi resize về 94x24."""
    w = crop.shape[1]
    out_h = max(2, int(round(float(np.mean(y_bot - y_top)))))
    t = np.linspace(0, 1, out_h, dtype=np.float32)[:, None]
    map_y = (y_top[None, :] + t * (y_bot - y_top)[None, :]).astype(np.float32)
    map_x = np.tile(np.arange(w, dtype=np.float32), (out_h, 1))
    strip = cv2.remap(crop, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return resize_to(strip, (IMG_W, IMG_H))


def split_two_rows(crop):
    h, w = crop.shape[:2]
    cut = np.linspace(*find_row_gap(crop), w, dtype=np.float32)
    ov = SPLIT_OVERLAP * h
    top = row_strip(crop, np.zeros(w, np.float32), np.minimum(cut + ov, h - 1))
    bot = row_strip(crop, np.maximum(cut - ov, 0), np.full(w, h - 1, np.float32))
    return np.concatenate((top, bot), axis=1)


def preprocess(crop, is_two_rows):
    """Ảnh biển BGR đã crop -> tensor [1, 3, 24, 94] (1 dòng) hoặc [1, 3, 24, 188] (2 dòng)."""
    x = split_two_rows(crop) if is_two_rows else resize_to(crop, (IMG_W, IMG_H))
    x = (x.astype(np.float32) - 127.5) * 0.0078125
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]


# ============================ Giải mã ============================
def decode_greedy(logits):
    path = logits[:, 0].argmax(dim=1).tolist()
    res, prev = [], None
    for idx in path:
        if idx != prev and idx != BLANK_IDX:
            res.append(CHARS[idx])
        prev = idx
    return "".join(res)


def fix_vn_plate(p):
    if len(p) < 7 or VN_PLATE_RE.match(p):
        return p
    c = list(p)
    for i in (0, 1):
        c[i] = TO_DIGIT.get(c[i], c[i])
    c[2] = TO_LETTER.get(c[2], c[2])
    for i in range(len(c) - 4, len(c)):
        c[i] = TO_DIGIT.get(c[i], c[i])
    fixed = "".join(c)
    return fixed if VN_PLATE_RE.match(fixed) else p


class PlateGrammar:
    """Ngữ pháp biển số cho giải mã ràng buộc.
    Có plate_grammar.json (xuất từ notebook): đúng 65 mã tỉnh / 35 seri / số chữ số như lúc đánh giá.
    Không có: ngữ pháp tổng quát (2 số + 1–2 chữ + 4–6 số), vẫn chặn được ký tự thừa và nhầm chữ ↔ số."""
    def __init__(self, path=None):
        letters = [c for c in CHARS if c.isalpha()]
        if path and os.path.exists(path):
            g = json.load(open(path, encoding="utf-8"))
            self.provinces, self.series = set(g["provinces"]), set(g["series"])
            self.digit_lens = {False: set(g["digit_lens"]["one_row"]), True: set(g["digit_lens"]["two_rows"])}
        else:
            self.provinces = {f"{i:02d}" for i in range(100)}
            self.series = set(letters) | {a + b for a in letters for b in letters}
            self.digit_lens = {False: {4, 5, 6}, True: {4, 5, 6}}
        self.series_prefix = {x[:k] for x in self.series for k in range(1, len(x) + 1)}
        self.province_prefix = {x[0] for x in self.provinces}
        self.max_digits = {two: max(v) for two, v in self.digit_lens.items()}
        self.state = functools.lru_cache(maxsize=None)(self._state)

    def _state(self, p, two):
        """None: p không thể là đầu của biển hợp lệ; False: đầu hợp lệ; True: biển hoàn chỉnh."""
        n = len(p)
        if n >= 1 and (not p[0].isdigit() or p[0] not in self.province_prefix):
            return None
        if n >= 2 and p[:2] not in self.provinces:
            return None
        if n <= 2:
            return False
        k = 0
        while 2 + k < n and p[2 + k].isalpha():
            k += 1
        series, digits = p[2:2 + k], p[2 + k:]
        if k == 0 or not digits.isdigit() and digits:
            return None
        if not digits:
            return False if series in self.series_prefix else None
        if series not in self.series or len(digits) > self.max_digits[two]:
            return None
        return len(digits) in self.digit_lens[two]


def logaddexp(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log1p(math.exp(-abs(a - b)))


def decode_constrained(logp, two, grammar, beam=BEAM_WIDTH, topk=6, prune=math.log(1e-3)):
    """Prefix beam search CTC chỉ giữ chuỗi có thể là biển hợp lệ. logp: [T, C] log-softmax (numpy)."""
    beams = {"": (0.0, -math.inf)}
    for row in logp:
        top = np.argpartition(row, -topk)[-topk:]
        cand = [c for c in top if c != BLANK_IDX and row[c] > prune]
        lb = float(row[BLANK_IDX])
        nxt = defaultdict(lambda: [-math.inf, -math.inf])
        for prefix, (pb, pnb) in beams.items():
            tot = logaddexp(pb, pnb)
            e = nxt[prefix]
            e[0] = logaddexp(e[0], tot + lb)
            last = prefix[-1] if prefix else None
            for c in cand:
                ch, lc = CHARS[c], float(row[c])
                if ch == last:
                    e[1] = logaddexp(e[1], pnb + lc)
                    src = pb
                else:
                    src = tot
                new = prefix + ch
                if grammar.state(new, two) is None:
                    continue
                e2 = nxt[new]
                e2[1] = logaddexp(e2[1], src + lc)
        beams = dict(sorted(nxt.items(), key=lambda kv: -logaddexp(*kv[1]))[:beam])
    done = [(logaddexp(*v), k) for k, v in beams.items() if grammar.state(k, two)]
    return max(done)[1] if done else None


# ============================ Pipeline ============================
@torch.no_grad()
def recognize_plate(net, img_bgr, bbox=None, is_two_rows=None, grammar=None, device="cpu"):
    """img_bgr: khung hình gốc (BGR); bbox: (xmin, ymin, xmax, ymax) từ YOLO, None nếu ảnh đã crop sẵn.
    is_two_rows: None -> đoán theo tỉ lệ rộng/cao. Trả về (biển số, is_two_rows)."""
    crop = img_bgr if bbox is None else crop_plate(img_bgr, [int(v) for v in bbox])
    if is_two_rows is None:
        if bbox is None:
            ratio = crop.shape[1] / max(1, crop.shape[0])
        else:
            ratio = (bbox[2] - bbox[0]) / max(1, bbox[3] - bbox[1])
        is_two_rows = ratio < TWO_ROW_RATIO_THR
    logits = net(preprocess(crop, is_two_rows).to(device))
    grammar = grammar or PlateGrammar()
    plate = decode_constrained(logits.log_softmax(2)[:, 0].cpu().numpy(), is_two_rows, grammar)
    return (plate if plate is not None else fix_vn_plate(decode_greedy(logits))), is_two_rows


def main():
    ap = argparse.ArgumentParser(description="Nhận dạng biển số VN bằng LPRNet v5")
    ap.add_argument("--weights", default="lprnet_v5/best_lprnet_vn_v5.pth")
    ap.add_argument("--image", required=True)
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    ap.add_argument("--layout", choices=["auto", "one_row", "two_rows"], default="auto")
    ap.add_argument("--grammar", default="lprnet_v5/plate_grammar.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_model(args.weights, device)
    img = cv2.imread(args.image, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    assert img is not None, f"Không đọc được ảnh: {args.image}"
    two = None if args.layout == "auto" else args.layout == "two_rows"
    plate, two = recognize_plate(net, img, args.bbox, two, PlateGrammar(args.grammar), device)
    print(f"{plate}  ({'2 dòng' if two else '1 dòng'})")


if __name__ == "__main__":
    main()
