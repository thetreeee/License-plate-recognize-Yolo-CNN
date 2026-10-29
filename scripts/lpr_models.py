"""Checkpoint-compatible OCR models and preprocessing for the VNLP notebooks.

Input crops are RGB uint8. No filename labels, ground-truth layout, or cached
OCR images are used by PlateReader at inference time.
"""
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


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
    def __init__(self,num_classes,hidden=256):
        super().__init__()
        def block(cin,cout):
            return [nn.Conv2d(cin,cout,3,padding=1,bias=False),
                    nn.BatchNorm2d(cout),nn.ReLU(inplace=True)]
        self.cnn=nn.Sequential(
            *block(3,64), nn.MaxPool2d(2,2),
            *block(64,128), nn.MaxPool2d(2,2),
            *block(128,256), *block(256,256), nn.MaxPool2d((2,1),(2,1)),
            *block(256,512), *block(512,512), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(512,512,(2,1),bias=False),nn.BatchNorm2d(512),nn.ReLU(inplace=True))
        self.rnn=nn.LSTM(512,hidden,num_layers=2,batch_first=True,
                         bidirectional=True,dropout=.1)
        self.head=nn.Linear(hidden*2,num_classes)
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
        elif isinstance(m,nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self,x):
        f=self.cnn(x)
        assert f.shape[2]==1, f'Feature height must be 1: {f.shape}'
        seq=f.squeeze(2).transpose(1,2)
        seq,_=self.rnn(seq)
        return self.head(seq).transpose(0,1)  # logits [T,B,C], log_softmax ngoài autocast


def row_separator(crop):
    # Tìm một đường phân cách có thể nghiêng; không giả định khe ngang ở h/2.
    h,w = crop.shape[:2]
    if h < 12 or w < 12:
        return np.full(w,h/2,dtype=np.float32), False, 'too_small', np.zeros(h), np.zeros((h,w),np.uint8)
    sh=96; sw=min(256,max(64,round(w*sh/h)))
    gray=cv2.cvtColor(cv2.resize(crop,(sw,sh)),cv2.COLOR_RGB2GRAY)
    gray=cv2.GaussianBlur(gray,(3,3),0)
    mask=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV,31,7)
    ink=(mask>0).astype(np.float32)
    # Bỏ hai mép dọc có viền/ốc vít khi chấm điểm; giữ nguyên ảnh RGB khi OCR.
    xs=np.arange(round(.12*sw),round(.88*sw))
    centers=np.arange(round(.30*sh),round(.70*sh)+1)
    drifts=np.linspace(-.35*sh,.35*sh,15)
    cc,dd=np.meshgrid(centers,drifts,indexing='ij')
    cc=cc.ravel(); dd=dd.ravel()
    ys=cc[:,None]+dd[:,None]*(xs[None,:]/max(1,sw-1)-.5)
    def line_ink(y_values):
        values=[]
        for band in (-1,0,1):
            yy=np.clip(np.rint(y_values+band).astype(int),0,sh-1)
            values.append(ink[yy,xs[None,:]].mean(axis=1))
        return np.mean(values,axis=0)
    gap_ink=line_ink(ys)
    # Nhẹ nhàng ưu tiên vùng giữa và ít nghiêng khi nhiều khe có cùng mức mực.
    scores=gap_ink+.04*np.abs(cc/sh-.5)+.003*np.abs(dd)/(.35*sh)
    best=int(np.argmin(scores))
    # Hai phía của khe phải có nét chữ; độ tin cậy chỉ là heuristic để audit.
    top=max(float(line_ink(ys[best:best+1]-offset*sh)[0]) for offset in (.12,.18,.24))
    bottom=max(float(line_ink(ys[best:best+1]+offset*sh)[0]) for offset in (.12,.18,.24))
    ratio=float(gap_ink[best])/max(1e-6,min(top,bottom))
    ok=min(top,bottom)>.03 and ratio<.65
    cut=(cc[best]+dd[best]*(np.arange(w)/max(1,w-1)-.5))*(h/sh)
    cut=np.clip(cut,2,h-2).astype(np.float32)
    projection=ink.mean(axis=1)
    projection=cv2.resize(projection[:,None],(1,h)).ravel()
    debug_mask=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
    # Không chắc chắn vẫn dùng ứng viên tốt nhất, gắn cờ; không đổi về h//2.
    return cut,bool(ok),('sloped_gap' if ok else 'low_confidence_candidate'),projection,debug_mask

def merge_rows(crop, cut, height):
    h,w=crop.shape[:2]
    boundary=np.full(w,float(cut),np.float32) if np.ndim(cut)==0 else np.asarray(cut,np.float32)
    if boundary.shape!=(w,) or not np.isfinite(boundary).all() or not ((boundary>1)&(boundary<h-1)).all():
        raise ValueError('Đường phân cách không hợp lệ')
    # Remap từng cột: giữ hai phía đường nghiêng rồi chuẩn hóa chiều cao hàng.
    # Đây không phải phép homography bốn góc; góc nhìn phối cảnh mạnh vẫn cần xử lý riêng.
    xx=np.tile(np.arange(w,dtype=np.float32),(height,1))
    t=np.linspace(0,1,height,dtype=np.float32)[:,None]
    upper=t*(boundary[None,:]-1)
    lower=boundary[None,:]+t*(h-1-boundary[None,:])
    lines=[]
    for yy,mean_h in ((upper,boundary.mean()),(lower,h-boundary.mean())):
        strip=cv2.remap(crop,xx,yy.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        new_w=max(1,round(w*height/mean_h))
        lines.append(cv2.resize(strip,(new_w,height),interpolation=cv2.INTER_LINEAR))
    gap=np.full((height,max(2,height//8),3),127,np.uint8)
    return np.concatenate([lines[0],gap,lines[1]],axis=1)

def letterbox(line, width, height):
    h,w=line.shape[:2]
    scale=min(height/h, width/w)
    nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    resized=cv2.resize(line,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    out=np.full((height,width,3),127,np.uint8)
    y=(height-nh)//2
    out[y:y+nh,:nw]=resized
    return out, nw, nh


@dataclass(frozen=True)
class OCRConfig:
    height: int
    width: int
    margin: float
    charset: str
    layout_size: int

    @classmethod
    def from_checkpoints(cls, crnn_state, layout_state):
        p = crnn_state["identity"]["preprocess"]
        layout = layout_state["identity"]
        if p["two_rows"] != "sloped_gap_v2_flagged_candidate":
            raise ValueError("Unsupported row preprocessing version in CRNN checkpoint")
        if p["normalize"] != "RGB [-1,1]" or p["bbox_mode"] != "xyxy":
            raise ValueError("Unexpected OCR normalization/bbox convention")
        if int(p["height"]) != 32 or int(p["width"]) % 4:
            raise ValueError("Checkpoint input size is incompatible with this CRNN")
        if layout["classes"] != {"one_row": 0, "two_rows": 1}:
            raise ValueError("Unexpected layout class mapping")
        if layout["bbox_mode"] != "xyxy" or abs(float(layout["crop_margin"]) - float(p["margin"])) > 1e-9:
            raise ValueError("MobileNet and CRNN were trained with different crop settings")
        charset = crnn_state["charset"]
        if not charset or charset != p["charset"] or len(set(charset)) != len(charset):
            raise ValueError("Unexpected checkpoint character dictionary")
        return cls(int(p["height"]), int(p["width"]), float(p["margin"]),
                   charset, int(layout["size"]))


def crop_from_box(rgb, bbox, margin):
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(f"Invalid clipped detection bbox: {bbox}")
    mx, my = round((x2-x1)*margin), round((y2-y1)*margin)
    box = (max(0,x1-mx), max(0,y1-my), min(w,x2+mx), min(h,y2+my))
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
        # Image arrays are saved separately rather than serialized into CSV/JSON.
        return {key: getattr(self,key) for key in (
            "text","layout","layout_confidence","split_ok","split_method","error"
        )}


class PlateReader:
    classes = ("one_row", "two_rows")

    def __init__(self, crnn_path, layout_path, device):
        self.device = torch.device(device)
        crnn_state, layout_state = load_checkpoint(crnn_path), load_checkpoint(layout_path)
        self.config = OCRConfig.from_checkpoints(crnn_state, layout_state)
        hidden = int(crnn_state["model"]["rnn.weight_hh_l0"].shape[1])
        self.crnn = CRNN(len(self.config.charset)+1, hidden=hidden)
        self.crnn.load_state_dict(crnn_state["model"], strict=True)
        self.crnn.to(self.device).float().eval()
        self.layout_model = mobilenet_v3_small(weights=None)
        self.layout_model.classifier[-1] = nn.Linear(self.layout_model.classifier[-1].in_features,2)
        self.layout_model.load_state_dict(layout_state["model"], strict=True)
        self.layout_model.to(self.device).float().eval()
        self.idx_to_char = {i+1:c for i,c in enumerate(self.config.charset)}
        self.mean = torch.tensor([.485,.456,.406])[:,None,None]
        self.std = torch.tensor([.229,.224,.225])[:,None,None]
        self.epochs = {"crnn":crnn_state["epoch"], "layout":layout_state["epoch"]}

    def layout_tensor(self, crop):
        image = Image.fromarray(crop).convert("RGB")
        w,h = image.size
        size = self.config.layout_size
        scale = min(size/w,size/h)
        nw,nh = max(1,round(w*scale)),max(1,round(h*scale))
        image = image.resize((nw,nh),Image.Resampling.BILINEAR)
        canvas = Image.new("RGB",(size,size),(127,127,127))
        canvas.paste(image,((size-nw)//2,(size-nh)//2))
        tensor = torch.from_numpy(np.array(canvas)).permute(2,0,1).float()/255
        return (tensor-self.mean)/self.std

    def ocr_tensor(self, crop, layout):
        if layout == "two_rows":
            cut, ok, method, _, _ = row_separator(crop)
            line = merge_rows(crop,cut,self.config.height)
        elif layout == "one_row":
            line,ok,method = crop,True,"one_row"
        else:
            raise ValueError(f"Unsupported layout: {layout}")
        image,_,_ = letterbox(line,self.config.width,self.config.height)
        tensor = torch.from_numpy(image.copy()).permute(2,0,1).float()/127.5-1
        return tensor,bool(ok),method,line,image

    def decode(self, logits):
        predictions = []
        for sequence in logits.argmax(2).transpose(0,1).cpu().tolist():
            chars,previous = [],None
            for index in sequence:
                if index != 0 and index != previous:
                    chars.append(self.idx_to_char[index])
                previous = index
            predictions.append("".join(chars))
        return predictions

    @torch.inference_mode()
    def read_many(self, crops, debug=False):
        readings = [OCRReading() for _ in crops]
        times = dict(layout_ms=0.,preprocess_ms=0.,crnn_decode_ms=0.)
        if not crops:
            return readings,times
        self.layout_model.eval()
        self.crnn.eval()
        synchronize(self.device)
        start = time.perf_counter()
        inputs = torch.stack([self.layout_tensor(crop) for crop in crops]).to(self.device)
        logits = self.layout_model(inputs)
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("MobileNet produced NaN/Inf")
        confidence,indices = logits.softmax(1).max(1)
        for reading,index,score in zip(readings,indices.cpu().tolist(),confidence.cpu().tolist()):
            reading.layout = self.classes[index]
            reading.layout_confidence = float(score)
        synchronize(self.device)
        after_layout = time.perf_counter()

        jobs = []
        for i,(crop,reading) in enumerate(zip(crops,readings)):
            try:
                tensor,ok,method,line,image = self.ocr_tensor(crop,reading.layout)
                reading.split_ok,reading.split_method = ok,method
                jobs.append((i,tensor))
                if debug:
                    reading.debug_images = {"crop_rgb":crop,"line_rgb":line,"ocr_input_rgb":image}
            except (ValueError,cv2.error) as exc:
                reading.error = str(exc)
        ocr_inputs = torch.stack([tensor for _,tensor in jobs]).to(self.device) if jobs else None
        synchronize(self.device)
        after_preprocess = time.perf_counter()

        if jobs:
            ocr_logits = self.crnn(ocr_inputs)
            if not bool(torch.isfinite(ocr_logits).all()):
                raise RuntimeError("CRNN produced NaN/Inf")
            for (i,_),text in zip(jobs,self.decode(ocr_logits)):
                readings[i].text = text
        synchronize(self.device)
        end = time.perf_counter()
        times.update(
            layout_ms=(after_layout-start)*1000,
            preprocess_ms=(after_preprocess-after_layout)*1000,
            crnn_decode_ms=(end-after_preprocess)*1000,
        )
        return readings,times
