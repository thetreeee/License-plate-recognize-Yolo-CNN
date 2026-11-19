"""Select one unused RGB crop per visible identity between OCR attempts."""
from dataclasses import dataclass
from math import log1p

import cv2
import numpy as np


def crop_quality(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    h, w = gray.shape
    interior = gray[h // 10:h - h // 10, w // 10:w - w // 10]
    normalized = cv2.resize(interior, (128, 64), interpolation=cv2.INTER_AREA)
    normalized = cv2.GaussianBlur(normalized, (3, 3), 0)
    sharpness = float(cv2.Laplacian(normalized, cv2.CV_64F).var())
    # shortcut: image quality is a heuristic; calibrate against labeled OCR results.
    quality = min(h, w, 96) * log1p(sharpness)
    return blur, quality


@dataclass(frozen=True)
class CropCandidate:
    image: np.ndarray
    frame: int
    quality: float


class BestCrops:
    def __init__(self, max_age):
        if max_age < 1:
            raise ValueError("Crop max_age must be positive")
        self.max_age = max_age
        self.pending = {}
        self.used_frames = {}

    def retain(self, identities, frame):
        visible = set(identities)
        self.pending = {identity: crop for identity, crop in self.pending.items()
                        if identity in visible and frame - crop.frame <= self.max_age}
        self.used_frames = {identity: used for identity, used in self.used_frames.items()
                            if identity in visible}

    def offer(self, identity, frame, crop, quality):
        if frame <= self.used_frames.get(identity, -1):
            return
        previous = self.pending.get(identity)
        if previous is None or quality >= previous.quality:
            self.pending[identity] = CropCandidate(crop.copy(), frame, quality)

    def consume(self, identity):
        crop = self.pending.pop(identity)
        self.used_frames[identity] = crop.frame
        return crop
