"""MobileNet layout classification and batched OCR with the team's LPRNet v5."""
from hashlib import sha256
from time import perf_counter
from types import SimpleNamespace

import cv2
import torch

from scripts import lprnet_v5 as lpr
from scripts.lpr_models import LayoutReader, check_weight_file, load_checkpoint, synchronize


class LPRReader(LayoutReader):
    name = "LPRNet v5"

    def __init__(self, weights, layout_weights, device):
        layout = load_checkpoint(layout_weights)
        identity = layout["identity"]
        if identity["classes"] != {"one_row": 0, "two_rows": 1} or identity["bbox_mode"] != "xyxy":
            raise ValueError("Unexpected MobileNet layout checkpoint")
        self.config = SimpleNamespace(margin=float(identity["crop_margin"]), layout_size=int(identity["size"]),
                                      height=lpr.IMG_H, width=lpr.IMG_W, wide_width=lpr.IMG_W2,
                                      charset="".join(lpr.CHARS))
        super().__init__(layout, device)
        weights = check_weight_file(weights)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if isinstance(state.get("model"), dict):
            state = state["model"]
        self.net = lpr.LPRNet()
        self.net.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
        self.net.to(self.device).eval()
        self.grammar = lpr.PlateGrammar()
        self.provenance = dict(source_branch="origin/HoangAnh_Feature",
                               source_commit="2bae356e7d6782c95cd99982870656738d1d1312",
                               weights_sha256=sha256(weights.read_bytes()).hexdigest(), grammar="generic")

    @torch.inference_mode()
    def read_many(self, crops):
        if not crops:
            return [], dict(layout_ms=0., preprocess_ms=0., lprnet_decode_ms=0.)
        synchronize(self.device)
        start = perf_counter()
        readings = self.classify_layouts(crops)
        after_layout = perf_counter()
        jobs = {False: [], True: []}
        for i, (crop, reading) in enumerate(zip(crops, readings)):
            two_rows = reading.layout == "two_rows"
            try:
                tensor = lpr.preprocess(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), two_rows)
            except (ValueError, cv2.error) as error:
                reading.error = str(error)
                continue
            jobs[two_rows].append((i, tensor))
        after_preprocess = perf_counter()
        for two_rows, batch in jobs.items():
            if not batch:
                continue
            logits = self.net(torch.cat([tensor for _, tensor in batch]).to(self.device))
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("LPRNet produced NaN/Inf")
            log_probs = logits.log_softmax(2).cpu()
            for j, (i, _) in enumerate(batch):
                text = lpr.decode_constrained(log_probs[:, j].numpy(), two_rows, self.grammar)
                readings[i].text = text or lpr.fix_vn_plate(lpr.decode_greedy(log_probs[:, j:j+1]))
                readings[i].split_ok = True
                readings[i].split_method = "lprnet_sloped" if two_rows else "one_row"
        synchronize(self.device)
        end = perf_counter()
        return readings, dict(layout_ms=(after_layout-start)*1000,
                              preprocess_ms=(after_preprocess-after_layout)*1000,
                              lprnet_decode_ms=(end-after_preprocess)*1000)
