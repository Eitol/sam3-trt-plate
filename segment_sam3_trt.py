#!/usr/bin/env python3
"""Runner SAM3 'license plate' via TensorRT FP16 (crop-1008) para re-segmentación de patentes.
Por imagen: recibe bbox de patente (rf-detr), saca un crop 1008 centrado, corre SAM3 TRT y
devuelve la MÁSCARA de patente precisa (RLE pycocotools) en coords de la imagen completa.

Engine: construido con la API nativa de TensorRT (NO onnxruntime TRT-EP: su optimizador de grafo
corrompe la atención windowed con RoPE de SAM3 -> usar el engine de trt_api.py).
Prompt 'license plate' horneado en el ONNX. Input fijo 1x3x1008x1008.
"""
import os, sys, json, time, glob, argparse
import numpy as np

class Sam3PlateTRT:
    S = 1008
    def __init__(self, engine_path, hf_token=None, score_thr=0.5):
        if hf_token: os.environ.setdefault("HF_TOKEN", hf_token); os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
        os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
        import tensorrt as trt, torch
        from transformers.models.sam3 import Sam3Processor
        self.torch = torch; self.trt = trt
        self.score_thr = score_thr
        log = trt.Logger(trt.Logger.ERROR)
        self.eng = trt.Runtime(log).deserialize_cuda_engine(open(engine_path, "rb").read())
        self.ctx = self.eng.create_execution_context()
        self.T2T = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
                    trt.int64: torch.int64, trt.bool: torch.bool}
        ios = [self.eng.get_tensor_name(i) for i in range(self.eng.num_io_tensors)]
        self.ins = [n for n in ios if self.eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outs = [n for n in ios if self.eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.obuf = {n: torch.empty(tuple(self.eng.get_tensor_shape(n)),
                     dtype=self.T2T[self.eng.get_tensor_dtype(n)], device="cuda") for n in self.outs}
        for n in self.outs: self.ctx.set_tensor_address(n, self.obuf[n].data_ptr())
        self.indt = self.T2T[self.eng.get_tensor_dtype(self.ins[0])]
        self.proc = Sam3Processor.from_pretrained("facebook/sam3")

    def _crop1008(self, img, pb, W, H):
        S = self.S
        cx = (pb[0] + pb[2]) // 2; cy = (pb[1] + pb[3]) // 2
        x1 = max(0, min(max(0, W - S), int(cx - S // 2))); y1 = max(0, min(max(0, H - S), int(cy - S // 2)))
        cr = img.crop((x1, y1, min(W, x1 + S), min(H, y1 + S)))
        cw, ch = cr.size
        if (cw, ch) != (S, S):  # imagen < 1008: pad arriba-izquierda (igual que el ref)
            from PIL import Image
            cv = Image.new("RGB", (S, S), (0, 0, 0)); cv.paste(cr, (0, 0)); cr = cv
        return cr, x1, y1, cw, ch

    @staticmethod
    def _iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        i = max(0, x2 - x1) * max(0, y2 - y1); ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
        return i / ua if ua > 0 else 0.0

    def segment_plate(self, img, plate_bbox):
        """img: PIL RGB; plate_bbox: [x1,y1,x2,y2] en coords de imagen completa.
        Devuelve dict(mask(bool HxW full), bbox_xyxy(full), score) o None si no detecta."""
        torch = self.torch
        W, H = img.size
        pb = [float(v) for v in plate_bbox]
        cr, ox, oy, cw, ch = self._crop1008(img, pb, W, H)
        pl = [pb[0]-ox, pb[1]-oy, pb[2]-ox, pb[3]-oy]   # bbox patente en coords del crop
        pv = self.proc(images=cr, text="license plate", return_tensors="pt")["pixel_values"]
        inp = pv.to("cuda", dtype=self.indt).contiguous(); self.ctx.set_tensor_address(self.ins[0], inp.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        logits = self.obuf["pred_logits"][0]; presence = self.obuf["presence_logits"][0]
        scores = (logits.sigmoid() * presence.sigmoid()).float()          # (200,)
        boxes = self.obuf["pred_boxes"][0].float() * self.S               # xyxy en px del crop
        keep = (scores > self.score_thr).nonzero(as_tuple=True)[0]
        if keep.numel() == 0:
            torch.cuda.synchronize(); return None
        bx = boxes[keep].detach().cpu().numpy()
        ious = np.array([self._iou(bx[j], pl) for j in range(len(bx))])
        sel = int(keep[int(ious.argmax())].item()) if ious.max() > 0 else int(keep[int(scores[keep].argmax().item())].item())
        m = self.obuf["pred_masks"][0, sel].float().sigmoid()[None, None]  # (1,1,288,288)
        m = torch.nn.functional.interpolate(m, size=(self.S, self.S), mode="bilinear", align_corners=False)[0, 0]
        mb = (m > 0.5).detach().cpu().numpy()                              # (1008,1008) crop space
        torch.cuda.synchronize()
        full = np.zeros((H, W), dtype=bool)
        mb = mb[:ch, :cw]                                                  # quitar padding si lo hubo
        full[oy:oy+ch, ox:ox+cw] = mb
        bsel = boxes[sel].detach().cpu().numpy()
        box_full = [float(bsel[0]+ox), float(bsel[1]+oy), float(bsel[2]+ox), float(bsel[3]+oy)]
        return {"mask": full, "bbox_xyxy": box_full, "score": float(scores[sel].item())}


def rle_encode(mask):
    """Mascara bool/uint8 (H,W) -> RLE pycocotools (counts decodeado a str), igual que segment_sam3.py."""
    from pycocotools import mask as cmask
    r = cmask.encode(np.asfortranarray(mask.astype(np.uint8)))
    r["counts"] = r["counts"].decode()
    return r


def write_sidecar(path, res, W, H, suffix=".sam3trt.seg.json"):
    """Escribe el sidecar con el MISMO schema que rf-detr/sam3 (klass license_plate, source sam3-trt)."""
    from pathlib import Path
    masks = []
    if res is not None:
        masks.append({"instance_id": "p0", "klass": "license_plate", "sub_type": None,
                      "prompt": "license plate", "is_target": True, "score": round(res["score"], 4),
                      "bbox_xyxy": [round(x, 1) for x in res["bbox_xyxy"]],
                      "rle": rle_encode(res["mask"])})
    side = str(Path(path).with_suffix("")) + suffix; tmp = side + ".tmp"
    json.dump({"coordinate_space": "fullframe", "width": W, "height": H,
               "source": "sam3-trt", "masks": masks}, open(tmp, "w"))
    os.replace(tmp, side)
    return side


def main():
    ap = argparse.ArgumentParser(description="Re-segmenta patentes (crop-1008) con SAM3 TensorRT FP16.")
    ap.add_argument("--engine", required=True, help="engine TRT FP16 (build_sam3_trt_engine.py)")
    ap.add_argument("--manifest", required=True,
                    help="JSON: [[img_path, [x1,y1,x2,y2]], ...] (bbox patente de rf-detr/parquet)")
    ap.add_argument("--suffix", default=".sam3trt.seg.json")
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--min-side", type=int, default=1500,
                    help="solo re-segmentar si max(W,H) >= esto (HD+); abajo se omite (ya hay resolucion)")
    ap.add_argument("--progress-every", type=int, default=200)
    a = ap.parse_args()
    from PIL import Image
    items = json.load(open(a.manifest))
    R = Sam3PlateTRT(a.engine, hf_token=a.hf_token, score_thr=a.score_thr)
    t0 = time.time(); done = skip = err = 0
    for i, (ip, bbox) in enumerate(items, 1):
        try:
            img = Image.open(ip).convert("RGB"); W, H = img.size
            if max(W, H) < a.min_side:
                skip += 1; continue
            res = R.segment_plate(img, bbox)
            write_sidecar(ip, res, W, H, a.suffix); done += 1
        except Exception as e:
            err += 1
            if err <= 15: print(f"  ERR {ip}: {type(e).__name__}: {e}", flush=True)
        if i % a.progress_every == 0:
            el = time.time() - t0
            print(f"[{i}/{len(items)}] done={done} skip={skip} err={err} | {done/max(el,1e-9):.2f} img/s", flush=True)
    print(f"FIN: done={done} skip={skip} err={err} en {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
