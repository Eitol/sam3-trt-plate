#!/usr/bin/env python3
"""Construye el engine TensorRT FP16 de SAM3 'license plate' para el crop pipeline (input fijo 1008).

Dos pasos (correr en una caja con GPU + sam3/transformers + tensorrt 10, p.ej. vastsam):
  1) export-onnx : HF Sam3Model -> ONNX estatico (1x3x1008x1008) con el prompt 'license plate'
                   HORNEADO como buffers constantes (input_ids/attention_mask). Asi el grafo es
                   solo imagen->(masks,boxes,logits,presence) y TRT constant-foldea el text encoder.
  2) build-engine: parsea el ONNX con la API NATIVA de TensorRT y construye un engine FP16.

###############################################################################################
# GOTCHA CRITICO (documentado tras depurar a fondo):                                          #
#   NO usar onnxruntime con TensorrtExecutionProvider (ni CUDAExecutionProvider) con el        #
#   optimizador de grafo por defecto. El optimizador de ORT (niveles BASIC/EXTENDED/ALL)       #
#   CORROMPE la atencion 'windowed' con RoPE del ViT de SAM3 en CUDA -> features rel-error ~1.0 #
#   -> mascaras IoU ~0.76 y detecciones perdidas. (Cada capa por separado exporta perfecta;     #
#   es una fusion cross-layer del optimizador la que rompe.)                                    #
#   FIX A (este script): construir el engine con la API nativa de trt.OnnxParser -> correcto.   #
#   FIX B (si se usa ORT): SessionOptions.graph_optimization_level = ORT_DISABLE_ALL            #
#         (pero entonces TRT-EP no ingiere el grafo: nodos If sin shape -> usar FIX A).         #
#   Validado: TRT FP16 nativo == PyTorch (maskIoU 0.953 mean / 0.964 median, det 60/60).        #
###############################################################################################

Uso:
  HF_TOKEN=hf_xxx python build_sam3_trt_engine.py export-onnx  --onnx /dev/shm/sam3_plate.onnx \
                                                  --ref-img /path/cualquier_imagen.jpg
  LD_LIBRARY_PATH=<venv>/lib/python3.X/site-packages/tensorrt_libs \
  python build_sam3_trt_engine.py build-engine --onnx /dev/shm/sam3_plate.onnx \
                                               --engine /dev/shm/sam3_plate_fp16.engine [--opt-level 5]
"""
import os, sys, time, argparse

PROMPT = "license plate"
RES = 1008


def export_onnx(args):
    os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/workspace/.hf_home"))
    import torch
    from PIL import Image
    from transformers.models.sam3 import Sam3Processor, Sam3Model

    dev = "cpu"  # exportar en CPU = maxima compatibilidad y sin OOM durante el trace
    model = Sam3Model.from_pretrained("facebook/sam3").to(dev).eval()
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    # el contenido de la imagen no importa: solo fija el shape 1008 y hornea el prompt
    img = Image.open(args.ref_img).convert("RGB") if args.ref_img else Image.new("RGB", (RES, RES), 0)
    inp = proc(images=img, text=PROMPT, return_tensors="pt")
    pv = inp["pixel_values"].to(dev).float()
    ii = inp["input_ids"].to(dev).long()       # 'license plate' -> [49406,10337,5135,49407,...] (fijo)
    am = inp["attention_mask"].to(dev).long()
    assert tuple(pv.shape) == (1, 3, RES, RES), f"pixel_values {tuple(pv.shape)} != (1,3,{RES},{RES})"

    class Wrapper(torch.nn.Module):
        def __init__(s, m, ii, am):
            super().__init__(); s.m = m
            s.register_buffer("ii", ii)        # prompt horneado
            s.register_buffer("am", am)

        def forward(s, pixel_values):
            o = s.m(pixel_values=pixel_values, input_ids=s.ii, attention_mask=s.am)
            return o.pred_masks, o.pred_boxes, o.pred_logits, o.presence_logits

    w = Wrapper(model, ii, am).to(dev).eval()
    os.makedirs(os.path.dirname(os.path.abspath(args.onnx)), exist_ok=True)
    t = time.time()
    with torch.inference_mode():
        torch.onnx.export(
            w, (pv,), args.onnx,
            input_names=["pixel_values"],
            output_names=["pred_masks", "pred_boxes", "pred_logits", "presence_logits"],
            dynamo=False, opset_version=17, do_constant_folding=True)
    print(f"[export] ONNX OK en {time.time()-t:.0f}s -> {args.onnx}", flush=True)


def build_engine(args):
    import tensorrt as trt
    log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(log)
    network = builder.create_network(0)        # TRT 10: batch explicito por defecto
    parser = trt.OnnxParser(network, log)
    if not parser.parse_from_file(args.onnx):
        for i in range(parser.num_errors):
            print("PARSE ERR", parser.get_error(i))
        raise SystemExit("parse failed")
    cfg = builder.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb << 30)
    if args.opt_level is not None:
        try: cfg.builder_optimization_level = args.opt_level
        except Exception as e: print("opt-level set fail:", e)
    print(f"[build] parse OK, construyendo engine FP16 (opt_level={args.opt_level}, varios min)...", flush=True)
    t = time.time()
    ser = builder.build_serialized_network(network, cfg)
    if ser is None:
        raise SystemExit("build_serialized_network returned None")
    with open(args.engine, "wb") as f:
        f.write(bytes(ser))
    print(f"[build] ENGINE OK en {time.time()-t:.0f}s, {ser.nbytes >> 20} MB -> {args.engine}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export-onnx"); e.add_argument("--onnx", required=True)
    e.add_argument("--ref-img", default=None, help="opcional: cualquier imagen (solo fija el shape 1008; si no, usa una negra)")
    e.set_defaults(func=export_onnx)
    b = sub.add_parser("build-engine"); b.add_argument("--onnx", required=True)
    b.add_argument("--engine", required=True)
    b.add_argument("--opt-level", type=int, default=None, help="builder_optimization_level TRT (0-5, def 3)")
    b.add_argument("--workspace-gb", type=int, default=12)
    b.set_defaults(func=build_engine)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
