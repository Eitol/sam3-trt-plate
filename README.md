# sam3-trt-plate

**SAM 3 (image model) corriendo en TensorRT FP16** para segmentar **patentes** (license plates) por
crop de 1008×1008. Salida **idéntica a PyTorch** (sin pérdida de calidad) y **~2× más rápido**.

Pensado para "clonar y usar": `./setup.sh` instala deps, exporta el ONNX y **construye el engine
TensorRT para la GPU de esa máquina**.

---

## Qué hace

Dada una imagen + el bounding box de una patente (p.ej. de un detector como RF-DETR), recorta una
ventana de **1008×1008 centrada en la patente**, corre **SAM 3** con el prompt fijo `"license plate"`
vía **TensorRT FP16**, y devuelve la **máscara precisa de la patente** (RLE de pycocotools) en
coordenadas de la imagen completa.

El prompt `"license plate"` está **horneado** en el grafo (no se re-tokeniza por imagen) y el input
es de shape fijo `1×3×1008×1008` → grafo 100% estático, ideal para TensorRT.

## Benchmark — TensorRT vs PyTorch ("lo normal")

Medido sobre **60 crops de patentes reales** de 1008×1008, 1 prompt `"license plate"`, en una
**RTX 3090** (TensorRT 10.16, torch 2.10 cu128). Warm (sin contar el primer batch).

### Velocidad

| Método | ms/img | img/s | speedup |
|---|---:|---:|---:|
| PyTorch nativo (bf16) — *lo normal, producción* | 167 | 6.0 | **1.0×** (baseline) |
| PyTorch HF (fp32, sin acelerar) | 472 | 2.1 | 0.35× |
| **TensorRT FP16 — cómputo puro de SAM3** | **75** | **13.2** | **2.2×** |
| **TensorRT FP16 — runner end-to-end** (load+crop+post) | **95** | **10.5** | **1.75×** |

> "Cómputo puro" = solo la inferencia del modelo (lo que TensorRT acelera). El runner end-to-end
> incluye además leer la imagen, recortar el crop 1008 y el post-process. Sobre las **~140k imágenes
> HD+** del proyecto, eso baja la corrida de re-segmentación de ~1 h a ~30 min.

### Calidad — **NO baja** ✅

| Comparación | maskIoU mean | median | min | detección |
|---|---:|---:|---:|---:|
| **TRT-FP16 vs el MISMO modelo en PyTorch fp32** | **0.9991** | 0.9995 | **0.9927** | — |
| TRT-FP16 vs PyTorch nativo bf16 (otra implementación) | 0.953 | 0.964 | 0.841 | 60/60 |
| *(referencia)* PyTorch HF fp32 vs nativo bf16 | 0.953 | 0.964 | 0.841 | 60/60 |

La máscara de TensorRT FP16 es **99.9% idéntica** a la del mismo modelo en fp32 PyTorch (100% de los
crops ≥ 0.99 IoU; |Δscore| medio 0.0009). El gap es redondeo sub-píxel de bordes, **no** una pérdida
de TensorRT: nótese que TRT-FP16 se desvía del PyTorch nativo **exactamente lo mismo** (0.953) que
otro PyTorch (HF fp32) → la diferencia es entre *implementaciones*, no por usar TensorRT/FP16.

## ⚠️ Por qué se construye con la API nativa de TensorRT (y NO con onnxruntime)

Esto es lo importante y lo que costó depurar:

> El **optimizador de grafo de ONNX Runtime** (cualquier `graph_optimization_level` ≥ BASIC)
> **corrompe la atención "windowed" con RoPE** del ViT de SAM 3 en CUDA → features con error
> relativo ~1.0 → máscaras malas (IoU ~0.76, detecciones perdidas). Como `onnxruntime` con
> `TensorrtExecutionProvider` **optimiza con ORT antes** de pasar el grafo a TensorRT, ese camino
> hereda la corrupción.

**Fix:** construir el engine con la **API nativa de TensorRT** (`trt.OnnxParser` →
`build_serialized_network`, FP16), sin pasar por el optimizador de ORT. El parser/optimizador propio
de TensorRT es correcto. (Diagnóstico: cada capa por separado y el backbone completo exportan
perfecto; solo se rompe cuando ORT optimiza el grafo grande. `CUDA opt=DISABLE_ALL → ✓; BASIC/ALL → ✗`.)

`build_sam3_trt_engine.py` ya implementa el camino correcto.

---

## Requisitos

- GPU NVIDIA + CUDA. Probado en RTX 3090 (sm_86), TensorRT 10.16, torch 2.10 cu128, transformers 5.12.
- `HF_TOKEN` de Hugging Face con acceso a `facebook/sam3` (repo *gated* → aceptá la licencia en HF).
- `torch` (build CUDA) y `tensorrt 10.x` instalados (en cajas vast.ai suelen venir en `/venv/main`).
  Las demás deps (transformers, pycocotools, pillow, numpy) las instala `setup.sh`.

> El engine TensorRT **no es portable** entre GPUs ni versiones de TRT. En cada máquina nueva hay que
> correr `setup.sh` para reconstruirlo (~10-15 min, una sola vez).

## Quickstart

```bash
git clone https://github.com/Eitol/sam3-trt-plate.git
cd sam3-trt-plate
HF_TOKEN=hf_xxxxx ./setup.sh
# (opcional) elegir intérprete / opt-level / carpeta de salida:
#   PYBIN=/venv/main/bin/python OPTLVL=5 WORKDIR=/dev/shm/sam3trt HF_TOKEN=hf_xxx ./setup.sh
```

Eso deja el engine en `/dev/shm/sam3trt/sam3_plate_fp16.engine`.

### Correr sobre un lote (CLI)

`manifest.json` = lista de `[ruta_imagen, [x1,y1,x2,y2]]` (bbox de patente en coords de la imagen):

```json
[
  ["/data/frames/0001.jpg", [2066, 585, 2357, 758]],
  ["/data/frames/0002.jpg", [120, 430, 360, 520]]
]
```

```bash
export LD_LIBRARY_PATH=$(python -c 'import os,tensorrt_libs;print(os.path.dirname(tensorrt_libs.__file__))'):$LD_LIBRARY_PATH
HF_TOKEN=hf_xxxxx python segment_sam3_trt.py \
    --engine /dev/shm/sam3trt/sam3_plate_fp16.engine \
    --manifest manifest.json \
    --min-side 1500            # solo re-segmenta imágenes HD+ (max(W,H) >= 1500)
```

Escribe un sidecar `<imagen>.sam3trt.seg.json` por imagen:

```json
{"coordinate_space":"fullframe","width":2560,"height":1440,"source":"sam3-trt",
 "masks":[{"instance_id":"p0","klass":"license_plate","is_target":true,"score":0.974,
           "bbox_xyxy":[2067.3,585.4,2358.7,757.6],"rle":{"size":[1440,2560],"counts":"..."}}]}
```

### Usarlo desde Python

```python
from segment_sam3_trt import Sam3PlateTRT
from PIL import Image

seg = Sam3PlateTRT("/dev/shm/sam3trt/sam3_plate_fp16.engine", hf_token="hf_xxx")
img = Image.open("frame.jpg").convert("RGB")
res = seg.segment_plate(img, [2066, 585, 2357, 758])   # bbox patente (xyxy, coords full)
# res = {"mask": <np.bool (H,W)>, "bbox_xyxy": [...], "score": 0.97}  o None si no detecta
```

## Archivos

- `setup.sh` — instala deps, exporta ONNX y construye el engine (clone-and-use).
- `build_sam3_trt_engine.py` — `export-onnx` (HF Sam3Model → ONNX estático, prompt horneado) y
  `build-engine` (API nativa TensorRT FP16). Documenta el gotcha de ORT.
- `segment_sam3_trt.py` — clase `Sam3PlateTRT` + CLI de lote (manifest → sidecars).
- `requirements.txt` — deps livianas (torch/tensorrt aparte).

## Notas técnicas

- Export: `torch.onnx.export(dynamo=False, opset=17)`. `dynamo=True` crashea (view/stride en el
  cross-attn del detr_decoder).
- Buffers de I/O de TensorRT vía tensores `torch.cuda` (`.data_ptr()` → `set_tensor_address`), sin pycuda.
- Post-process *lean* en GPU: se interpola **solo la máscara elegida** (288→1008), no las 200 queries
  (transferir las 200 eran ~40 ms/img de copia GPU→CPU inútil).
- Selección de la patente entre las detecciones: mayor IoU vs el bbox de RF-DETR; fallback al mayor score.

## Licencia

MIT (este código). El modelo SAM 3 (`facebook/sam3`) tiene su propia licencia de Meta — revisá los
términos en Hugging Face.
