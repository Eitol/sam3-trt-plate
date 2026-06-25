#!/usr/bin/env bash
# setup.sh — clonar y usar: instala deps, exporta el ONNX y CONSTRUYE el engine TensorRT FP16
# de SAM3 'license plate' para LA GPU de esta máquina.
#
# El engine TRT NO es portable entre GPUs / versiones de TensorRT -> hay que construirlo acá.
# Tarda ~10-15 min (opt-level 3). Una sola vez por máquina.
#
# Uso:
#   HF_TOKEN=hf_xxxxx ./setup.sh
# Variables opcionales:
#   PYBIN=/venv/main/bin/python   # intérprete (default: autodetecta /venv/main o python3)
#   WORKDIR=/dev/shm/sam3trt       # dónde dejar onnx + engine (default: /dev/shm/sam3trt)
#   OPTLVL=5                       # builder_optimization_level de TRT (0-5). 5 = engine ~15% más
#                                  #   rápido en runtime pero build ~22min (vs ~3min con 3). Default 5
#                                  #   porque se construye 1 sola vez y se corre sobre muchas imágenes.
set -euo pipefail

: "${HF_TOKEN:?Definí HF_TOKEN (facebook/sam3 es un repo gated): HF_TOKEN=hf_xxx ./setup.sh}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
WORKDIR="${WORKDIR:-/dev/shm/sam3trt}"
OPTLVL="${OPTLVL:-5}"
ONNX="$WORKDIR/sam3_plate.onnx"
ENGINE="$WORKDIR/sam3_plate_fp16.engine"
HERE="$(cd "$(dirname "$0")" && pwd)"

# --- 1. intérprete de Python ---
if [ -z "${PYBIN:-}" ]; then
  if [ -x /venv/main/bin/python ]; then PYBIN=/venv/main/bin/python; else PYBIN="$(command -v python3 || command -v python)"; fi
fi
echo "[setup] python = $PYBIN"

# --- 2. LD_LIBRARY_PATH para libnvinfer (tensorrt_libs) ---
TRT_LIBS="$($PYBIN -c 'import os,tensorrt_libs;print(os.path.dirname(tensorrt_libs.__file__))' 2>/dev/null || true)"
if [ -n "$TRT_LIBS" ]; then export LD_LIBRARY_PATH="$TRT_LIBS:${LD_LIBRARY_PATH:-}"; echo "[setup] LD_LIBRARY_PATH += $TRT_LIBS"; fi

# --- 3. deps livianas ---
echo "[setup] instalando deps livianas..."
$PYBIN -m pip install -q -r "$HERE/requirements.txt"

# --- 4. verificar torch(cuda) + tensorrt ---
if ! $PYBIN - <<'PY'
import sys
try:
    import torch, tensorrt
    assert torch.cuda.is_available(), "torch no ve CUDA"
    print(f"[setup] torch {torch.__version__} (CUDA ok) | tensorrt {tensorrt.__version__}")
    assert int(tensorrt.__version__.split('.')[0]) == 10, "se requiere TensorRT 10.x"
except Exception as e:
    print("FALTA torch(CUDA) y/o tensorrt 10:", e); sys.exit(1)
PY
then
  echo "[setup] Instalá torch (build CUDA) y tensorrt 10 y reintentá. Ej:"
  echo "        $PYBIN -m pip install torch --index-url https://download.pytorch.org/whl/cu128"
  echo "        $PYBIN -m pip install 'tensorrt<11'"
  exit 1
fi

mkdir -p "$WORKDIR"

# --- 5. export ONNX (idempotente) ---
if [ -f "$ONNX" ]; then
  echo "[setup] ONNX ya existe: $ONNX (borralo para re-exportar)"
else
  echo "[setup] exportando ONNX (descarga checkpoint ~3.4GB la primera vez)..."
  $PYBIN "$HERE/build_sam3_trt_engine.py" export-onnx --onnx "$ONNX"
fi

# --- 6. build engine (idempotente) ---
if [ -f "$ENGINE" ]; then
  echo "[setup] engine ya existe: $ENGINE (borralo para reconstruir)"
else
  echo "[setup] construyendo engine TRT FP16 (opt-level=$OPTLVL, ~10-15 min)..."
  $PYBIN "$HERE/build_sam3_trt_engine.py" build-engine --onnx "$ONNX" --engine "$ENGINE" --opt-level "$OPTLVL"
fi

echo ""
echo "[setup] LISTO ✅  engine -> $ENGINE"
echo ""
echo "Para usarlo (acordate del LD_LIBRARY_PATH y HF_TOKEN):"
echo "  export LD_LIBRARY_PATH=$TRT_LIBS:\$LD_LIBRARY_PATH"
echo "  HF_TOKEN=$HF_TOKEN $PYBIN $HERE/segment_sam3_trt.py \\"
echo "      --engine $ENGINE --manifest plates.json"
echo ""
echo "manifest plates.json = [[\"/ruta/img.jpg\", [x1,y1,x2,y2]], ...]  (bbox patente de rf-detr)"
