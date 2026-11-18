#!/usr/bin/env bash
# Install SadTalker + 256 checkpoints + isolated venv (M2-friendly).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAD="${SADTALKER_DIR:-$ROOT/third_party/SadTalker}"

if [[ ! -f "$SAD/inference.py" ]]; then
  echo "Cloning SadTalker into $SAD ..."
  mkdir -p "$(dirname "$SAD")"
  git clone --depth 1 https://github.com/OpenTalker/SadTalker.git "$SAD"
fi

cd "$SAD"
mkdir -p checkpoints gfpgan/weights

dl() {
  local url="$1" out="$2"
  if [[ -f "$out" && -s "$out" ]]; then
    echo "OK $out"
    return 0
  fi
  echo "Downloading $out ..."
  curl -L --fail --retry 3 -C - -o "$out" "$url"
}

dl https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar ./checkpoints/mapping_00109-model.pth.tar
dl https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar ./checkpoints/mapping_00229-model.pth.tar
dl https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors ./checkpoints/SadTalker_V0.0.2_256.safetensors
dl https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth ./gfpgan/weights/detection_Resnet50_Final.pth
dl https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth ./gfpgan/weights/alignment_WFLW_4HG.pth
dl https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth ./gfpgan/weights/parsing_parsenet.pth

if [[ ! -x "$SAD/.venv/bin/python" ]]; then
  echo "Creating SadTalker venv..."
  python3 -m venv "$SAD/.venv"
fi

# shellcheck disable=SC1091
source "$SAD/.venv/bin/activate"
pip install -U pip wheel setuptools
pip install torch torchvision torchaudio
pip install 'numpy<2' 'scipy<1.12' 'opencv-python<4.10' 'contourpy<1.3' \
  face_alignment imageio imageio-ffmpeg 'librosa>=0.9.2' numba resampy pydub \
  kornia tqdm yacs pyyaml joblib scikit-image safetensors av facexlib gfpgan basicsr || true

# Patch basicsr for newer torchvision
DEG="$SAD/.venv/lib/python3."*/site-packages/basicsr/data/degradations.py
for f in $DEG; do
  if [[ -f "$f" ]]; then
    sed -i.bak 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$f" || true
  fi
done

# Patch deprecated numpy aliases inside SadTalker
find "$SAD/src" -name '*.py' -print0 | xargs -0 perl -pi -e 's/np\.float\b/np.float64/g; s/np\.int\b/np.int64/g; s/np\.bool\b/np.bool_/g' || true

echo
echo "SadTalker ready at: $SAD"
echo "Add to .env:"
echo "  SADTALKER_DIR=$SAD"
echo "  SADTALKER_PYTHON=$SAD/.venv/bin/python"
