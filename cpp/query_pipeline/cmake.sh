export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"

# 2) (optional) thread env you’ve been using
export MKL_THREADING_LAYER=GNU
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=CLOSE
export OMP_PLACES=cores

TORCH_PREFIX=$(python - <<'PY'
import torch, os

print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake"))
PY
)

echo "TORCH_PREFIX=$TORCH_PREFIX"
ls "$TORCH_PREFIX/Torch/TorchConfig.cmake"

# 3) clean + configure + build
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" \
  -DCMAKE_VERBOSE_MAKEFILE=ON

cmake --build build -j