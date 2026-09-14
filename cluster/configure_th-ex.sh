#!/bin/bash
# configure_th-ex.sh — 在 TH-eX **登录节点**上构建本 Athena++ 分支（计算节点没有编译器）
#
# 用法：
#   cd ~/athena
#   export HDF5_MODULE=hdf5/<版本>          # 先跑 cluster/login-recon.sh 得到确切名字
#   bash cluster/configure_th-ex.sh                 # 默认 Intel 路线（手册 P6 推荐）
#   TOOLCHAIN=gcc bash cluster/configure_th-ex.sh   # GCC 路线（与本地 --cxx=g++ 一致）
#   MAKE_JOBS=16 bash cluster/configure_th-ex.sh    # 登录节点共享，建议 8~16
#
# 可覆盖的环境变量：TOOLCHAIN / COMPILER_MODULE / MPI_MODULE / HDF5_MODULE / HDF5_PREFIX / MAKE_JOBS / EXTRA_FLAGS
set -euo pipefail

TOOLCHAIN="${TOOLCHAIN:-intel}"
MAKE_JOBS="${MAKE_JOBS:-8}"

# ── 定位代码树（脚本在 cluster/ 下）───────────────────────────────────────────
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
echo "代码树: $ROOT"

# ── 模块与编译器（手册 P5–P6：Intel 编译器优先；MPI 用 Intel 后端的 mpich-x）──
if [ "$TOOLCHAIN" = "gcc" ]; then
  COMPILER_MODULE="${COMPILER_MODULE:-GCC/8.5.0}"
  MPI_MODULE="${MPI_MODULE:-MPI/mpich/4.0.2-mpi-x-gcc8.5}"
  CXX_NAME="g++"
else
  COMPILER_MODULE="${COMPILER_MODULE:-Intel_compiler/19.0.4}"
  MPI_MODULE="${MPI_MODULE:-MPI/mpich/4.0.2-mpi-x-icc19.0}"
  CXX_NAME="icpc"
fi

if [ -z "${HDF5_MODULE:-}" ]; then
  cat >&2 <<'EOF'
ERROR: 需要 HDF5 模块名。先在集群上执行：
    bash cluster/login-recon.sh          # 看 "hdf5 --" 那一段
    export HDF5_MODULE=hdf5/<版本>
再重跑本脚本。（本地 Makefile 里的 /home/izx/bin/hdf5 是别人家目录，集群上不存在。）
EOF
  exit 2
fi

echo "加载模块: $COMPILER_MODULE | $MPI_MODULE | $HDF5_MODULE"
# module 通常是 shell 函数：若当前是非登录 shell，退回到 bash -l
if ! type module >/dev/null 2>&1; then
  echo "(当前 shell 无 module 函数 → 需要 login shell，请用 'bash -l -c \"bash cluster/configure_th-ex.sh\"' 重跑)"
  exit 2
fi
module purge >/dev/null 2>&1 || true
module add "$COMPILER_MODULE"
module add "$MPI_MODULE"
module add "$HDF5_MODULE"

# ── HDF5 前缀（--hdf5_path）────────────────────────────────────────────────
HDF5_PREFIX="${HDF5_PREFIX:-${HDF5_ROOT:-${HDF5_DIR:-}}}"
if [ -z "$HDF5_PREFIX" ]; then
  # 模块没设环境变量时，从 module 的 prepend-path 里猜一次（并打印出来让人核对）
  HDF5_PREFIX="$(module show "$HDF5_MODULE" 2>&1 | grep -oE '/[^ :]*/include' | head -1 | sed 's#/include$##')"
fi
if [ -z "$HDF5_PREFIX" ] || [ ! -f "$HDF5_PREFIX/include/hdf5.h" ]; then
  echo "ERROR: 无法确定 HDF5 前缀（$HDF5_PREFIX）；请 export HDF5_PREFIX=<含 include/ 与 lib/ 的目录>" >&2
  exit 2
fi
echo "HDF5 前缀: $HDF5_PREFIX"

# ── 编译器可用性自检 ───────────────────────────────────────────────────────
for b in python3 "$CXX_NAME" mpicxx make; do
  command -v "$b" >/dev/null || { echo "ERROR: 找不到 $b（模块是否加载成功？）" >&2; exit 3; }
done

# ── configure：与本地 zx_config.sh 同一套物理开关，但 MPI 打开、HDF5 换集群 ──
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
set -x
python3 configure.py \
  --prob=disk_snowline_2D_RT_erg_2_twopop \
  --ndustfluids=4 --np=1 --nz=2 \
  --flux=hllc --coord=spherical_polar --eos=general/eos_phase_change --nghost=2 \
  -mpi -omp -hdf5 --hdf5_path="$HDF5_PREFIX" -h5double \
  --cxx="$CXX_NAME" --mpiccmd=mpicxx \
  $EXTRA_FLAGS
set +x

echo "=== 生成结果自检（应看到 MPI parallelism: ON）==="
grep -E "Problem generator|Coordinate system|Equation of state|Riemann solver|Number of dust fluids|Number of pebbles|Number of compositions|Linker flags|Floating-point precision|Number of ghost cells|MPI parallelism|OpenMP" configure.log || true

echo "=== 编译（登录节点，-j $MAKE_JOBS）==="
make clean
make -j "$MAKE_JOBS"

echo "=== 产物 ==="
ls -lh bin/athena
if command -v ldd >/dev/null; then ldd bin/athena | grep -iE "hdf5|mpi|not found" || true; fi
cat <<'EOF'

下一步：
  1) 复制作业模板：  cp cluster/sub.sh sub.sh   （改分区/核数）
  2) 冒烟测试：      yhbatch sub.sh            → yhq / yhcontrol show jobs <jobid>
  3) 正确性对齐：    与本地 OpenMP 结果比 hst 守恒量 / prim 极值（注意集群版是 MPI 域分解）
EOF
