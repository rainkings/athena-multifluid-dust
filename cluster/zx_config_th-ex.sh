#!/bin/bash
# zx_config_th-ex.sh —— 你的 zx_config.sh 的集群版（在 TH-eX 登录节点执行）
#
# 与原文件的差别只有三处，其余 flag 逐字未改：
#   ① 先加载集群模块（编译器等）；② ${hdf5_path} 由本机 /home/izx/bin/hdf5 换成集群模块前缀；
#   ③ make -j 16 → 可调（登录节点共享，默认 8）。
# 原本的 MPI OFF 保持不变（=你本地那套）；要开 MPI 用 MPI=1（多一行 -mpi --mpiccmd=mpicxx）。
#
# 用法：
#   cd ~/athena && bash cluster/zx_config_th-ex.sh            # = 你的原配置（OpenMP，单节点）
#   cd ~/athena && MPI=1 bash cluster/zx_config_th-ex.sh      # 加 -mpi，可 yhrun -n 56 多进程/多节点
#   MAKE_JOBS=16 bash cluster/zx_config_th-ex.sh              # 想快一点（登录节点别拉满）
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
echo "代码树: $PWD"

# ── ① 集群模块（与你上次编通的那套完全相同，别再换）──
if ! type module >/dev/null 2>&1; then
  echo "ERROR: 当前 shell 没有 module 函数，请用 bash -l -c 'bash cluster/zx_config_th-ex.sh'" >&2; exit 2
fi
module purge >/dev/null 2>&1
module add GCC/14.3.0                    || { echo "ERROR: GCC/14.3.0 加载失败" >&2; exit 3; }
module add MPI/mpich/4.0.2-mpi-x-gcc14.3 || { echo "ERROR: MPI 加载失败" >&2; exit 3; }
module add hdf5/1.14.5-gcc14.3-mpi-x     || { echo "ERROR: HDF5 加载失败" >&2; exit 3; }

# ── ② hdf5_path：你本地是 export hdf5_path=/home/izx/bin/hdf5；集群用模块路径 ──
#    （若你已自己 export hdf5_path=...，这里就不覆盖）
: "${hdf5_path:=/fs2/software/hdf5/1.14.5-gcc14.3-mpi-x}"
if [ ! -f "$hdf5_path/include/hdf5.h" ]; then
  echo "ERROR: $hdf5_path/include/hdf5.h 不存在 —— 模块没加载对？" >&2
  echo "（这个检查很重要：你的原脚本用了 \${hdf5_path}，若它为空，configure 会拿到 --hdf5_path= 而不报错）" >&2
  exit 4
fi
echo "hdf5_path = $hdf5_path"

# ── ③ -mpi 开关（默认关，保持你的原配置）──
if [ "${MPI:-0}" = "1" ]; then MPI_FLAG="-mpi --mpiccmd=mpicxx"; else MPI_FLAG=""; fi

# ── 你的配置（flag 逐字保留，只插入 ${MPI_FLAG}）──
python3 configure.py --prob=disk_snowline_2D_RT_erg_2_twopop --ndustfluid=4 --flux=hllc \
  ${MPI_FLAG} -omp -hdf5 --hdf5_path=${hdf5_path} -h5double --nghost=2 \
  --coord=spherical_polar --eos=general/eos_phase_change --np=1 --nz=2 --cxx=g++
rc=$?
[ $rc -eq 0 ] || { echo "ERROR: configure 失败 rc=$rc" >&2; exit $rc; }

echo "--- configure 回显（MPI parallelism 应为 $([ "${MPI:-0}" = 1 ] && echo ON || echo OFF)）---"
grep -E "Problem generator|Coordinate system|Equation of state|Riemann solver|Number of dust fluids|Number of pebbles|Number of compositions|Number of ghost|Floating-point precision|MPI parallelism|Linker flags" configure.log

make clean
make -j "${MAKE_JOBS:-8}"
rc=$?
[ $rc -eq 0 ] || { echo "编译失败 rc=$rc；把 make 输出末 60 行贴出来" >&2; exit $rc; }

echo "--- 产物 ---"
ls -lh bin/athena
ldd bin/athena | grep -iE 'hdf5|mpi' || true
echo
if [ "${MPI:-0}" = "1" ]; then
  echo "已编好 MPI 版：作业里用  yhrun -n 56 ./bin/athena -i <input> -d <rundir>"
else
  echo "已编好 OpenMP 版（无 MPI）：作业里用  yhrun -n 1 -c 56 ./bin/athena ...，并设 OMP_NUM_THREADS=56"
  echo "注意：你的输入文件里 num_threads = 64 > 单节点 56 核 → 建议改成 56（或设 OMP_NUM_THREADS=56）"
fi
