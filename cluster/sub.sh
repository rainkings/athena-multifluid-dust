#!/bin/bash
# sub.sh — TH-eX 批处理作业模板（yhbatch）
#
# 用法：
#   cd ~/athena && cp cluster/sub.sh .            # 作业脚本放代码树根，$SLURM_SUBMIT_DIR 即 ~/athena
#   yhbatch sub.sh                                # 提交（手册 3.2.3）
#   yhq                                           # 看自己的作业
#   yhcontrol show jobs <jobid>                   # 看明细
#   yhcancel <jobid>                              # 取消
#
# 手册约束（务必遵守）：
#   * 计算节点没有编译器 → 先在登录节点把 bin/athena 编好（cluster/configure_th-ex.sh）
#   * yhbatch 申请的资源 ≥ 脚本内 yhrun 申请的资源
#   * 正式算例一律用 yhbatch；yhrun 交互式只在短测时用（关终端会中断）
#   * 抢占式调度：占 1 核即独占整节点 → 核数要和实际用量匹配

#SBATCH -p debug6          # ← 改成你有权限的分区：debug6(调试,≤2节点/112核) / cp6 / cps6 / short6 / long6 / cp6bm(1TB) …
#SBATCH -N 1               # 节点数（cp6 类节点 56 核/节点；cp5 类 36 核/节点）
#SBATCH -n 56              # 任务数（核数）—— 必须与下面 N= 一致
#SBATCH -J iceline         # 作业名
#SBATCH -o iceline.%j.out  # 标准输出（手册附录 E：建议显式指定）
#SBATCH -e iceline.%j.err  # 标准错误
#SBATCH -t 00:30:00        # 时限（days-hours:minutes:seconds）；冒烟测试先给 30 分钟

set -euo pipefail

N=56                       # ← 与 #SBATCH -n 保持一致（本算例 160x48 / meshblock 10x12 = 64 blocks，56 核已基本吃满）

# ── 运行环境（与构建时同一套模块；先跑 login-recon.sh 得到 HDF5 模块名）──
TOOLCHAIN="${TOOLCHAIN:-intel}"
HDF5_MODULE="${HDF5_MODULE:-}"      # 例：export HDF5_MODULE=hdf5/1.12.1（提交前在登录节点 export 即可被作业继承）
if [ -z "$HDF5_MODULE" ]; then
  echo "ERROR: 请先 export HDF5_MODULE=hdf5/<版本> 再 yhbatch（或直接在本行写死）" >&2
  exit 2
fi
module purge >/dev/null 2>&1 || true
if [ "$TOOLCHAIN" = "gcc" ]; then
  module add "${COMPILER_MODULE:-GCC/8.5.0}"
  module add "${MPI_MODULE:-MPI/mpich/4.0.2-mpi-x-gcc8.5}"
else
  module add "${COMPILER_MODULE:-Intel_compiler/19.0.4}"
  module add "${MPI_MODULE:-MPI/mpich/4.0.2-mpi-x-icc19.0}"
fi
module add "$HDF5_MODULE"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"   # 纯 MPI：1（避免 rank×thread 超额订阅）
                                                # 混合并行：改成每 rank 的线程数，并让 -n × OMP == 总核数

# ── 运行目录：每次作业一个目录，输入副本 + HDF5 输出 + 日志都在里面 ──
cd "${SLURM_SUBMIT_DIR:-$PWD}"
INPUT_SRC="inputs/snowline/athinput.iceline"
RUN="runs/iceline_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-manual}"
mkdir -p "$RUN"
cp -f "$INPUT_SRC" "$RUN/athinput.iceline"

echo "[$(date '+%F %T')] host=$(hostname) nodes=${SLURM_JOB_NUM_NODES:-?} tasks=$N omp=$OMP_NUM_THREADS"
echo "[$(date '+%F %T')] run dir: $RUN"

# ── 启动（手册 3.2.1 / 附录 E：用系统推荐 MPI 编译的程序可直接 yhrun，无需 mpirun）──
yhrun -n "$N" ./bin/athena -i "$RUN/athinput.iceline" -d "$RUN" 2>&1 | tee "$RUN/run.log"

echo "[$(date '+%F %T')] exit=${PIPESTATUS[0]}"
echo "结果文件："; ls -lh "$RUN" | head -20
