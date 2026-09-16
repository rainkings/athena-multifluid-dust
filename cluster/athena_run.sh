#!/bin/bash
# athena_run.sh —— 你的 athena_run.sh 的集群版（TH-eX）
#
# 相对你的原版，只改了四件事（模式名、参数、提示语都保留原样）：
#   1) 集群上「后台跑」= 提交 Slurm 作业（yhbatch），不再用 nohup
#      —— 登录节点禁止跑计算程序（会被杀/封号），且终端一断任务就断
#   2) 作业内部用 yhrun 起 MPI（-n = 分配到的核数），而不是直接执行二进制
#   3) 自动加载模块（作业是非交互 shell，不读 ~/.bashrc）
#   4) 线程数用命令行覆盖 mesh/num_threads=<N> —— Athena++ 只认输入文件里的 num_threads
#      （src/mesh/mesh.cpp:118），不看 OMP_NUM_THREADS；不覆盖就是 56 rank × 64 线程抢 56 核
#   没有 yhbatch/yhrun 的机器（如 Callisto）自动退回你原来的 nohup / gdb 行为，同一个脚本两处都能用。
#
# 用法（**在算例目录里**执行，算例目录 = 当前目录）：
#   ~/athena/cluster/athena_run.sh              # gdb 交互调试（集群上先申请 1 核）
#   ~/athena/cluster/athena_run.sh gdb          # 同上
#   ~/athena/cluster/athena_run.sh gdb 744      # 从 iceline.00744.rst 起，gdb 调试
#   ~/athena/cluster/athena_run.sh nohup        # 从头算：提交作业（默认 debug6 / 1 节点 / 56 核 / 30 分钟）
#   ~/athena/cluster/athena_run.sh nohup 744    # 从 iceline.00744.rst 续算
#
# 可覆盖的环境变量（放在命令前）：
#   PARTITION=cp6 TIME=24:00:00 ~/athena/cluster/athena_run.sh nohup      # 正式分区/长时限
#   RUN_MODE 默认 omp：1 进程 × 56 线程（你本地那套 OpenMP 版二进制，不需要重编）
#   RUN_MODE=mpi   ... nohup        # 56 rank × 1 线程（需要 MPI=1 编出来的二进制）
#   NODES=2 NTASKS=64 ... nohup     # 想跨节点（注意 rank 数不能超过 meshblock 数 64）
#   ATHENA_SB_DIR=/path/to/bin ...  # 换二进制所在目录（默认 $HOME/athena/bin）
set -uo pipefail

ATHENA_SB_DIR="${ATHENA_SB_DIR:-$HOME/athena/bin}"
BIN="$ATHENA_SB_DIR/athena"
CASE_DIR="$PWD"
INPUT="${INPUT:-athinput.iceline}"
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

PARTITION="${PARTITION:-debug6}"
NODES="${NODES:-1}"
NTASKS="${NTASKS:-56}"
TIME="${TIME:-00:30:00}"
RUN_MODE="${RUN_MODE:-omp}"     # 默认 omp：1 进程 × NTHREADS 线程（默认 32，见下）
                                 # 想用 MPI 版多进程：RUN_MODE=mpi（需要 MPI=1 编出来的二进制）
# 线程数：omp 模式一个进程就用 NTHREADS 个核 → CPU 申请量对齐到 NTHREADS（别多占额度）
# 默认 32：实测（EPR_SF，两次 30 分钟）32 线程略快于 56 线程 —— 算例太小（64 blocks × 80~120 单元、
# 全在 cache 里），瓶颈是 OpenMP 同步/动态调度开销而非算力，线程越多同步越贵；而且只申请一半的核。
if [ "$RUN_MODE" = "omp" ]; then
  NTHREADS="${NTHREADS:-32}"
  if [ "$NTASKS" != "$NTHREADS" ]; then
    echo "omp 模式: CPU 申请量 NTASKS=$NTASKS → $NTHREADS（一个 OMP 进程正好用这么多核）"
    NTASKS="$NTHREADS"
  fi
else
  NTHREADS="${NTHREADS:-1}"
fi
# 硬限制：nranks × num_threads 不能超过 meshblock 数（64），否则 Athena++ 直接 FATAL
if [ "$NTHREADS" -gt 64 ] 2>/dev/null; then
  echo "⚠️  NTHREADS=$NTHREADS 超过 meshblock 数（64）：Athena++ 会报 FATAL" >&2
  echo "    'There are fewer MeshBlocks than OpenMP threads on each MPI rank'（src/mesh/amr_loadbalance.cpp:132）" >&2
fi

have() { command -v "$1" >/dev/null 2>&1; }

# 时间字符串 → 秒（支持 分钟 / MM:SS / HH:MM:SS / days-HH:MM:SS）
to_seconds() {
  local t="$1"
  case "$t" in
    *-*)  echo $(( ${t%%-*} * 86400 + $(to_seconds "${t#*-}") )) ;;
    *:*:*) IFS=: read -r h m s <<< "$t"; echo $(( 10#$h*3600 + 10#$m*60 + 10#$s )) ;;
    *:*)   IFS=: read -r m s <<< "$t"; echo $(( 10#$m*60 + 10#$s )) ;;
    ''|*[!0-9]*) echo 0 ;;
    *)     echo $(( 10#$t * 60 )) ;;
  esac
}

# 尽力读取分区的最大时限（仅当 scontrol 可用）；失败则返回空
partition_maxtime() {
  have scontrol || return 0
  scontrol show partition "$PARTITION" 2>/dev/null \
    | sed -n 's/.*TimeLimit=\([0-9:-]*\).*/\1/p' | head -1
}

# 判断二进制是不是 MPI 版：查它**自身**是否引用 MPI 符号。
# 不能用 ldd！本集群的并行 HDF5(hdf5/*-mpi-x) 自己依赖 libmpi，串行构建的 athena 也会在 ldd 里
# 显示 libmpi → 会把串行版误判成 MPI 版，于是 `-n 56` 放出 56 个独立进程互相覆盖输出。
bin_is_mpi() {
  if have nm && nm "$BIN" >/dev/null 2>&1; then
    # 动态链接时是未定义符号(U)，静态链接时是已定义符号(T)，两种都能匹配
    nm "$BIN" 2>/dev/null | grep -qE '\bMPI_(Init|Init_thread|Comm_rank|Comm_size|Allreduce|Irecv|Isend)\b'
    return $?
  fi
  # nm 不可用/二进制被 strip 时的退路：ldd（可能误判，宁可误报 MPI）
  have ldd && ldd "$BIN" 2>/dev/null | grep -qiE 'libmpi|libmpich'
}

# 只设环境变量（在任何机器上都安全，包括 Callisto）
load_env() {
  # HDF5(≥1.10.7) 默认给打开的文件加 flock 锁；/fs2 是 Lustre，常不支持该锁 →
  # "H5FD__sec2_lock(): unable to lock file, errno = 11 (Resource temporarily unavailable)"
  # → 创建 iceline.out1.00000.athdf 直接失败。关掉文件锁即可（每个算例目录只有本作业在写，不需要锁）。
  export HDF5_USE_FILE_LOCKING=FALSE
  export HDF5_IGNORE_DISABLED_FILE_LOCK=TRUE
}

# 加载集群模块（只在 TH-eX 上做；Callisto 上不要 purge，免得打坏你自己的环境）
load_modules() {
  have module || return 0
  module purge >/dev/null 2>&1
  module add GCC/14.3.0                    >/dev/null 2>&1
  module add MPI/mpich/4.0.2-mpi-x-gcc14.3 >/dev/null 2>&1
  module add hdf5/1.14.5-gcc14.3-mpi-x     >/dev/null 2>&1
}

usage() {
  cat <<'EOF'
Usage:
  ./athena_run.sh            # Default gdb command  (集群上自动申请 1 核)
  ./athena_run.sh gdb        # gdb command
  ./athena_run.sh gdb <num>  # gdb with restart file
  ./athena_run.sh nohup      # 提交作业（集群）；无 Slurm 时退回 nohup
  ./athena_run.sh nohup <num># 提交作业并从 iceline.<num>.rst 续算
  ./athena_run.sh nohup latest # 同上，但自动取最新快照（接力续算用）
EOF
}

MODE="${1:-gdb}"
NUM="${2:-}"

# 只想问一句"我这份二进制是哪个版本"：
#   ./athena_run.sh check
if [ "$MODE" = "check" ] || [ "$MODE" = "info" ]; then
  echo "二进制: $BIN"
  [ -x "$BIN" ] || { echo "  不存在或不可执行（先跑 cluster/zx_config_th-ex.sh 编译）"; exit 1; }
  n_sym="$(nm "$BIN" 2>/dev/null | grep -cE '\bMPI_(Init|Init_thread|Comm_rank|Comm_size|Allreduce)\b')"
  echo "  MPI 符号引用数: ${n_sym:-?}（0 = 串行/OpenMP 版；≥1 = MPI 版）"
  if bin_is_mpi; then
    echo "  判定: **MPI 版**  → 用 RUN_MODE=mpi：yhrun -n <核数>（每个 rank 1 线程）"
  else
    echo "  判定: **串行/OpenMP 版** → 用 RUN_MODE=omp：yhrun -n 1 -c <线程数>"
    echo "        （不要用 -n 56！那会启动 56 个独立进程，互相覆盖同一个输出文件）"
  fi
  exit 0
fi

# 续算文件（与你原版一致：iceline.<5 位数字>.rst）
RST=""
case "$NUM" in
  "") ;;                                    # 从头算
  latest|newest)                            # 自动取最新快照（续算/接力时用，不必预先知道编号）
    # 快照号是单调递增的 → 取「最大的编号」比按 mtime 排序更可靠（不受文件时间戳精度/缓存影响）
    _n="$(ls -1 "$CASE_DIR"/iceline.*.rst 2>/dev/null \
          | sed -n 's#.*iceline\.\([0-9]\{1,\}\)\.rst$#\1#p' | sort -n | tail -1)"
    [ -n "$_n" ] || { echo "找不到任何 iceline.*.rst，无法续算" >&2; exit 1; }
    RST="$(printf 'iceline.%05d.rst' "$((10#$_n))")"   # 10# 强制十进制：printf %d 会把 00744 当八进制
    echo "续算：自动选最新快照 $RST" ;;
  *[!0-9]*)
    echo "第二个参数应为快照号（数字）或 latest，收到: $NUM" >&2; usage; exit 1 ;;
  *)
    RST="$(printf 'iceline.%05d.rst' "$((10#${NUM}))")"   # 10# 强制十进制（用户写 00744 也不会被当八进制）
    [ -f "$CASE_DIR/$RST" ] || { echo "找不到 $CASE_DIR/$RST" >&2; exit 1; } ;;
esac
if [ ! -f "$CASE_DIR/$INPUT" ]; then
  echo "找不到 $CASE_DIR/$INPUT —— 请在算例目录里执行本脚本" >&2; exit 1
fi

# 传给 athena 的参数：-i 输入、可选 -r 续算、以及线程数命令行覆盖
AT_ARGS=(-i "$INPUT")
[ -n "$RST" ] && AT_ARGS+=(-r "$RST")
AT_ARGS+=("mesh/num_threads=$NTHREADS")

# ─────────────────────────────────────────────────────────────────────────
# 情况 A：已经在 Slurm 分配里（yhbatch 拉起本脚本，或 yhrun --pty bash 进来）
#          → 这里才真正运行
# ─────────────────────────────────────────────────────────────────────────
if [ -n "${SLURM_JOB_ID:-}" ]; then
  load_env
  load_modules
  [ -x "$BIN" ] || { echo "找不到可执行文件 $BIN（先跑 cluster/zx_config_th-ex.sh 编译）" >&2; exit 1; }
  if [ "$MODE" = "gdb" ]; then
    CMD=(gdb --args "$BIN" "${AT_ARGS[@]}")
  elif [ "$RUN_MODE" = "omp" ]; then
    export OMP_NUM_THREADS="$NTHREADS"
    CMD=(yhrun -n 1 -c "$NTHREADS" "$BIN" "${AT_ARGS[@]}")
  else
    # 安全阀：MPI 模式必须配 MPI 版二进制。配错（拿串行版跑 yhrun -n 56）会启动 N 个
    # 各自独立的模拟、互相覆盖同一个输出文件 —— 静默毁数据，所以这里直接拦下。
    export OMP_NUM_THREADS="$NTHREADS"     # 混合并行时每 rank 的线程数
    if ! bin_is_mpi; then
      cat >&2 <<EOF
ERROR: RUN_MODE=mpi，但这份二进制不是 MPI 版（是串行/OpenMP 版）：$BIN
  · 想跑 MPI：重新编译   MPI=1 bash cluster/zx_config_th-ex.sh
  · 想跑 OpenMP：RUN_MODE=omp ~/athena/cluster/athena_run.sh nohup [num]
  · 先自我确认：~/athena/cluster/athena_run.sh check
  确实要用当前二进制跑多进程（不建议，输出会互相覆盖），加 FORCE=1。
EOF
      [ "${FORCE:-0}" = "1" ] || exit 5
    fi
    CMD=(yhrun -n "${SLURM_NTASKS:-$NTASKS}" "$BIN" "${AT_ARGS[@]}")
  fi
  echo "[$(date '+%F %T')] job=${SLURM_JOB_ID} node=$(hostname) dir=$CASE_DIR"
  echo "Running: ${CMD[*]}"
  "${CMD[@]}" 2>&1 | tee "athena.${SLURM_JOB_ID}.log"
  rc=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] exit=$rc"
  grep -E "zone-cycles|Number of MeshBlocks" "athena.${SLURM_JOB_ID}.log" | tail -6
  exit $rc
fi

# ─────────────────────────────────────────────────────────────────────────
# 情况 B：在登录节点
# ─────────────────────────────────────────────────────────────────────────
case "$MODE" in
  nohup|run|submit)
    if have yhbatch; then
      # 提交前就拦下最常见的致命误配：串行/OpenMP 版二进制 + RUN_MODE=mpi（= N 个进程互写输出）
      if [ "$RUN_MODE" = "mpi" ] && [ -x "$BIN" ] && ! bin_is_mpi; then
        cat >&2 <<EOF
ERROR: $BIN 是串行/OpenMP 版，不能按 RUN_MODE=mpi 提交 —— 那会启动 N 个独立进程互相覆盖输出。
  · 跑 MPI：  MPI=1 bash cluster/zx_config_th-ex.sh         # 先重编 MPI 版，再提交
  · 跑 OpenMP：RUN_MODE=omp ~/athena/cluster/athena_run.sh nohup [num]
  · 自我确认：~/athena/cluster/athena_run.sh check
EOF
        [ "${FORCE:-0}" = "1" ] || exit 5
      fi
      if [ "$RUN_MODE" = "omp" ] && [ -x "$BIN" ] && bin_is_mpi; then
        echo "提示: 这份二进制是 MPI 版，但按 RUN_MODE=omp（默认）会跑成 1 进程 × ${NTHREADS} 线程。"
        echo "      想用多进程（${NTASKS} rank）：RUN_MODE=mpi ~/athena/cluster/athena_run.sh nohup [num]"
      fi

      # 提交前提示分区时限（scontrol 可用时）
      pmax="$(partition_maxtime)"
      if [ -n "$pmax" ]; then
        echo "分区 $PARTITION 最大时限: $pmax ；本次请求 TIME=$TIME"
        if [ "$(to_seconds "$TIME")" -gt "$(to_seconds "$pmax")" ]; then
          echo "⚠️  请求的 TIME 超过分区上限，yhbatch 会拒绝提交 —— 改小 TIME 或换分区。" >&2
        fi
      fi

      # 工作目录 = 算例目录（--chdir），Slurm 的 stdout/err 也用绝对路径钉在算例目录里
      CMD=(yhbatch --chdir="$CASE_DIR" -J "$(basename "$CASE_DIR")" -p "$PARTITION" \
           -N "$NODES" -n "$NTASKS" -t "$TIME" \
           -o "$CASE_DIR/slurm-%j.out" -e "$CASE_DIR/slurm-%j.err" \
           --export=ALL,RUN_MODE="$RUN_MODE",NTHREADS="$NTHREADS" "$SELF" nohup ${NUM:+"$NUM"})
      echo "提交作业: ${CMD[*]}"
      if ! subm_out="$("${CMD[@]}" 2>&1)"; then
        echo "$subm_out" >&2
        if echo "$subm_out" | grep -qi "time limit"; then
          cat >&2 <<EOF
提交被拒：时限不合法或超过分区上限。怎么办：
  · 看分区上限:   yhi | grep -w $PARTITION         # TIMELIMIT 列
  · 改小时限:     TIME=00:30:00 ~/athena/cluster/athena_run.sh nohup [num]
  · 跑长算例:     换到你有权限的长时分区，例如 PARTITION=cp6 TIME=12:00:00 ... nohup
  · 分段跑:       时限到点会被 SIGTERM/SIGKILL 硬杀 —— 用 nohup <最新快照号> 续算
EOF
        elif echo "$subm_out" | grep -qiE "not available|node configuration|cpu|memory"; then
          cat >&2 <<EOF
提交被拒：要不到这么多资源（通常是 CPU 数超过单节点核数）。怎么办：
  · 看节点 CPU 数: sinfo -p $PARTITION -o "%P %D %c %m"       # %c = 每节点 CPU 数
  · 6348 机型是 56 核/节点 → 单节点 omp 最多用 56 线程: NTHREADS=56 NODES=1 ... nohup
  · 想要 64 路并行请改用 MPI 版: RUN_MODE=mpi NODES=2 NTASKS=64 ... nohup（需 MPI=1 编译）
EOF
        fi
        exit 6
      fi
      echo "$subm_out"
      echo "看状态: yhq    看明细: yhcontrol show jobs <jobid>    取消: yhcancel <jobid>"
      echo "输出都写在: $CASE_DIR/"
      echo "  slurm-<jobid>.out / .err    Slurm 的作业 stdout/stderr"
      echo "  athena.<jobid>.log          Athena++ 运行日志（stdout+stderr）"
      echo "  iceline.*.rst|hst           Athena++ 输出（与你本地命名一致）"
      echo "  iceline.out1/out2.*.athdf(+.xdmf)"
    else
      echo "（本机没有 yhbatch → 退回你原来的 nohup 行为）"
      load_env
      echo "Running: nohup $BIN ${AT_ARGS[*]} &"
      nohup "$BIN" "${AT_ARGS[@]}" > nohup.out 2>&1 &
      echo "已后台启动 pid=$!"
    fi ;;
  gdb)
    if have yhrun; then
      echo "申请 1 核并进入 gdb（退出 gdb 后释放）…"
      yhrun --pty -p "$PARTITION" -N 1 -n 1 -t "$TIME" \
        bash -lc "cd '$CASE_DIR' && '$SELF' gdb ${NUM:-}" || {
          cat >&2 <<EOF

如果上面报 yhrun 不支持 --pty，就用两步法：
  yhrun -p $PARTITION -N 1 -n 1 -t $TIME --pty bash     # 拿到一个交互节点
  cd $CASE_DIR && $SELF gdb ${NUM:-}                    # 在该节点里跑 gdb
EOF
          exit 4
        }
    else
      echo "（本机没有 yhrun → 直接起 gdb）"
      load_env
      echo "Running: gdb --args $BIN ${AT_ARGS[*]}"
      gdb --args "$BIN" "${AT_ARGS[@]}"
    fi ;;
  *) usage; exit 1 ;;
esac
