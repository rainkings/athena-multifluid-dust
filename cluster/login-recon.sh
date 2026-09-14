#!/bin/bash
# login-recon.sh — 在 TH-eX 登录节点上跑一次（只读侦察，不改任何东西）
# 用法：  ssh th0 'bash -s' < cluster/login-recon.sh     或   ssh th0; cd ~/athena; bash cluster/login-recon.sh
# 目的：把「模块名 / HDF5 前缀 / 分区权限 / 配额 / Slurm 命令」这些本地判定不了的事实钉死。
set -uo pipefail

echo "===== 1) 身份与家目录 ====="
hostname; whoami; echo "HOME=$HOME"; echo "PWD=$(pwd)"; echo "SHELL=$SHELL"

echo; echo "===== 2) 磁盘配额（手册表 2-1 口径）====="
df -h "$HOME" 2>/dev/null | tail -2
command -v quota >/dev/null && quota -s 2>/dev/null | head -8
command -v lfs   >/dev/null && lfs quota -h -u "$USER" /fs2 2>/dev/null | head -8

echo; echo "===== 3) 共享目录 ====="
ls -d /fs2 /fs2/software /fs2/home "/fs2/home/$USER" 2>/dev/null
ls -1 /fs2 2>/dev/null | head -10
echo "-- 是否已存在代码目录 --"; ls -d "$HOME"/athena "$HOME"/athena* 2>/dev/null || echo "(无，需新建)"

echo; echo "===== 4) module 系统 + 关键模块名 ====="
bash -lc '
  module --version 2>&1 | head -2
  echo "-- hdf5 --"; module av 2>&1 | tr " " "\n" | grep -iE "^hdf5" | sort -u | head -10
  echo "-- fftw（本分支不需要，仅备查）--"; module av 2>&1 | tr " " "\n" | grep -iE "^fftw" | sort -u | head -5
  echo "-- 编译器 --"; module av 2>&1 | tr " " "\n" | grep -iE "^(Intel_compiler|GCC|nvhpc|pgi)/" | sort -u | head -12
  echo "-- MPI --"; module av 2>&1 | tr " " "\n" | grep -iE "^(MPI|impi|mvapich2|openmpi)/" | sort -u | head -12
'

echo; echo "===== 5) 加载手册推荐环境后，编译器/包装器是否可用 ====="
bash -lc '
  module purge >/dev/null 2>&1
  module add Intel_compiler/19.0.4 >/dev/null 2>&1
  module add MPI/mpich/4.0.2-mpi-x-icc19.0 >/dev/null 2>&1
  for b in python3 icpc icc mpicxx mpicc make rsync; do printf "%-8s %s\n" "$b" "$(command -v "$b" || echo -)"; done
  echo "module list:"; module list 2>&1 | sed -n "1,12p"
'

echo; echo "===== 6) HDF5 模块的前缀（--hdf5_path 用）====="
bash -lc '
  m=$(module av 2>&1 | tr " " "\n" | grep -iE "^hdf5" | sort -u | head -1)
  if [ -z "$m" ]; then echo "(未找到 hdf5 模块：需联系 support@nscc-tj.cn 确认 HDF5 提供方式)"; exit 0; fi
  echo "取模块: $m"
  module show "$m" 2>&1 | grep -E "HDF5|HDF5_ROOT|HDF5_DIR|PATH|LD_LIBRARY_PATH|prepend-path" | head -15
  module add "$m" >/dev/null 2>&1
  echo "HDF5_ROOT=${HDF5_ROOT:-<unset>}  HDF5_DIR=${HDF5_DIR:-<unset>}"
  if command -v h5cc >/dev/null; then echo "-- h5cc -show --"; h5cc -show 2>&1 | head -3; fi
  echo "-- 头文件/库探测 --"
  for p in "${HDF5_ROOT:-}" "${HDF5_DIR:-}" /fs2/software/hdf5*; do
    [ -n "$p" ] || continue
    [ -f "$p/include/hdf5.h" ] && echo "hdf5.h -> $p/include"
    ls "$p"/lib/libhdf5.so* 2>/dev/null | head -3
  done
'

echo; echo "===== 7) Slurm：分区状态与我自己的队列 ====="
if command -v yhi >/dev/null 2>&1; then yhi 2>&1 | head -25; else echo "(yhi 不可用)"; fi
if command -v yhinfo >/dev/null 2>&1; then yhinfo 2>&1 | head -15; else echo "(yhinfo 不可用)"; fi
if command -v sinfo >/dev/null 2>&1; then sinfo -o "%20P %5a %10l %6D %6t %N" 2>&1 | head -20; else echo "(sinfo 不可用)"; fi
echo "-- 我的作业 --"
if command -v yhq >/dev/null 2>&1; then yhq 2>&1 | head -12; elif command -v squeue >/dev/null 2>&1; then squeue -u "$USER" 2>&1 | head -12; else echo "(yhq/squeue 不可用)"; fi

echo; echo "===== 8) 登录节点约束提醒 ====="
echo "登录节点禁止跑计算程序（手册 P1）；编译请用 make -j 8 级别，别 -j 100 抢共享节点。"
echo "===== recon done ====="
