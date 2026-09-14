#!/bin/bash
# rsync-to-th.sh — 把本地代码树同步到 TH-eX 登录节点（默认 ~/athena）
#
# 前置：VPN 在线（手册 2.1）；~/.ssh/config 里有别名（本机已有 th0/th1 → User saturn_lizx）
# 用法：
#   bash cluster/rsync-to-th.sh --dry-run     # 先看会传哪些文件
#   bash cluster/rsync-to-th.sh               # 真传（会提示输入集群密码）
#   REMOTE=th1 DEST=athena-2/ bash cluster/rsync-to-th.sh     # 换登录节点/目标目录
set -euo pipefail

DRY=""
if [ "${1:-}" = "--dry-run" ]; then DRY="--dry-run"; fi

REMOTE="${REMOTE:-th0}"          # 登录节点别名（th0=192.168.10.50 / th1=.51）
DEST="${DEST:-athena/}"          # 相对 $HOME 的目标目录

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/"

echo "源  : $SRC"
echo "目标: ${REMOTE}:${DEST}  (= /fs2/home/<你的账号>/${DEST})"
echo "排除: obj/ bin/ .git/ *.o Makefile configure.log（构建产物与生成文件；HDF5 路径在本机是 /home/izx/bin/hdf5，传过去没用）"
echo "体积: $(timeout 30 du -sh --exclude=obj --exclude=bin --exclude=.git "$SRC" 2>/dev/null | cut -f1)"
echo

rsync -avP $DRY \
  --exclude 'obj/' \
  --exclude 'bin/' \
  --exclude '.git/' \
  --exclude '*.o' \
  --exclude 'Makefile' \
  --exclude 'configure.log' \
  "$SRC" "${REMOTE}:${DEST}"

cat <<EOF

完成。集群侧接下来：
  ssh $REMOTE
  cd ~/athena
  bash cluster/login-recon.sh            # 首次：确认 hdf5 模块名 / 分区 / 配额
  export HDF5_MODULE=hdf5/<版本>
  bash cluster/configure_th-ex.sh        # 登录节点编译（计算节点无编译器）
  cp cluster/sub.sh . && yhbatch sub.sh  # 提交作业
EOF
