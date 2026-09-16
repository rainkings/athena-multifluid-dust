#!/bin/bash
# rsync-to-th.sh — 把 Athena 代码传到 TH-eX 登录节点（默认 ~/athena）
#
# 事实前提（本机实测）：
#   * 代码的真身在 Callisto：/home/izx/athena-multifluid-dust（本机 /home/lzx/mnt/callisto-athena-multifluid-dust 是 SSHFS 挂载）
#   * Callisto **连不到**集群（无 VPN 客户端、192.168.10.50:22 unreachable）→ 必须经本机中转
#   * 本机 VPN 在线（utun7），192.168.10.50:22 OPEN
#   * 因此默认用 tar 流式（Callisto 本地读 → 本机管道 → 集群），避免经 SSHFS 逐文件往返
#
# 用法：
#   bash cluster/rsync-to-th.sh                 # 默认 MODE=tar：Callisto → 本机 → th0:athena/
#   MODE=rsync bash cluster/rsync-to-th.sh      # 也可用 rsync（逐文件，慢一些，但可增量/预览）
#   bash cluster/rsync-to-th.sh --dry-run       # 只预览（仅 rsync 模式支持）
#   REMOTE=th1 DEST=athena-2/ ...               # 换登录节点或目标目录
#   MODE=tar-dry bash cluster/rsync-to-th.sh    # tar 模式预览（列出会传的条目，不实际写）
set -uo pipefail

MODE="${MODE:-tar}"
if [ "${1:-}" = "--dry-run" ] && [ "$MODE" = "tar" ]; then MODE="tar-dry"; fi
REMOTE="${REMOTE:-th0}"                 # th0=192.168.10.50 / th1=192.168.10.51（~/.ssh/config 已有）
DEST="${DEST:-athena}"                  # 相对 $HOME → /fs2/home/saturn_lizx/$DEST
CALLISTO="${CALLISTO:-Callisto}"        # ssh 别名（密钥免密，实测可用）
SRC_REMOTE="${SRC_REMOTE:-/home/izx/athena-multifluid-dust}"

EXCLUDES=(--exclude=obj --exclude=bin --exclude=.git --exclude=.preflight \
          --exclude='*.o' --exclude=Makefile --exclude=configure.log)

echo "模式    : $MODE"
echo "源      : ${CALLISTO}:${SRC_REMOTE}"
echo "目标    : ${REMOTE}:~/${DEST}/    (= /fs2/home/<你的账号>/${DEST}/)"
echo "排除    : obj/ bin/ .git/ .preflight/ *.o Makefile configure.log （构建产物与生成文件）"
echo

case "$MODE" in
  tar)
    # 一条管道：Callisto 打包 → 本机 → 集群解包。tar 在两端各自本地磁盘干活，快。
    ssh -o BatchMode=yes "$CALLISTO" "tar cf - -C '$SRC_REMOTE' ${EXCLUDES[*]} ." \
      | ssh "$REMOTE" "mkdir -p '$DEST' && tar xf - -C '$DEST'"
    rc=${PIPESTATUS[1]}
    ;;
  tar-dry)
    ssh -o BatchMode=yes "$CALLISTO" "tar cf - -C '$SRC_REMOTE' ${EXCLUDES[*]} ." | tar tf - | head -40
    echo "...（预览前 40 条；共 $(ssh -o BatchMode=yes "$CALLISTO" "tar cf - -C '$SRC_REMOTE' ${EXCLUDES[*]} ." | tar tf - | wc -l) 条）"
    rc=0
    ;;
  rsync)
    DRY=""
    [ "${1:-}" = "--dry-run" ] && DRY="--dry-run"
    # 经本机 SSHFS 挂载读取 → 推到集群
    HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SRC="$(cd "$HERE/.." && pwd)/"
    echo "rsync 源: $SRC"
    rsync -avP $DRY "${EXCLUDES[@]/#/}" "$SRC" "${REMOTE}:${DEST}/"
    rc=$?
    ;;
  *)
    echo "ERROR: 未知 MODE=$MODE（可选 tar / tar-dry / rsync）" >&2; exit 2;;
esac

if [ "$rc" != "0" ]; then
  echo "传输失败（rc=$rc）。若是 Permission denied → 集群侧需要密码：先手工 ssh $REMOTE 登录一次（输密码）再重跑。" >&2
  exit "$rc"
fi

cat <<EOF

传输完成。集群侧下一步：
  ssh $REMOTE
  cd ~/${DEST}
  MPI=1 bash cluster/zx_config_th-ex.sh     # 编译（= 你的 zx_config + MPI；不要 MPI 就去掉 MPI=1）
  # 跑算例（在算例目录里）：
  cd ~/athena_works/<case> && ~/${DEST}/cluster/athena_run.sh nohup
EOF
