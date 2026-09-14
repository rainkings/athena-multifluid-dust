# 在 TH-eX（天河）上跑本 Athena++ 分支 — 配置与作业手册

> 集群侧事实来源：用户手册《TH-eX 系统用户手册（2026-04）》，本仓库 `../TH-eX用户登陆手册-全文.layout.txt`（同目录另一份在 `~/Documents/tianhe/`）。
> 本仓库事实：Athena++ 多流体尘埃分支，problem = `disk_snowline_2D_RT_erg_2_twopop`，本地构建为 **MPI OFF / OpenMP ON**、HDF5 指向 `/home/izx/bin/hdf5`。

---

## 0. 一句话回答两个问题

**代码放哪**：放 **`$HOME/athena`**（即 `/fs2/home/saturn_lizx/athena`）。
理由：登录节点的家目录本身就在 **Lustre `/fs2`** 上（手册 P15–18；附录 D 里 VS Code 远程默认打开 `/fs2/home/<用户名>`），**所有登录节点与计算节点都挂载 `/fs2`**，而计算节点**看不到**登录节点的其他目录（手册 P3）。所以代码、二进制、输入、输出**全部必须在 `/fs2` 下**——放 `/tmp` 或节点本地盘＝作业里看不见。

**怎么配置**：在**登录节点**上装模块 → `configure.py -mpi -omp -hdf5 ...` → `make`（**计算节点没有编译器**，手册 P2），然后用 `yhbatch` 提交、脚本内用 `yhrun` 拉起。全套命令见 §2–§4，脚本已备好：`login-recon.sh`、`configure_th-ex.sh`、`sub.sh`、`rsync-to-th.sh`。

---

## 1. 目录布局（建议）

```
/fs2/home/saturn_lizx/
├── athena/                     # 代码树（本仓库镜像）→ 编译产物 bin/athena、obj/ 都在这里
│   ├── bin/athena              # 只在该集群上编译出来的可执行文件
│   ├── inputs/snowline/athinput.iceline
│   └── cluster/                # 本工具包
└── runs/                       # 每个算例一个目录：输入副本 + HDF5 输出 + 日志
    └── iceline_20260914_1730/
```

- **为什么把 runs/ 放在代码树外**：HDF5 输出会迅速吃掉配额，而代码树经常 `make clean`/重编；分开后重编不影响数据，清理数据也不会误伤代码。
- **配额**：登录时会打印磁盘配额表（手册表 2-1：Kbytes/Quota/Limit/Grace + 文件数）。先 `quota -s` / `lfs quota -h -u $USER /fs2` 看清上限再决定输出频率（`<output>.dt`）和保留几个 `rst`。
- **备份**：Lustre **不提供多副本保护**，手册明确要求自行备份重要数据与结果（P3）。干净的做法：算完一个阶段就 `rsync` 回本地（`rsync -avP th0:runs/iceline_xxx/ ./backup/`）。
- 手册没有文档化"独立 scratch 分区"——**不要假设有**；需要更大空间就联系 `support@nscc-tj.cn`（硬件）/`service@nscc-tj.cn`（软件）。

## 2. 传输（本地 → 集群）

你的 `~/.ssh/config` 里已经有别名了（`th0` = 192.168.10.50、`th1` = .51，User `saturn_lizx`），且 **VPN 必须在线**（当前 `utun7` 隧道在、`192.168.10.50:22` 可达）。

```bash
cd /home/lzx/mnt/callisto-athena-multifluid-dust
bash cluster/rsync-to-th.sh --dry-run   # 先看会传什么
bash cluster/rsync-to-th.sh             # 真传（会提示输入集群密码）
```

**不要传 `obj/` 与 `bin/`**：本地 `bin/athena` 动态链的是 `/home/izx/bin/hdf5/lib/libhdf5.so.320`，在集群上必然是 `not found`（已实测 `ldd`）。传过去也只会在集群上重新 `make`。源码树排除构建产物后只有 **16 MB**（含 `.git` 是 62 MB，`.git` 默认排除）。

## 3. 在登录节点上配置与编译

> 只在登录节点编译（计算节点无编译环境）。先在集群上跑一次侦察脚本，把模块名/HDF5 前缀/分区权限钉死：

```bash
ssh th0
mkdir -p ~/athena && exit          # 仅首次建目录
# 本地：bash cluster/rsync-to-th.sh
ssh th0
cd ~/athena && bash cluster/login-recon.sh        # 输出贴回来，我据此定稿模块名
```

拿到 HDF5 模块名后：

```bash
cd ~/athena
# 手册推荐（P6）：Intel 编译器 + 基于 Intel 的 MPI，性能最好
export HDF5_MODULE=hdf5/<版本>        # 由 login-recon.sh 得到，例如 hdf5/1.12.1
bash cluster/configure_th-ex.sh       # 默认 Intel_compiler/19.0.4 + MPI/mpich/4.0.2-mpi-x-icc19.0
```

等价的手工命令（脚本就是干这个）：

```bash
module purge
module add Intel_compiler/19.0.4
module add MPI/mpich/4.0.2-mpi-x-icc19.0
module add "$HDF5_MODULE"
python3 configure.py \
  --prob=disk_snowline_2D_RT_erg_2_twopop \
  --ndustfluids=4 --np=1 --nz=2 \
  --flux=hllc --coord=spherical_polar --eos=general/eos_phase_change --nghost=2 \
  -mpi -omp -hdf5 --hdf5_path="$HDF5_PREFIX" -h5double \
  --cxx=icpc --mpiccmd=mpicxx
make clean && make -j 8
```

要点（对照你本地的 `zx_config.sh`）：

| 项 | 本地（`zx_config.sh` / 现 `Makefile`） | 集群（本工具包） | 为什么 |
|---|---|---|---|
| MPI | **OFF** | **`-mpi`** | 不编 MPI 就只能单进程跑，`yhrun -n 56` 白给 |
| HDF5 路径 | `/home/izx/bin/hdf5` | 集群 `hdf5` 模块 | 那是别人家目录，集群上不存在 |
| 编译器 | `--cxx=g++`（MESA SDK 的 g++） | `--cxx=icpc` + `--mpiccmd=mpicxx` | 手册 P6：计算节点是 Intel 芯片，推荐 Intel 编译器 + Intel 后端 MPI（`mpich/4.0.2-mpi-x-icc19.0`） |
| 优化 | `-O3 -std=c++11 -fopenmp -g -O0`（**实际是 -O0，慢**） | 默认 `-O3` 一套 Flags | 本地那串里 `-O0` 在最后，会覆盖 `-O3` |
| 并行 | `num_threads = 64`（本机只有 **2** 核） | 纯 MPI 时 `num_threads = 1`；混合并行时每 rank 的线程数 ≤ 核数/rank | 64 线程在 2 核上是超额订阅；集群节点 56 核（6348）/36 核（6354） |

**GCC 路线**（如果 Intel 19.0.4 编不过你的自定义代码，或你想与本地一致）：

```bash
TOOLCHAIN=gcc bash cluster/configure_th-ex.sh
# = GCC/8.5.0 + MPI/mpich/4.0.2-mpi-x-gcc8.5 + --cxx=g++ --mpiccmd=mpicxx
```

**要不要两套构建树**：`configure.py` 会就地重写 `Makefile` 与 `src/defs.hpp`，所以"生产 MPI 版"和"调试版"不能在同一个树里共存——需要时 `cp -a athena athena-debug` 各编一套。

## 4. 提交作业

本算例网格 `160×48`、meshblock `10×12` → **64 个 block**；单节点 56 核基本吃满（多出的 rank 会空转，`-n` 超过 64 没有收益）。

```bash
cd ~/athena
yhbatch sub.sh                       # 用 cluster/sub.sh 模板（先复制到 ~/athena/sub.sh 或直接用路径）
```

`cluster/sub.sh` 关键行（**先按你的账号权限改分区**）：

```bash
#SBATCH -p debug6        # 调试分区（≤2 节点/112 核）；正式算例换成你有权限的分区
#SBATCH -N 1
#SBATCH -n 56            # ← 必须与脚本里 yhrun -n 的值一致
yhrun -n 56 ./bin/athena -i runs/iceline_xxx/athinput.iceline -d runs/iceline_xxx
```

手册要点（P20–23、附录 E）：
- **`yhbatch` 申请的资源应不小于脚本内 `yhrun` 申请的资源**；
- 交互式 `yhrun` 不重定向 I/O 时，关终端会**杀掉任务** → 正式跑一律走 `yhbatch`（终端关闭不影响）；
- 脚本里建议显式 `-o/-e`，作业默认输出是 `slurm-<jobid>.out`；
- 分区与队列：`debug4/5/6`（调试，≤2 节点/112 核）、`cp*/cps*/short*/long*`（包机时/包规模，节点数核数**由账号权限决定**）、`cp6bm`（大内存 1TB）、`x11`（可视化）；
- **抢占式调度**：作业哪怕只占 1 个核，该节点也不会再分给别人 → 申请核数与实际用量要匹配，否则浪费的是你自己（也容易被管理员盯）。
- 登录节点上**禁止跑计算程序**（会被杀、可封号）——只能编辑/编译/提交/传数据。

监控与操作：

```bash
yhi -l                 # 节点/分区状态（手册 3.1.1）
yhq                    # 只能看到自己的作业（3.1.2）
yhcontrol show jobs <jobid>   # 作业明细（3.1.3）
yhcancel <jobid>       # 取消（3.1.4）
```

> 若 `yhi`/`yhq` 不可用，试 `sinfo -o "%20P %5a %10l %6D %t %N"` 与 `squeue -u $USER`（同一套调度系统的原生命令）。

## 5. 首轮验证（别一上来就跑生产）

1. **短时限冒烟**：`-t 00:10:00`、1 节点 56 核、把 `<time> nlim = 200`（或用 `-t` 限制），确认能在集群上正常启动、写 HDF5、正常退出。
2. **正确性对齐**：同样的 `athinput` + 更小的 `nlim`，与本地 OpenMP 串行结果对比（`hst` 文件里同一时刻的守恒量/极值；或 `python3 -c "import h5py..."` 比 `prim` 的 max/mean）。MPI 版本在域分解、tracer 行（你的 `CHANGELOG_LZX.md` 改的那部分）上最值得核对。
3. **再看性能**：`ncycle_out` 让 stdout 每 5000 步打印一次，从 `.out` 里看 `zone-cycles/cpu_second`，比较 `-n 28/56` 的加速比；`OMP_NUM_THREADS=1` 与 `-n 28 -c 2` 两种布局各测一次。

## 6. 已知坑（本地 vs 集群）

- **本地那个 `bin/athena` 在集群上不可用**（链的是 `/home/izx/bin/hdf5`）。
- **本地 git 报 "dubious ownership"**（目录属主是 uid 1005，你是 lzx）→ `git`/`git status` 在本仓库静默失败。修：`git config --global --add safe.directory /home/lzx/mnt/callisto-athena-multifluid-dust`（一条本地配置，可撤销）。
- **本仓库没有 git remote**（`git remote -v` 为空）→ 集群侧不能 `git clone`，只能 rsync；建议给仓库加个 remote（GitHub/GitLab）以便日后增量同步。
- VPN 掉线会让 ssh 卡住：手册建议 `~/.ssh/config` 加 `ServerAliveInterval 90 / ServerAliveCountMax 3 / TCPKeepAlive no`。
- 密码：首次 ssh 登录会强制改密；规则 ≥12 位且 ≥3 类字符；**90 天强制更换**；连错 5 次锁 10 分钟（手册 2.2.2）。
- HDF5 双精度（`-h5double`）与 `-hdf5` 都要带，否则输出类型和本地不一致，后处理脚本会踩坑。

## 7. 还需要确认的三件事（我无法从本地判定）

1. **你的账号有权限哪些分区**、配额多少 → `login-recon.sh` 的 `yhi`/`quota` 输出（或问 `service@nscc-tj.cn`）。
2. **HDF5 模块的确切名字与前缀** → `login-recon.sh` 的 `module av | grep hdf5`。
3. **首轮规模**：建议 `debug6` 1 节点 56 核跑冒烟；生产算例规模（节点数/核数/时限）按你的机时额度定。
