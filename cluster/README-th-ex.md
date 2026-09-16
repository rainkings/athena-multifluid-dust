# 在 TH-eX 上跑 Athena++（本分支）

账号 `saturn_lizx`；登录节点别名 `th0`（192.168.10.50）、`th1`（.51），`~/.ssh/config` 已配好。

## 1. 放哪里

```
/fs2/home/saturn_lizx/athena/                 代码 + bin/athena（编好的二进制）
/fs2/home/saturn_lizx/athena_works/<case>/    算例目录（自己建）：athinput.iceline + 输出
```

`/fs2` 是登录节点和所有计算节点**共同挂载**的共享存储（家目录就在其上）。放别处（`/tmp`、节点本地盘）作业里看不见。
输出直接落在算例目录，命名与本地一致：`iceline.NNNNN.rst`、`iceline.out1/out2.NNNNN.athdf(+.xdmf)`、`iceline.hst`。
配额用 `quota -s` / `df -h ~` 看（Lustre 无多副本，重要数据及时拉回本机）。

## 2. 编译（登录节点，通常只需一次）

```bash
ssh th0
cd ~/athena
MPI=1 bash cluster/zx_config_th-ex.sh     # = 你的 zx_config + MPI（推荐上集群）
bash cluster/zx_config_th-ex.sh           # 不加 MPI=1 就是 OpenMP 版（配 RUN_MODE=omp 用）
```

模块组合已内置并实测通过：`GCC/14.3.0` + `MPI/mpich/4.0.2-mpi-x-gcc14.3` + `hdf5/1.14.5-gcc14.3-mpi-x`。

## 3. 跑（在算例目录里执行，就用你自己的 athena_run.sh）

```bash
ssh th0
mkdir -p ~/athena_works/mycas && cd ~/athena_works/mycas
# 把 athinput.iceline 放进来（自己写/从别处拷），然后：

~/athena/cluster/athena_run.sh nohup          # 从头算：提交作业（默认 debug6 / 1 节点 / 56 核 / 30 分钟）
~/athena/cluster/athena_run.sh nohup 744      # 从 iceline.00744.rst 续算
~/athena/cluster/athena_run.sh gdb            # 交互调试（集群上先自动申请 1 核）
~/athena/cluster/athena_run.sh gdb 744        # 从 iceline.00744.rst 起，gdb 调试

PARTITION=cp6 TIME=24:00:00 ~/athena/cluster/athena_run.sh nohup     # 换正式分区/长时限
NTHREADS=56                 ~/athena/cluster/athena_run.sh nohup     # 线程数（默认模式 omp：1 进程 × 56 线程）
RUN_MODE=mpi                ~/athena/cluster/athena_run.sh nohup     # 换成多进程 56 rank（需要 MPI 版二进制）
~/athena/cluster/athena_run.sh check                                 # 问一句：这份二进制是哪个版本

yhq                                  # 看状态
yhcontrol show jobs <jobid>          # 看明细
yhcancel <jobid>                     # 取消
```

日志在算例目录：`slurm-<case>.<jobid>.out` 与 `athena.<jobid>.log`。
按你原来的习惯（把 `athena_run.sh` 拷进算例目录、`./athena_run.sh nohup`）也完全一样。
在没有 Slurm 的机器上（例如 Callisto）本脚本自动退回你原来的 `nohup` / `gdb` 行为，同一个脚本两处通用。

## 4. 拉数据回本机

```bash
cd ~/Science/hydro/athena_works/hydro_scripts
./sync_data_th-ex.sh latest <case>     # 或 ./sync_data_th-ex.sh 744 <case>
```

## 5. 四条硬约束（都踩过）

1. **核数要对齐三处**：Slurm `-n`（rank 数）× 每 rank 线程数 = 实际占用核数，且不超过请求量。
   Athena++ **只认输入文件 `[mesh] num_threads`**（`src/mesh/mesh.cpp:118`），**不看 `OMP_NUM_THREADS`**。
   `athena_run.sh` 用命令行覆盖 `mesh/num_threads=<N>`（Athena++ 支持 `athena [options] [block/par=value ...]`）——
   **不要手改 athinput 里的 num_threads**，否则同一份输入在两台机器上必然有一边是错的。
   默认 `RUN_MODE=omp` = **1 进程 × 32 线程**（`yhrun -n 1 -c 32`，申请量自动对齐到 32 个核）；
   `-n` 是进程数、`-c` 才是每进程核数，写成 `-n 56` 会变成 56 个独立进程互写同一批输出
   （日志里 "entering main loop" 会出现多次）。线程数上限是 block 数（64），超了 Athena++ 直接 FATAL
   （`src/mesh/amr_loadbalance.cpp:132`）。
   **实测（EPR_SF，两次 30 分钟）：32 线程略快于 56 线程。** 你这类算例只有 64 blocks × 80~120 单元、
   数据全在 cache 里 → 瓶颈是 OpenMP 同步/动态调度开销而不是算力，线程越多同步越贵；所以默认 32，
   同时机时省一半。想复现 56 档：`NTHREADS=56 ... nohup`。
2. **rank 数不能超过 meshblock 数**（`src/mesh/mesh.cpp:499`，超了直接 fatal）。你这套算例是 64 blocks
   （`160×32/48` + meshblock `10×8/10×12`）→ 单作业最多 64 rank；**单节点 56 rank 最合适**，跨节点通常更慢。
   要更多核必须先做大问题（缩小 meshblock 会增加 ghost 开销 2.10× → 2.70×）；真正有效的并行是**多个算例各占一个节点**。
3. **HDF5 模块必须与 MPI 同套**：后缀 `-mpi-x`/`-mpi-n` = 站点 mpich；带 `-openmpi` 后缀的会把它自己那套编译器 + MPI
   拉进来造成冲突。另：`module av` 在本集群会**卡死**（用 `module -t avail <pattern>`）；`icpc` 编不过
   `src/eos/general/general_hydro.cpp` 的 `#pragma omp simd` + `ATHENA_ERROR`（GCC 无此问题）；本地 HDF5 是 2.0.0、
   集群 1.14.5，代码无 2.0 专属 API，跨版本编译与读文件都无碍。
4. **HDF5 文件锁在 `/fs2` 上会失败**：HDF5 ≥1.10.7 默认给打开的文件加 `flock`，Lustre 常不支持 →
   `H5FD__sec2_lock(): unable to lock file, errno = 11 (Resource temporarily unavailable)` →
   `### FATAL ERROR in athdf5 initialization / Could not open iceline.out1.00000.athdf`（初始化即死，与算到哪无关）。
   修法：`export HDF5_USE_FILE_LOCKING=FALSE`（可加 `HDF5_IGNORE_DISABLED_FILE_LOCK=TRUE`）——
   `athena_run.sh` 三条路径（作业内 / 退回 nohup / 退回 gdb）都已自动设置。
   注意：串行（MPI OFF）构建走 HDF5 的 sec2 驱动才会碰到这个锁；MPI 版走 parallel HDF5(MPI-IO)，一般不涉及。
   另外确保**同一个算例目录只有一个作业在写**（两个作业同目录会互相锁/覆盖）。

## 6. 文件（`cluster/` 共 4 个）

| 文件 | 作用 |
|---|---|
| `athena_run.sh` | 你的运行脚本的集群版：`(无参数)`/`gdb [num]`/`nohup [num]`/`check`；**默认 `RUN_MODE=omp`（1 进程 × 32 线程）**，集群上自动走 yhbatch/yhrun，无 Slurm 时退回本地行为 |
| `zx_config_th-ex.sh` | 编译（你的 `zx_config.sh` + 集群模块；`MPI=1` 开 MPI） |
| `rsync-to-th.sh` | 从 Callisto 经本机把代码树同步到集群（排除 `obj/ bin/ .git/`） |
| `README-th-ex.md` | 本文件 |
