# WSL 环境与克隆部署

目标发行版为 `template-clone`，新工程位于 Windows `D:\codes\triton_back`，同步后完整源码位于 WSL `/opt/triton_back`。编译、运行均使用 WSL 内文件，不再引用原 `tensorrt_cpp` 工程。

环境现场清单可生成到 WSL `/root/system_env_info.txt`，可选项目副本为 `docs/system_env_info.txt`。这些文件含本机采集时间和运行状态，不入库；版本和运行状态以重新采集时刻为准。

## 环境组成

| 路径 | 职责 |
| --- | --- |
| `/opt/triton/2.71.0` | Triton r26.07 / 2.71.0 的原生 Core C API、头文件和 TensorRT backend |
| `/usr/local/cuda-13.3` | CUDA Toolkit 及 Runtime；Toolkit、Runtime 与驱动 API 分别在清单中记录 |
| `/usr/lib/x86_64-linux-gnu` | TensorRT 11.1.0.106、OpenCV、httplib、yaml-cpp 等系统运行库及开发包 |
| `/usr/lib/wsl/lib` | Windows NVIDIA 驱动向 WSL 暴露的 CUDA Driver API |
| `/opt/nvidia` | 原环境安装的 Nsight 性能分析工具 |
| `/opt/triton-staging` | Triton 源码、依赖和构建缓存，重建 Triton 时使用 |
| `/opt/triton-validation-20261014` | 前次环境验证和旧服务对照产物 |
| `/opt/triton_backend` | 已删除的旧服务部署目录；当前工程不依赖此目录 |
| `/opt/triton_back` | 新独立工程及其配置、模型、构建产物 |
| `/opt/triton_back/src` | 生产 C++ 源码，包含 app、api、services、configuration、domain、inference |

业务服务只有一个 C++ 进程：它加载 `libtritonserver.so`，通过进程内 C API 调用 TensorRT backend；不存在另一个需要同时启动的 Triton HTTP/gRPC 服务。当前原生 Triton 构建开启 GPU、statistics、logging、tracing、ensemble，关闭 Prometheus metrics、DCGM、cloud repositories、NVTX、MALI 和 NCCL multi-device。

## 重生成环境清单

在完成工程同步后，从 Windows PowerShell 执行：

```powershell
wsl.exe -d template-clone -u root --exec python3 /opt/triton_back/tools/environment/collect_environment.py --output /root/system_env_info.txt --copy-to /mnt/d/codes/triton_back/docs/system_env_info.txt
```

也可以在 WSL 内执行：

```bash
python3 /opt/triton_back/tools/environment/collect_environment.py --output /root/system_env_info.txt
```

脚本仅查询系统、包版本、GPU 信息、动态库依赖及相关进程；通过 `TRITONSERVER_ApiVersion` 查询 Core API，但不创建服务器、不加载模型、不执行推理。报告使用 UTF-8 和原子替换；若遇到不是该脚本生成的同名文件，会先保存带 UTC 时间的 `.previous-*` 备份。

## 克隆发行版

下面是手动克隆示例。先退出上位机并停止后端，再执行；`wsl --terminate` 会停止该发行版中的所有进程。目录和新发行版名可自行选择，导入目录应是专用空目录。

```powershell
New-Item -ItemType Directory -Force -Path D:\wsl\exports
wsl.exe --terminate template-clone
wsl.exe --export template-clone D:\wsl\exports\template-clone.tar
New-Item -ItemType Directory -Force -Path D:\wsl\triton-prod
wsl.exe --import triton-prod D:\wsl\triton-prod D:\wsl\exports\template-clone.tar --version 2
wsl.exe -d triton-prod -u root --exec python3 /opt/triton_back/tools/environment/collect_environment.py --output /root/system_env_info.txt
```

克隆保留 Linux 文件系统中的 CUDA、TensorRT、Triton 及开发依赖。Windows GPU 驱动、WSL 版本和主机硬件由目标电脑提供，不包含在导出包中。现有 TensorRT Engine 与本机 GPU / TensorRT 绑定，当前 Triton CUDA backend 也使用 `sm120f` 编译目标；换 GPU 或更改 TensorRT 后必须验证加载和真实图片推理，需要时重新构建 Engine / backend。

## 更新工程后构建与运行

先使用工程的 Windows 部署脚本同步完整项目，再在 WSL 内执行以下操作；Windows 生命周期操作详见 `docs/windows-host.md`。

```bash
cd /opt/triton_back
bash scripts/wsl/build.sh
bash scripts/wsl/run.sh
```

`build.sh` 使用发行版已安装的编译依赖及 `/opt/triton/2.71.0`，`run.sh` 指定工程自身配置。正常业务构建和运行不需要重新编译 Triton，也不需要原项目目录或测试对照目录。

运行后的兼容验收必须包括 `GET /api/status` 查询全部模型版本和能力、通过 `POST /api/models/select` 依次选择两个模型、核对状态中的 `modelReady` 与 `activeModel`，以及 `POST /infer` 的实际 PNG/原始像素推理。状态查询 HTTP 200 只表示 HTTP 可访问。还应检查候选准备期间旧模型继续推理、失败不改变旧模型、成功后旧请求完成并自动卸载；切换期间显存需同时容纳两个实例。仅查询版本、`ldd` 成功或 Core API 成功不足以证明模型可以运行。
