# WSL 克隆、更新与构建

Windows 项目根为 `D:\codes\triton_back`，WSL 原生项目根为 `/opt/triton_back`。完整项目包含源码、配置、两个 TensorRT 模型、ONNX 源模型和测试图片的原始像素；构建和运行无需原 `tensorrt_cpp` 项目、`/opt/triton-staging` 或 `/opt/triton-validation-20261014`。

## 克隆已有环境

在服务停止时导出已验证的发行版，随后导入新的发行版。以下名字和保存位置是示例，可替换：

```powershell
wsl.exe --export template-clone D:\wsl-images\triton-template.tar
wsl.exe --import camera-backend D:\wsl\camera-backend D:\wsl-images\triton-template.tar --version 2
```

克隆包含 `/opt/triton/2.71.0`、CUDA/TensorRT 系统库和编译依赖。Windows 的 NVIDIA 驱动及实际 GPU 来自目标电脑，导入不会安装驱动。`models/model_build/` 中的 `model.plan` 已在 TensorRT 11.1.0.106 和 RTX 5060 Laptop GPU 验证；更换 GPU 或 TensorRT 时需重新验证，必要时使用 `models/model_sources/` 中的 ONNX 构建对应 Engine。

## 更新、构建和启动

先停止正在运行的后端。项目来自 Windows checkout 时，在 Windows 项目根用 Windows Python 同步到目标发行版：

```powershell
python .\tools\sync_wsl.py --distribution camera-backend
```

同步工具用 Windows Python 读取授权文件的实际字节，通过标准输入传递带 SHA256 清单的归档，并在 WSL 中校验路径、文件类型和模型哈希。该方式适用于 Windows 文件系统透明加密的源码。`build`、`bin`、`runtime`、Python 缓存和 Git 元数据保留在本机。工具用 `.triton-back-project` 识别 `/opt/triton_back`。首次同步可以创建目标目录。

WSL 中已存在的 `config` 默认保留，内容有差异的传入配置放在 `/opt/triton_back/runtime/update-candidates/<时间>/config` 供审查。从旧模型目录迁移时，将保留配置的 `triton.repository` 改为 `../models/model_build`。若确认要整体采用项目配置，可带 `--replace-config` 同步：

```powershell
python .\tools\sync_wsl.py --distribution camera-backend --replace-config
```

同步后可从 Windows PowerShell 启动 WSL 中的原生项目，脚本会自动构建 Release 并在前台运行：

```powershell
wsl.exe -d camera-backend --cd /opt/triton_back --exec bash src/run.sh --release
```

上位机通过 `tools/manage.py` 受管启动时，先在 WSL 中构建测试与可执行文件并发布到 `bin/`：

```bash
cd /opt/triton_back
source /opt/triton/2.71.0/setup_env.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel 4
ctest --test-dir build -L unit --no-tests=error --output-on-failure
install -D -m 0755 build/bin/triton_backend bin/triton_backend
```

默认 `autoStart: false`，未指定模型时只开放 HTTP。查询 `GET /api/status` 获取所有模型和版本，再通过 `POST /api/models/select` 明确加载目标。启动时可同时传 `--model-id best --model-version 1.0.0`。受管启动的进程身份检查与停止方式见 [Windows 接入说明](windows-host.md#c-集成)。

## 局域网访问（WSL2 镜像网络）

当前项目配置的 `server.host` 为 `"0.0.0.0"`、端口为 6577，使 C++ 服务监听 WSL 的 IPv4 接口。读取旧部署配置时应核对该字段，修改后需重启后端。配置校验支持 `127.0.0.1`、`localhost`、`::1`、`0.0.0.0`，未填写 host 时默认 `127.0.0.1`。Windows 本机客户端可访问 `http://127.0.0.1:6577`。

Windows 的 WSL2 使用镜像网络（`networkingMode=mirrored`）。局域网客户端通过 Windows 服务器的局域网 IPv4 直接访问服务，无需配置 NAT 端口代理，也无需跟踪 WSL 的内部 IP。网络模式在 Windows 用户的 `.wslconfig` 中设置，修改后需要重启 WSL 才能生效；重启会影响该机器上正在运行的 WSL 进程，应先停止后端。

若 Windows 本机也要使用服务器的局域网 IP 调试，在 `%USERPROFILE%\.wslconfig` 中同时配置 `hostAddressLoopback`：

```ini
[wsl2]
networkingMode=mirrored

[experimental]
hostAddressLoopback=true
```

保留 `.wslconfig` 中其他已有设置。`hostAddressLoopback` 用于本机通过局域网 IP 回连，其他电脑的局域网访问不依赖此选项；本机也始终可使用 `127.0.0.1`。

由 Windows 管理员核对 TCP 6577 的 Windows 和 Hyper-V 入站规则。Windows 规则使用服务器实际网卡、局域网 IPv4 和允许访问的 IPv4 子网；Hyper-V 规则对应 WSL，并使用相同本地地址、客户端子网和端口。服务器局域网地址或子网变化时，更新入站规则和客户端地址。

网络配置完成后，在 Windows 项目根照常启动后端：

```powershell
wsl.exe -d template-clone --cd /opt/triton_back --exec bash src/run.sh --release --model-id best --model-version 2.0.0
```

从调用电脑实际请求 `http://<服务器Windows局域网IPv4>:6577/api/status?simple=true` 判断 HTTP 服务是否可访问，推理就绪还需检查 `modelReady` 和 `activeModel`。其他电脑的三个 HTTP 路由及 Apifox `baseUrl` 设置见 [windows-host.md](windows-host.md)。浏览器跨域调用另需匹配 `allowedOrigins`。

## Linux/WSL 研发快速启动

源码入口是 `src/app/main.cpp` 中的 `main()`。研发启动脚本位于 `src/run.sh`，会通过项目根 CMake 构建 `triton_backend`，然后直接 `exec` 可执行程序；日志继承当前终端，Ctrl+C 走服务正常退出流程。

```bash
cd /opt/triton_back/src
./run.sh
./run.sh --model-id best --model-version 2.0.0
./run.sh --release --model-id best --model-version 2.0.0
./run.sh --help
```

不需要必传命令行参数。默认配置为项目 `config/server.yaml`，当前 `autoStart=false`，零参数只开放 HTTP，之后通过选择接口加载模型。指定模型时同时提供 `--model-id` 和 `--model-version`；版本是 `2.0.0` 这样的业务版本。`--config FILE` 更换配置文件，`--jobs N` 指定 1–16 个编译任务，默认 4。

Debug、Release 分别增量构建到 `build/dev-debug` 和 `build/dev-release`，只构建服务目标，不自动运行测试。压测使用 `--release`。需要 GDB 断点时，可在脚本完成 Debug 构建后运行：

```bash
cd /opt/triton_back
source /opt/triton/2.71.0/setup_env.sh
gdb --args build/dev-debug/bin/triton_backend \
  --config /opt/triton_back/config/server.yaml \
  --model-id best --model-version 2.0.0
```

脚本不限制项目路径，也不执行 `sudo`；开发者需要对 checkout 和构建目录有写权限，并安装 CMake、Ninja、编译器和项目运行库。默认 `TRITON_ROOT=/opt/triton/2.71.0`，可用环境变量指定其他已配置的 Triton 根目录；它影响 CMake 和动态库搜索，YAML 中运行时使用的 `triton.root` 需要同步配置。`CUDA_HOME` 可指定 CUDA 根目录，默认 `/usr/local/cuda-13.3`。

默认配置按脚本所在项目定位。自定义 `--config` 的相对路径以调用者终端目录为准，YAML 内的模型仓库和模型注册文件等相对路径以该 YAML 文件目录为准。

## 运行文件和记录

| 路径 | 作用 |
|---|---|
| `/opt/triton_back/bin/triton_backend` | C++ HTTP、前后处理及进程内 Triton Core |
| `/opt/triton_back/config` | HTTP、运行设置及模型业务信息 |
| `/opt/triton_back/models/model_build` | Triton 模型仓库及模型哈希清单 |
| `/opt/triton_back/models/model_sources` | 重新构建 Engine 用的 ONNX 源模型 |
| `/opt/triton/2.71.0` | Triton Core、TensorRT backend 与 C API |
| `/opt/triton_back/runtime/service.pid.json` | Linux PID、启动 tick、命令行及调用者 token |
| `/opt/triton_back/runtime/sync-manifest.json` | 最新同步清单与配置保留结果 |
| `/opt/triton_back/runtime/logs/service.log` | C++ stdout/stderr，直接写入原生文件 |

切换目标加载、校验和预热成功后，新请求转到新模型，旧请求完成后自动卸载旧实例。新模型准备期间旧模型继续服务，加载失败保持旧模型。前台运行按 Ctrl+C 正常退出。调用 `tools/manage.py stop` 时，先向受管进程发送 SIGTERM，35 秒未退出时重新核对身份，再结束该进程并报告 `forced: true`。

## 模型切换与维护

对外只有 `GET /api/status`、`POST /api/models/select` 和 `POST /infer`。状态查询 HTTP 200 表示 HTTP 可访问，是否可以推理要检查 `modelReady`；`models[].versions` 同时提供业务版本、文件可用性和张量信息。切换期间已有活动模型时 `state` 保持 `running`，另用 `switching`、`pendingModel` 表示准备中的目标；旧实例排空与卸载进度位于 `retiringModels`。

服务只创建一个 Triton Core。每次加载使用独立的内部模型名和临时仓库；仓库中的 `source` 符号链接指向正式模型目录，通过 Triton 仓库映射加载为独立名字。因此同一模型的不同版本和输入尺寸也能并存，不覆盖仍在执行的旧实例。成功切流后，后台线程等待旧实例的在途计数归零，再卸载模型、注销临时仓库并删除临时目录。`retiringModels` 在这些步骤全部完成后才移除；清理失败记录为 `cleanupError` 并重试，不影响新模型接收推理。

准备新模型时显存需同时容纳新旧实例，加载和预热也可能导致推理延迟波动。显存不足或候选验证失败只清理候选，旧实例继续服务。不要在服务运行时覆盖或删除 Engine 文件；版本目录和配置变更仍应通过停止进程、部署、重启和真实图片验证完成。

### best 2.0.0 动态合批

`best` 业务版本 `2.0.0` 映射到 `models/model_build/best/2/model.plan`，原 `1.0.0` 对应 `models/model_build/best/1/model.plan`。Engine 从 `models/model_sources/best.onnx` 构建，保持 FP32、`--noTF32`，输入宽高固定 960，batch profile 为 min=1、opt=4、max=8。以下命令输出到独立的构建路径：

```bash
cd /opt/triton_back
source /opt/triton/2.71.0/setup_env.sh
mkdir -p build/model-export
trtexec \
  --onnx=models/model_sources/best.onnx \
  --minShapes=images:1x3x960x960 \
  --optShapes=images:4x3x960x960 \
  --maxShapes=images:8x3x960x960 \
  --memPoolSize=workspace:1024 --noTF32 \
  --profilingVerbosity=detailed \
  --saveEngine=build/model-export/best-batch8.plan --skipInference
```

构建成功后仍需实际执行 batch 1、2、3、4、8，检查输出为 `[N,300,6]`，并逐图对照旧 Engine。历史构建与数值验证记录曾保存在本地 `build/dynamic-batch-20261016`。发布经验证的正式 `model.plan` 时，更新 `models/model_build/manifest.json` 中的版本、路径与 SHA256。

`config/models.json` 中 `best/2.0.0` 的完整注册记录沿用原类别映射，并增加以下运行字段：

```json
{
  "instanceCount": 1,
  "batching": {
    "maxBatchSize": 8,
    "preferredBatchSizes": [4],
    "maxQueueDelayMicroseconds": 0
  }
}
```

当前以 4 张为首选批量，凑批等待预算为 0 微秒。每个请求仍只包含一张 PNG，Triton 根据已排队的请求组成可执行批量，批次可能包含不同相机或拍次。GPU 忙碌、前处理和传输仍会增加延迟。历史四相机压测使用 10000 微秒的凑批等待预算。

相机数变化时可调整 `preferredBatchSizes`，其中每个值必须为正数、互不重复并且不超过 `maxBatchSize`。当前 Engine 支持 batch 1–8；`maxBatchSize` 超过 8 或更改固定输入尺寸必须重建 Engine。`instanceCount` 配置独立 GPU 执行实例数，增加实例会消耗更多显存，并可能改变合批效果，需实测。旧单图版本不设置 `batching`，默认 `maxBatchSize: 0`，不启用动态合批。

`models/model_build/best/config.pbtxt` 仍是旧版单图的静态后备配置。服务按 `config/models.json` 生成指定版本的 Triton 配置。动态版设置 `max_batch_size: 8`，输入配置省略 batch 轴，使用 `[3,960,960]`，输出使用 `[300,6]`。Triton 元数据中的完整形状为 `[-1,3,960,960]` 和 `[-1,300,6]`。调整服务的合批行为时修改 `config/models.json`。

修改 YAML、模型注册或 batching 字段后，重启整个 C++ 进程；选择接口不会重新读取配置。重启后通过既有 `POST /api/models/select` 指定 `best`、`2.0.0`。合批是否生效应读取 `GET /api/status` 中的 `activeModel.executionStatistics.model_stats`：`inference_count` 是处理的样本数，`execution_count` 是实际执行次数，`batch_stats[].batch_size` 及对应 `compute_infer.count` 表示各实际批量执行了多少次。统计按当前运行代累计且包含预热；测量某个窗口时取前后差值，切换或重新加载后重新取基线，耗时 `ns` 为累计纳秒。

### best 3.0.0 可选 TF32 版本

`best/3.0.0` 映射到 `models/model_build/best/3/model.plan`。其输入输出类型、类别映射、最大及首选批量与 `2.0.0` 相同，合批等待预算当前为 10000 微秒。构建启用 TF32 时，按本机 TensorRT 11.1 的 `trtexec --help` 使用默认设置，并确保环境没有 `NVIDIA_TF32_OVERRIDE=0`。该 Engine 的 Inspector 中有 110 个执行策略名称明确含 `tf32`，也有 FP32 等策略。更多细节见 [NVIDIA 精度说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html#tf32)。

```bash
cd /opt/triton_back
source /opt/triton/2.71.0/setup_env.sh
mkdir -p build/model-export
env -u NVIDIA_TF32_OVERRIDE /usr/bin/trtexec \
  --onnx=models/model_sources/best.onnx \
  --minShapes=images:1x3x960x960 \
  --optShapes=images:4x3x960x960 \
  --maxShapes=images:8x3x960x960 \
  --memPoolSize=workspace:1024 --profilingVerbosity=detailed \
  --saveEngine=build/model-export/best-tf32-batch8.plan --skipInference
```

通过选择接口传 `{"ModelId":"best","ModelVersion":"3.0.0"}`，或在启动命令传 `--model-id best --model-version 3.0.0`。原 FP32 模型仍可通过 `1.0.0`、`2.0.0` 选择。修改配置后需要重启进程。

本次全 377 张 PNG 的串行对照未通过旧 FP32 的置信度 `1e-5` / 坐标 `0.001 px` 数值阈值：5 张图片的数量和类别分布发生变化；其余结构一致的图片用同类别最大 IoU 一对一匹配，对应框的最大置信度差约 `0.08845`，最大坐标差约 `10.29974 px`。最大值包含所有有效对应框，未按严格容差过滤；结构变化的图片单列，不强行数值配对。无标注数据，不能据此断言准确率相同或下降。速度与差异摘要见 [tf32-20261016.json](benchmarks/tf32-20261016.json)。

测试中的 `expected-best-tf32.json` 是该 TF32 Engine 对固定 `best.raw` 直接推理的参考，覆盖 batch 1–8 的每个位置，记录 Engine 和像素 SHA256。它检验服务传输与该 Engine 的一致性，不是放宽旧 FP32 阈值或证明检测准确率。重建 TF32 Engine 后应检查精度及标注数据，再重新生成这一版本的参考；不要修改原 `expected-baseline.json` 来让 TF32 通过。

参考导出器不参与生产链接，另需 OpenSSL 开发头文件和 `libcrypto`。在完成项目构建、更新新 Engine 对应的 manifest 后，导出到独立路径供审查；导出器会拒绝覆盖已有文件：

```bash
mkdir -p build/model-export
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
  -Isrc -I/usr/local/cuda-13.3/include \
  tests/integration/export_tf32_reference.cpp build/libimage_processing.a \
  -o build/model-export/export_tf32_reference \
  $(pkg-config --cflags --libs opencv4) \
  -L/usr/local/cuda-13.3/lib64 -Wl,-rpath,/usr/local/cuda-13.3/lib64 \
  -lcudart -lnvinfer -lnvinfer_plugin -lcrypto
build/model-export/export_tf32_reference \
  models/model_build/best/3/model.plan build/model-export/expected-best-tf32.json
```

数值差异和标注验证完成后，再备份并更新 `tests/fixtures/expected-best-tf32.json`。`test_business_reference.py` 检查旧 FP32 严格失败不会被 TF32 功能验证掩盖，也检查身份、超容差和无效结果会被拒绝。

## 验证

```bash
cd /opt/triton_back
ctest --test-dir build -L unit --output-on-failure
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_tools.py
PYTHONDONTWRITEBYTECODE=1 python3 tests/integration/test_business_reference.py
ctest --test-dir build -L gpu --output-on-failure
```

GPU 测试需要 `models/model_build/` 中两个模型的 Engine 和可用 GPU。真实业务协议测试使用 `tests/protocol/testProtocol.py`，会执行模型加载与切换，应在验证窗口内运行。
