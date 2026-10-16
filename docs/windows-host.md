# Windows 上位机启动推理后端

推荐上位机启动并保留一个前台 `wsl.exe` 进程，由 WSL 的 `run.sh` 最终 exec C++ 服务。HTTP 通信地址默认 `http://127.0.0.1:6577`。运行时无需 Windows Python、旧 Python 控制服务或 D 盘源码；更新与构建是单独的部署动作。

本机验证使用现有 PowerShell 7 会话。系统 `powershell.exe` 5.1 当前脚本执行策略阻止 `.ps1` 运行；没有修改执行策略。可在可运行脚本的现有 PowerShell 7 中执行下述命令，C# 直接启动 `wsl.exe` 不受 PowerShell 脚本策略影响。

## 已提供的 PowerShell 工具

```powershell
D:\codes\triton_back\scripts\windows\backend.ps1 -Action Start -Distribution template-clone -ModelId best -ModelVersion 2.0.0
D:\codes\triton_back\scripts\windows\backend.ps1 -Action Status -Distribution template-clone
D:\codes\triton_back\scripts\windows\backend.ps1 -Action Stop -Distribution template-clone
```

Start 在后台持有前台 WSL/C++ 进程并隐藏窗口；等待 `GET /api/status`，核对调用者 token。提供 ModelId 和 ModelVersion 时，通过 `POST /api/models/select` 加载目标，并核对返回的 `modelReady: true` 及 `activeModel`。省略这两个参数时只等待 HTTP 可查询并返回状态和目录，不隐式选择模型。重复 Start 复用已验证的该项目进程；没有模型参数就保持其当前选择。启动工具的结果统一放在 `apiStatus` 中。

`best/2.0.0` 为 FP32 动态合批版；需要测试可选 TF32 版时，将启动参数或选择请求中的版本改为 `3.0.0`。TF32 的传输类型仍为 FP32、每请求仍为一张 PNG；它与 FP32 的检测结果存在数值和少量检测数量变化，见 [部署说明](deployment.md#best-300-可选-tf32-版本) 和 [benchmark 摘要](benchmarks/tf32-20261016.json)。原 `1.0.0`、`2.0.0` 均保留。

C++ stdout/stderr 直接写入 `/opt/triton_back/runtime/logs/service.log`，不会因 PowerShell 退出后无人读取重定向管道而堵塞；Windows 中的日志记录启动器输出。Stop 校验调用者 token 和 Linux 进程身份，先正常停止，必要时仅终止同一管理进程。

工具使用原生 CreateProcess 的限定句柄列表，只传入 NUL stdin 和两个启动器日志文件，避免继承父会话的管道。后台仍由一个前台 WSL 命令持有 C++ 生命周期。

## C# 集成

下面展示 .NET 6+ 的启动关键路径。将 `backendProcess` 和 `ownerToken` 保存为上位机成员；两个流持续读取以获得启动器异常。C++ 完整日志保存在 WSL 原生文件中，可另行读取该文件。

```csharp
using System.Diagnostics;
using System.Net.Http;
using System.Text;
using System.Text.Json;

string distribution = "template-clone";
string ownerToken = Guid.NewGuid().ToString("N");
var start = new ProcessStartInfo("wsl.exe") {
    UseShellExecute = false,
    CreateNoWindow = true,
    RedirectStandardOutput = true,
    RedirectStandardError = true
};
foreach (string argument in new[] {
    "-d", distribution, "-u", "root", "--cd", "/opt/triton_back",
    "--exec", "/usr/bin/env", "TRITON_BACK_OWNER_TOKEN=" + ownerToken,
    "bash", "scripts/wsl/run.sh"
}) start.ArgumentList.Add(argument);

var backendProcess = new Process { StartInfo = start };
backendProcess.OutputDataReceived += (_, e) => { if (e.Data != null) Log(e.Data); };
backendProcess.ErrorDataReceived += (_, e) => { if (e.Data != null) Log(e.Data); };
backendProcess.Start();
backendProcess.BeginOutputReadLine();
backendProcess.BeginErrorReadLine();

using var http = new HttpClient { BaseAddress = new Uri("http://127.0.0.1:6577") };
http.Timeout = TimeSpan.FromSeconds(90);
using var startup = new CancellationTokenSource(TimeSpan.FromSeconds(90));
while (true) {
    startup.Token.ThrowIfCancellationRequested();
    if (backendProcess.HasExited)
        throw new InvalidOperationException("后端启动失败，请查看日志");
    try {
        using var status = await http.GetAsync("/api/status", startup.Token);
        if (status.IsSuccessStatusCode) break;
    } catch (HttpRequestException) { }
    await Task.Delay(200, startup.Token);
}
var identityStart = new ProcessStartInfo("wsl.exe") {
    UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true
};
foreach (string argument in new[] {
    "-d", distribution, "-u", "root", "--exec", "python3",
    "/opt/triton_back/tools/manage.py", "status"
}) identityStart.ArgumentList.Add(argument);
using var identityProcess = Process.Start(identityStart)!;
string identityJson = await identityProcess.StandardOutput.ReadToEndAsync();
await identityProcess.WaitForExitAsync(startup.Token);
if (identityProcess.ExitCode != 0)
    throw new InvalidOperationException("后端进程身份检查失败");
using var identity = JsonDocument.Parse(identityJson);
if (identity.RootElement.GetProperty("process").GetProperty("ownerToken").GetString() != ownerToken)
    throw new InvalidOperationException("端口对应的服务不属于本次启动");
using var catalogResponse = await http.GetAsync("/api/status", startup.Token);
catalogResponse.EnsureSuccessStatusCode();
string catalogJson = await catalogResponse.Content.ReadAsStringAsync();
// 展示 catalogJson 中的 models[].versions，下面使用用户选择的值。
string selectedModelId = "best", selectedModelVersion = "2.0.0";
string selectionJson = JsonSerializer.Serialize(new {
    ModelId = selectedModelId, RequestedModelVersion = selectedModelVersion
});
using var loaded = await http.PostAsync("/api/models/select",
    new StringContent(selectionJson, Encoding.UTF8, "application/json"), startup.Token);
string loadedJson = await loaded.Content.ReadAsStringAsync();
if (!loaded.IsSuccessStatusCode)
    throw new InvalidOperationException("模型加载失败：" + loadedJson);
using var loadedStatus = JsonDocument.Parse(loadedJson);
var active = loadedStatus.RootElement.GetProperty("activeModel");
if (!loadedStatus.RootElement.GetProperty("modelReady").GetBoolean() ||
    active.GetProperty("id").GetString() != selectedModelId ||
    active.GetProperty("version").GetString() != selectedModelVersion)
    throw new InvalidOperationException("返回的活动模型与指定目标不一致");
```

`Log` 替换为上位机日志方法；回调在后台线程执行，更新 UI 时需使用界面的线程调度机制。生产上位机在启动前应调用 `tools/manage.py status` 判断是否已有该项目进程；已有进程时可复用 API，避免再启动一份占用相同端口。

结束自己启动的后端，执行以下命令，并等它及 `backendProcess` 退出。使用 `ProcessStartInfo.ArgumentList` 逐项传参，其中 token 使用启动时保存的值：

```text
wsl.exe -d template-clone -u root --exec python3 /opt/triton_back/tools/manage.py stop --owner-token <启动时的ownerToken>
```

Linux helper 只接受匹配 token、PID 启动时间及命令行的进程；最多 35 秒等待正常排空，然后再次核对身份才强制终止。复用已有服务的上位机不拥有该服务，退出连接时不应结束其他上位机持有的进程。模型由切换流程自动清理，无需上位机主动卸载。

## HTTP 调用顺序

| 接口 | 含义 |
|---|---|
| `GET /api/status` | 一次查询服务状态、所有模型版本、张量及协议能力、当前选择和切换进度；不加载模型 |
| `POST /api/models/select` | 明确指定 ModelId、RequestedModelVersion，首次加载与运行中切换共用此入口 |
| `POST /infer` | 唯一推理入口，multipart `metadata` 和 PNG 文件二进制 `image`；兼容 Raw |

`GET /api/status` 始终以 HTTP 200 返回状态；200 表示 HTTP 可访问，`modelReady: true` 才表示当前有模型可以推理。上位机不需要独立模型停止接口；切换自动清理旧模型，关闭后端进程仍使用部署工具的 Stop。

本次接口收敛有兼容性变更：旧 `/health`、`/api/health`、`/api/info`、`/api/models`、`/api/start`、`/api/stop` 路由已移除，调用返回 404。旧查询能力统一读取 `/api/status`，其 `models` 从扁平列表改为按 ID 分组的 `models[].versions`；首次加载也必须向选择接口明确传入两个字段。旧启动脚本输出中的 `readiness.health` 改为 `apiStatus`，上位机需要同步更新。

上位机保持比例缩放后编码为 PNG 上传；服务解码后完成 letterbox 和模型前处理，再把检测框还原到上传 PNG 的像素坐标。每帧携带 RequestId、FrameId、CameraId 等标识，结果回显用来对应相机和帧。遇到 503 SERVICE_BUSY 时由上位机执行队列限流或丢弃过期帧，避免无界积压。

## 模型版本查询、选择与指定启动

查询 `GET /api/status`，`models` 按业务模型分组，`versions` 数组包含每个已注册版本的 `version`、`tritonVersion`、`inputSize`、`classCount`、`available`、`active`、`ready`，以及 `inputs`、`outputs` 张量名称、类型和形状。`available` 表示相应 `model.plan` 文件存在，不表示已经通过加载和准确率验证。`activeModel` 为当前接收新请求的模型，首次启动尚未选择时为 null。

当前 `best/2.0.0` 对应动态 batch Engine，固定 960 输入、FP32、禁用 TF32，支持 batch 1–8，最佳构建批量为 4。状态的 `batching` 包含 `maxBatchSize`、`preferredBatchSizes`、`maxQueueDelayMicroseconds`，`instanceCount` 显示 GPU 实例数；默认分别为 8、`[4]`、10000 微秒及 1 个实例。`inputs.shape` 为 `[-1,3,960,960]`，`outputs.shape` 为 `[-1,300,6]`，`-1` 是内部动态批量轴。`batchSize: 1` 表示每个 HTTP 请求仍是一张图，不表示 GPU 每次只执行一张；原 `best/1.0.0` 继续可选。

| 状态字段 | 含义 |
|---|---|
| `serviceAlive`、`tritonLive` | HTTP 服务与进程内 Triton Core 存活状态 |
| `state`、`modelReady`、`activeModel` | 运行阶段、能否推理和当前模型版本 |
| `models[].versions` | 所有注册模型版本、文件可用性、张量契约和活动状态 |
| `apiVersion`、`triton` | Triton Core API 版本与服务元数据 |
| `schemaVersion`、`imageEncodings`、`pixelFormats` | 业务协议版本、支持的图像编码和 Raw 像素格式 |
| `uptimeMs`、`heartbeatIntervalMs` | HTTP 已运行时间与建议状态查询间隔 |
| `inFlight`、`lastError` | 已接收的在途请求总数和最近一次模型加载错误 |
| `switching`、`pendingModel` | 是否正在准备模型，以及准备中的目标 |
| `retiringModels` | 不再接收新任务的旧实例，含 `generation`、`inFlight`、`unloading`、`cleanupError` |
| `activeModel.batching`、`activeModel.instanceCount` | 当前版本的合批参数和 GPU 执行实例数 |
| `activeModel.executionStatistics` | 当前运行代的 Triton 实际推理及批量执行统计，包含加载时预热 |

`executionStatistics.model_stats[].inference_count` 为处理样本数，`execution_count` 为执行次数；`batch_stats[].batch_size` 与 `compute_infer.count` 可确认实际执行批量及次数。对某段运行取前后差值，不能把进程启动以来的累计值或客户端并发数直接当作该段批量；重新加载会产生新运行代并重新累计。各阶段 `ns` 是累计纳秒。

加载或切换请求使用 `application/json`：

```http
POST /api/models/select
Content-Type: application/json

{"ModelId":"best","RequestedModelVersion":"2.0.0"}
```

成功返回与查询接口相同的完整状态 JSON，`activeModel` 指明生效的模型和版本。同一个版本已经活动时直接成功。`POST /api/models/select {}` 是错误请求，两个选择字段都必须明确提供，不再恢复上次选择或隐式选择目录第一项。

切换分为准备、切流和清理：新模型加载、契约校验及预热时，旧模型继续接收和处理推理，`state` 仍为 `running`，`switching: true`，`pendingModel` 为目标。首次加载尚无旧模型时为 `starting`、`modelReady: false`。目标完全准备成功后，服务原子更新 `activeModel`，后续新请求使用新模型；已经接收的旧请求仍按旧模型完成。旧实例在途计数归零后自动后台卸载，完成物理卸载和仓库注销后才从 `retiringModels` 移除，选择请求无需等待旧实例清理。

目标加载失败时不改变旧实例，无需重新加载旧模型。错误响应包含实际 `activeModel`、`modelReady` 和 `lastError`。旧实例的清理错误单独放在 `retiringModels[].cleanupError` 并重试，不把已生效的新模型改判为失败。切换期间需要容纳新旧实例的显存，并可能产生延迟波动；显存不足会返回错误并保持旧模型。

推理 metadata 必须继续携带当前 ModelId、RequestedModelVersion。准备期间仍发送旧模型选择；切流后新提交的旧选择返回 `409 MODEL_NOT_ACTIVE`，已经接收的旧请求不受影响。上位机应在选择成功后更新后续帧的模型字段，并保留各请求原有版本关联。并发提交撞上切流边界时，查询状态确认当前选择，再按业务决定是否重新提交该帧，避免把旧帧任务静默改用其他模型。

运行期间切换也使用同一个接口，以下代码沿用启动示例中的 `http` 和选择变量：

```csharp
string nextModelId = "yolo26n", nextModelVersion = "1.0.0";
string switchJson = JsonSerializer.Serialize(new {
    ModelId = nextModelId, RequestedModelVersion = nextModelVersion
});
using var switched = await http.PostAsync("/api/models/select",
    new StringContent(switchJson, Encoding.UTF8, "application/json"));
string switchResult = await switched.Content.ReadAsStringAsync();
if (!switched.IsSuccessStatusCode)
    throw new InvalidOperationException("切换失败，原模型保持活动：" + switchResult);
using var switchStatus = JsonDocument.Parse(switchResult);
var nextActive = switchStatus.RootElement.GetProperty("activeModel");
if (!switchStatus.RootElement.GetProperty("modelReady").GetBoolean() ||
    nextActive.GetProperty("id").GetString() != nextModelId ||
    nextActive.GetProperty("version").GetString() != nextModelVersion)
    throw new InvalidOperationException("返回的活动模型与切换目标不一致");
selectedModelId = nextModelId;
selectedModelVersion = nextModelVersion;
```

| HTTP / error | 含义 |
|---|---|
| 404 MODEL_NOT_FOUND | 业务模型 ID 未注册 |
| 404 MODEL_VERSION_NOT_FOUND | 该模型的业务版本未注册 |
| 404 MODEL_FILE_NOT_FOUND | 已注册，但对应整数版本目录的 model.plan 不存在 |
| 400 INVALID_REQUEST / INVALID_METADATA | 选择字段缺失、类型错误或空字符串 |
| 500 MODEL_CONTROL_FAILED | 候选模型加载、契约检查或预热失败；原活动模型保持不变 |
| 503 BACKEND_NOT_READY | 首次加载尚未成功、尚无活动模型，或进程正在退出 |
| 409 MODEL_NOT_ACTIVE | 推理请求选择与当前活动模型版本不符 |

版本注册仍使用 config/models.json：复制一条完整模型记录，保留相同 id 和 tritonName，设置新的 businessVersion、tritonVersion，并核对 inputSize、classNames。业务版本和整数版本组合必须唯一，Engine 放在 models/<tritonName>/<tritonVersion>/model.plan。未注册的磁盘目录不会成为可选择的业务版本；新增注册记录后重启进程。

指定版本加载由服务按 `config/models.json` 生成完整 Triton 配置，包括 FP32 images/output0 契约、指定版本、batching 和 instanceCount，覆盖仓库 config.pbtxt；后者保留旧版静态单图配置作为后备。每次加载具有独立的内部实例，允许相同模型的不同版本在切换期间共存。修改合批或实例字段后需重启整个 C++ 进程，选择接口不重新读取配置；超过当前 Engine 的 batch 上限 8 需重建模型。不要在运行期间覆盖旧实例仍可能使用的 Engine 文件。完整配置与重建命令见 [deployment.md](deployment.md)。

若上位机在启动前已知目标，也可以通过进程参数直接加载：

```text
wsl.exe -d template-clone -u root --cd /opt/triton_back --exec bash scripts/wsl/run.sh --model-id best --model-version 2.0.0
```

两个参数必须同时提供。底层 C++ 同样支持 `triton_backend --config /opt/triton_back/config/server.yaml --model-id best --model-version 2.0.0`。指定模型加载成功后才开放 HTTP；模型或版本不存在时输出带错误码的消息并以非零状态退出。上位机使用 ProcessStartInfo.ArgumentList 逐项传递参数，并保存 owner token 及前台 WSL 进程。

## PNG 推理请求

`POST /infer` 使用 `multipart/form-data`，且必须声明完整 `Content-Length`。只提供以下两部分：

| 部分 | Content-Type | 内容 |
|---|---|---|
| `metadata` | `application/json; charset=utf-8` | 下述 JSON |
| `image` | `image/png` 或 `application/octet-stream` | 完整 PNG 文件的二进制字节 |

```json
{
  "SchemaVersion": "1.0",
  "RequestId": "camera-1-frame-42",
  "Sequence": 42,
  "FrameId": 42,
  "CameraId": "camera-1",
  "ChannelId": "1",
  "CapturedAtUtc": "2026-10-14T01:02:03.123Z",
  "Width": 800,
  "Height": 800,
  "SourcePixelFormat": "Mono8",
  "ImageEncoding": "Png",
  "ModelId": "best",
  "RequestedModelVersion": "2.0.0",
  "ConfidenceThreshold": 0.25
}
```

除 `ConfidenceThreshold` 可省略（默认 0.25）外，示例字段均为必填。`Width`、`Height` 为 PNG 本身的像素尺寸，不是缩放前相机尺寸，也不是模型输入尺寸。`SourcePixelFormat` 记录相机源像素格式，实际颜色解码由 PNG 文件决定；`PixelFormat`、`StrideBytes`、`RowOrder` 不必提供。标识字段按原值回显，检测框位于上传 PNG 的坐标系。

使用动态版时 metadata 的 `RequestedModelVersion` 保持与当前选择的 `2.0.0` 一致，每个请求仍只上传一张 PNG，无需把四张图拼接成一张。四路相机可以并发提交各自请求，Triton 尽量合成 4 张；10 毫秒等待预算耗尽可执行小批次，不保证四张来自不同相机或同一拍次。每个响应仍只对应自己的 RequestId、CameraId、FrameId，客户端需按这些字段关联结果。相机数及等待策略由部署配置的 `batching` 决定，不通过单帧 metadata 修改。

支持每通道最多 8 位的灰度、彩色和带透明通道 PNG。透明通道被忽略，不与背景合成；不应用 EXIF 方向旋转。16 位 PNG 会被拒绝，避免静默改变相机数值。PNG 解码后的三通道 BGR 像素最多 64 MiB，整个 HTTP 请求默认最多 64 MiB。非法、损坏、超限、位深不支持或宽高不匹配的 PNG 返回 `400 INVALID_PNG_IMAGE`。

在前面的 `HttpClient` 启动示例基础上发送 PNG：

```csharp
byte[] pngBytes = await File.ReadAllBytesAsync(pngPath); // 完整 PNG 文件字节
string metadataJson = JsonSerializer.Serialize(new {
    SchemaVersion = "1.0", RequestId = Guid.NewGuid().ToString("N"),
    Sequence = (ulong)42, FrameId = (ulong)42,
    CameraId = "camera-1", ChannelId = "1", CapturedAtUtc = DateTime.UtcNow,
    Width = pngWidth, Height = pngHeight, SourcePixelFormat = "Mono8",
    ImageEncoding = "Png", ModelId = selectedModelId, RequestedModelVersion = selectedModelVersion,
    ConfidenceThreshold = 0.25
});
using var form = new MultipartFormDataContent();
form.Add(new StringContent(metadataJson, Encoding.UTF8, "application/json"), "metadata");
var image = new ByteArrayContent(pngBytes);
image.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("image/png");
form.Add(image, "image", "frame.png");
using var result = await http.PostAsync("/infer", form);
string resultJson = await result.Content.ReadAsStringAsync();
result.EnsureSuccessStatusCode();
```

`pngPath`、`pngWidth`、`pngHeight` 替换为实际文件路径和 PNG 尺寸。`ByteArrayContent` 和 `StringContent` 的长度已知，`HttpClient` 会计算 multipart 的 `Content-Length`；不要显式启用 chunked。也可直接使用上位机在内存中生成的 PNG 字节，无需落盘。

旧的原始像素调用仍可使用 `ImageEncoding: "Raw"`，`image` 类型为 `application/octet-stream`，并提供 `PixelFormat`（Mono8/RGB8Packed/BGR8Packed）、`StrideBytes`、`RowOrder`（TopDown/BottomUp）。`GET /api/status` 的 `imageEncodings` 返回支持的编码。
