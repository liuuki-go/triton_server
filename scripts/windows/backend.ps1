[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Sync', 'Build', 'Start', 'Stop', 'Status')]
    [string]$Action,
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')][string]$Distribution = 'template-clone',
    [ValidateRange(1, 16)][int]$Jobs = 4,
    [switch]$ReplaceConfig,
    [string]$PythonExe,
    [string]$ModelId,
    [string]$ModelVersion,
    [ValidateRange(5, 300)][int]$StartupTimeoutSeconds = 90
)
$ErrorActionPreference = 'Stop'
if (($PSBoundParameters.ContainsKey('ModelId') -or $PSBoundParameters.ContainsKey('ModelVersion')) -and
    ([string]::IsNullOrWhiteSpace($ModelId) -or [string]::IsNullOrWhiteSpace($ModelVersion) -or
     $ModelId.Length -gt 256 -or $ModelVersion.Length -gt 256)) {
    throw 'ModelId and ModelVersion must both be nonempty strings of at most 256 characters.'
}
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$NativeRoot = '/opt/triton_back'
$Utf8 = New-Object System.Text.UTF8Encoding($false)
$DistroHash = [System.Security.Cryptography.SHA256]::Create()
try { $DistroKey = ([System.BitConverter]::ToString($DistroHash.ComputeHash($Utf8.GetBytes($Distribution))).Replace('-', '').Substring(0, 16)).ToLowerInvariant() }
finally { $DistroHash.Dispose() }
$WindowsRuntime = Join-Path $ProjectRoot ('runtime\windows\' + $DistroKey)
$PidFile = Join-Path $WindowsRuntime 'service.pid.json'

function Invoke-Wsl([string[]]$Arguments) {
    $Lines = & wsl.exe -d $Distribution -u root --exec @Arguments
    if ($LASTEXITCODE -ne 0) { throw "WSL command failed with exit code $LASTEXITCODE" }
    return ($Lines -join "`n")
}

function Invoke-Manage([string]$Operation, [string]$OwnerToken) {
    $Arguments = @('python3', "$NativeRoot/tools/manage.py", $Operation, '--target', $NativeRoot)
    if (-not [string]::IsNullOrWhiteSpace($OwnerToken)) { $Arguments += @('--owner-token', $OwnerToken) }
    return (Invoke-Wsl -Arguments $Arguments | ConvertFrom-Json)
}

function Read-WindowsIdentity([int]$ProcessId) {
    $Process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $Process) { return $null }
    return @{ pid = $Process.Id; creationTimeTicks = $Process.StartTime.ToUniversalTime().Ticks; name = $Process.ProcessName }
}

function Wait-Endpoint([string]$Url, [System.Diagnostics.Process]$Child) {
    $Deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
    while ([DateTime]::UtcNow -lt $Deadline) {
        if ($null -ne $Child -and $Child.HasExited) { throw "WSL foreground process exited with code $($Child.ExitCode); inspect /opt/triton_back/runtime/logs/service.log and $WindowsRuntime" }
        try { return Invoke-RestMethod -Uri $Url -TimeoutSec 2 }
        catch { Start-Sleep -Milliseconds 200 }
    }
    throw "Backend did not become reachable: $Url; inspect /opt/triton_back/runtime/logs/service.log and $WindowsRuntime"
}

function Initialize-ModelSelection([string]$Url) {
    if ([string]::IsNullOrWhiteSpace($ModelId)) { return Wait-Endpoint "$Url/api/status" $null }
    $Body = @{ ModelId = $ModelId; RequestedModelVersion = $ModelVersion } | ConvertTo-Json -Compress
    $Status = Invoke-RestMethod -Uri "$Url/api/models/select" -Method Post -ContentType 'application/json' -Body $Body -TimeoutSec $StartupTimeoutSeconds
    if (-not $Status.modelReady -or $Status.activeModel.id -cne $ModelId -or $Status.activeModel.version -cne $ModelVersion) {
        throw 'The requested model and version did not become active; inspect the backend status and lastError.'
    }
    return $Status
}

function Write-Record($Record) {
    [System.IO.Directory]::CreateDirectory($WindowsRuntime) | Out-Null
    [System.IO.File]::WriteAllText($PidFile, ($Record | ConvertTo-Json -Depth 12), $Utf8)
}

function Start-ForegroundWsl([string]$ArgumentText, [string]$StdoutPath, [string]$StderrPath) {
    if (-not ('TritionBackWindows.NativeProcess' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
namespace TritionBackWindows {
    public static class NativeProcess {
        [StructLayout(LayoutKind.Sequential)] struct SecurityAttributes {
            public int Length; public IntPtr Descriptor; public int InheritHandle;
        }
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)] struct StartupInfo {
            public int Size; public string Reserved; public string Desktop; public string Title;
            public int X; public int Y; public int XSize; public int YSize;
            public int XCount; public int YCount; public int FillAttribute; public int Flags;
            public short ShowWindow; public short ReservedSize; public IntPtr ReservedBytes;
            public IntPtr StandardInput; public IntPtr StandardOutput; public IntPtr StandardError;
        }
        [StructLayout(LayoutKind.Sequential)] struct StartupInfoEx {
            public StartupInfo Startup; public IntPtr Attributes;
        }
        [StructLayout(LayoutKind.Sequential)] struct ProcessInformation {
            public IntPtr Process; public IntPtr Thread; public int ProcessId; public int ThreadId;
        }
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        static extern IntPtr CreateFile(string name, uint access, uint share,
            ref SecurityAttributes security, uint creation, uint flags, IntPtr template);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool InitializeProcThreadAttributeList(IntPtr list, int count, int flags, ref IntPtr size);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool UpdateProcThreadAttribute(IntPtr list, uint flags, IntPtr attribute,
            IntPtr value, IntPtr size, IntPtr previous, IntPtr returned);
        [DllImport("kernel32.dll")]
        static extern void DeleteProcThreadAttributeList(IntPtr list);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        static extern bool CreateProcess(string application, StringBuilder command,
            IntPtr processSecurity, IntPtr threadSecurity, bool inherit, uint flags,
            IntPtr environment, string directory, ref StartupInfoEx startup,
            out ProcessInformation information);
        [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);

        public static Process Start(string application, string arguments, string stdout, string stderr) {
            var security = new SecurityAttributes {
                Length = Marshal.SizeOf(typeof(SecurityAttributes)), InheritHandle = 1
            };
            IntPtr input = IntPtr.Zero, output = IntPtr.Zero, error = IntPtr.Zero;
            IntPtr attributes = IntPtr.Zero, handles = IntPtr.Zero;
            bool initialized = false;
            try {
                input = CreateFile("NUL", 0x80000000, 3, ref security, 3, 0, IntPtr.Zero);
                output = CreateFile(stdout, 0x40000000, 3, ref security, 2, 0, IntPtr.Zero);
                error = CreateFile(stderr, 0x40000000, 3, ref security, 2, 0, IntPtr.Zero);
                if (input == new IntPtr(-1) || output == new IntPtr(-1) || error == new IntPtr(-1))
                    throw new Win32Exception(Marshal.GetLastWin32Error());
                IntPtr size = IntPtr.Zero;
                InitializeProcThreadAttributeList(IntPtr.Zero, 1, 0, ref size);
                attributes = Marshal.AllocHGlobal(size);
                if (!InitializeProcThreadAttributeList(attributes, 1, 0, ref size))
                    throw new Win32Exception(Marshal.GetLastWin32Error());
                initialized = true;
                handles = Marshal.AllocHGlobal(3 * IntPtr.Size);
                Marshal.WriteIntPtr(handles, 0, input);
                Marshal.WriteIntPtr(handles, IntPtr.Size, output);
                Marshal.WriteIntPtr(handles, 2 * IntPtr.Size, error);
                if (!UpdateProcThreadAttribute(attributes, 0, new IntPtr(0x20002), handles,
                    new IntPtr(3 * IntPtr.Size), IntPtr.Zero, IntPtr.Zero))
                    throw new Win32Exception(Marshal.GetLastWin32Error());
                var startup = new StartupInfoEx();
                startup.Startup.Size = Marshal.SizeOf(typeof(StartupInfoEx));
                startup.Startup.Flags = 0x100;
                startup.Startup.StandardInput = input;
                startup.Startup.StandardOutput = output;
                startup.Startup.StandardError = error;
                startup.Attributes = attributes;
                ProcessInformation information;
                uint flags = 0x08000000 | 0x00000200 | 0x00080000 | 0x01000000;
                string command = "\"" + application + "\" " + arguments;
                bool started = CreateProcess(application, new StringBuilder(command), IntPtr.Zero,
                    IntPtr.Zero, true, flags, IntPtr.Zero, null, ref startup, out information);
                if (!started && Marshal.GetLastWin32Error() == 5)
                    started = CreateProcess(application, new StringBuilder(command), IntPtr.Zero,
                        IntPtr.Zero, true, flags & ~0x01000000u, IntPtr.Zero, null,
                        ref startup, out information);
                if (!started) throw new Win32Exception(Marshal.GetLastWin32Error());
                try { return Process.GetProcessById(information.ProcessId); }
                finally { CloseHandle(information.Thread); CloseHandle(information.Process); }
            } finally {
                if (initialized) DeleteProcThreadAttributeList(attributes);
                if (attributes != IntPtr.Zero) Marshal.FreeHGlobal(attributes);
                if (handles != IntPtr.Zero) Marshal.FreeHGlobal(handles);
                foreach (IntPtr handle in new[] { input, output, error })
                    if (handle != IntPtr.Zero && handle != new IntPtr(-1)) CloseHandle(handle);
            }
        }
    }
}
'@
    }
    $WslExe = (Get-Command wsl.exe -ErrorAction Stop).Source
    return [TritionBackWindows.NativeProcess]::Start($WslExe, $ArgumentText, $StdoutPath, $StderrPath)
}

if ($Action -eq 'Sync') {
    if ([string]::IsNullOrWhiteSpace($PythonExe)) {
        $Bundled = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
        if (Test-Path -LiteralPath $Bundled -PathType Leaf) { $PythonExe = $Bundled }
        else {
            $Available = Get-Command python.exe -ErrorAction SilentlyContinue
            if ($null -eq $Available -or $Available.Source -like '*\WindowsApps\*') {
                throw 'Sync requires a usable Windows Python 3. Pass -PythonExe C:\path\python.exe. Start, Stop and Status require no Windows Python.'
            }
            $PythonExe = $Available.Source
        }
    }
    $SyncArguments = @((Join-Path $ProjectRoot 'tools\sync_wsl.py'), '--source', $ProjectRoot, '--distribution', $Distribution)
    if ($ReplaceConfig) { $SyncArguments += '--replace-config' }
    & $PythonExe @SyncArguments
    if ($LASTEXITCODE -ne 0) { throw "Project sync failed with exit code $LASTEXITCODE" }
    exit 0
}
if ($Action -eq 'Build') {
    Invoke-Wsl -Arguments @('bash', "$NativeRoot/scripts/wsl/build.sh", "$Jobs")
    exit 0
}
if ($Action -eq 'Stop') {
    $Record = if (Test-Path -LiteralPath $PidFile) { Get-Content -LiteralPath $PidFile -Raw | ConvertFrom-Json } else { $null }
    $OwnerToken = if ($null -ne $Record) { $Record.ownerToken } else { $null }
    $Result = Invoke-Manage 'stop' $OwnerToken
    if ($null -ne $Record) {
        $Deadline = [DateTime]::UtcNow.AddSeconds(10)
        do {
            $Identity = Read-WindowsIdentity $Record.windowsPid
            if ($null -eq $Identity -or $Identity.creationTimeTicks -ne $Record.windowsCreationTimeTicks) {
                Remove-Item -LiteralPath $PidFile -Force
                break
            }
            Start-Sleep -Milliseconds 100
        } while ([DateTime]::UtcNow -lt $Deadline)
        if (Test-Path -LiteralPath $PidFile) { throw 'The owned Linux service stopped, but its foreground WSL process did not exit.' }
    }
    $Result | ConvertTo-Json -Depth 12
    exit 0
}
$Settings = Invoke-Manage 'settings'
$NativeStatus = Invoke-Manage 'status'
if ($Action -eq 'Status') {
    $Record = if (Test-Path -LiteralPath $PidFile) { Get-Content -LiteralPath $PidFile -Raw | ConvertFrom-Json } else { $null }
    $Identity = if ($null -ne $Record) { Read-WindowsIdentity $Record.windowsPid } else { $null }
    $Status = $null
    if ($null -ne $NativeStatus.process) { $Status = Invoke-RestMethod -Uri "$($Settings.url)/api/status" -TimeoutSec 5 }
    @{ distribution = $Distribution; settings = $Settings; linux = $NativeStatus; windows = $Identity; apiStatus = $Status } | ConvertTo-Json -Depth 12
    exit 0
}
if ($null -ne $NativeStatus.process) {
    $Status = Initialize-ModelSelection $Settings.url
    @{ alreadyRunning = $true; linux = $NativeStatus.process; apiStatus = $Status } | ConvertTo-Json -Depth 12
    exit 0
}
[System.IO.Directory]::CreateDirectory($WindowsRuntime) | Out-Null
$Owner = [Guid]::NewGuid().ToString('N')
$Stdout = Join-Path $WindowsRuntime 'service.stdout.log'
$Stderr = Join-Path $WindowsRuntime 'service.stderr.log'
# wsl.exe parses its distribution option itself; simple validated names need no quotes.
$Arguments = "-d $Distribution -u root --cd $NativeRoot --exec /usr/bin/env TRITON_BACK_OWNER_TOKEN=$Owner bash scripts/wsl/run.sh"
$Child = Start-ForegroundWsl $Arguments $Stdout $Stderr
$Identity = Read-WindowsIdentity $Child.Id
if ($null -eq $Identity) { throw "WSL exited during startup; inspect $WindowsRuntime" }
$Record = @{ windowsPid = $Child.Id; windowsCreationTimeTicks = $Identity.creationTimeTicks;
    distribution = $Distribution; ownerToken = $Owner; url = $Settings.url; nativeRoot = $NativeRoot;
    anchorKind = 'foreground-wsl-cpp'; stdoutLog = $Stdout; stderrLog = $Stderr;
    startedAtUtc = [DateTime]::UtcNow.ToString('o') }
Write-Record $Record
try {
    $Status = Wait-Endpoint "$($Settings.url)/api/status" $Child
    $Registered = Invoke-Manage 'status'
    if ($null -eq $Registered.process -or $Registered.process.ownerToken -ne $Owner) {
        throw 'The backend API did not belong to the foreground process started by this command.'
    }
    $Status = Initialize-ModelSelection $Settings.url
    $Record.linuxProcess = $Registered.process
    Write-Record $Record
    @{ alreadyRunning = $false; process = $Record; apiStatus = $Status } | ConvertTo-Json -Depth 12
} catch {
    # Only terminate the process with the newly issued ownership token.
    try {
        $Registered = Invoke-Manage 'status'
        if ($null -ne $Registered.process -and $Registered.process.ownerToken -eq $Owner) { Invoke-Manage 'stop' $Owner | Out-Null }
    } catch { Write-Warning $_.Exception.Message }
    throw
}
