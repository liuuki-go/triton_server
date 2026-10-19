<#
.SYNOPSIS
Manage dedicated Windows and Hyper-V firewall rules for this mirrored WSL backend.
.DESCRIPTION
Configure and Remove require an elevated PowerShell. Status and -WhatIf are read-only.
Run Configure again if the LAN adapter address or subnet changes. This script does not
change WSL networking, global firewall settings, network categories, services or portproxy.
.EXAMPLE
.\lan-access.ps1 -Action Configure -InterfaceAlias WLAN -WhatIf
.EXAMPLE
.\lan-access.ps1 -Action Configure -StateDirectory C:\triton-runtime\lan-access
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [ValidateSet('Configure', 'Status', 'Remove')][string]$Action = 'Configure',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')][string]$Distribution = 'template-clone',
    [ValidatePattern('^[^\x00-\x1F*?\[\]]{1,256}$')][string]$InterfaceAlias = 'WLAN',
    [ValidateRange(1, 65535)][int]$Port = 6577,
    [string]$StateDirectory
)
$ErrorActionPreference = 'Stop'
$WindowsName = 'TritonBack-LAN-6577'
$WindowsGroup = 'Triton Back LAN'
$HyperVName = 'TritonBack-Mirrored-6577'
$CreatorId = '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}'
$Owner = 'triton_back.mirrored-lan-access'
$Utf8 = New-Object System.Text.UTF8Encoding($false)
if ([string]::IsNullOrWhiteSpace($StateDirectory)) {
    $StateDirectory = Join-Path (Join-Path $PSScriptRoot '..\..') 'runtime\windows\lan-access'
}
$StateDirectory = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($StateDirectory)
$StateFile = Join-Path $StateDirectory 'state.json'

function Test-IPv4([string]$Address) {
    $Parsed = $null
    if (-not [Net.IPAddress]::TryParse($Address, [ref]$Parsed) -or
        $Parsed.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork -or
        $Parsed.ToString() -cne $Address) { return $false }
    $Bytes = $Parsed.GetAddressBytes()
    return $Bytes[0] -gt 0 -and $Bytes[0] -lt 224 -and $Bytes[0] -ne 127 -and
        -not ($Bytes[0] -eq 169 -and $Bytes[1] -eq 254)
}

function Get-Subnet([string]$Address, [int]$PrefixLength) {
    if (-not (Test-IPv4 $Address) -or $PrefixLength -lt 1 -or $PrefixLength -gt 32) {
        throw 'A canonical unicast IPv4 address and prefix length 1..32 are required.'
    }
    $Bytes = ([Net.IPAddress]::Parse($Address)).GetAddressBytes()
    for ($Index = 0; $Index -lt 4; ++$Index) {
        $Bits = [Math]::Max(0, [Math]::Min(8, $PrefixLength - 8 * $Index))
        $Mask = if ($Bits -eq 0) { 0 } else { (255 -shl (8 - $Bits)) -band 255 }
        $Bytes[$Index] = $Bytes[$Index] -band $Mask
    }
    return ([Net.IPAddress]::new($Bytes)).ToString() + '/' + $PrefixLength
}

function Normalize-Subnet([string]$Value) {
    $Parts = $Value.Split('/')
    if ($Parts.Count -eq 1) { $Parts = @($Value, '32') }
    if ($Parts.Count -ne 2) { return $null }
    if ($Parts[1] -match '^\d{1,2}$') { $Prefix = [int]$Parts[1] }
    else {
        $Mask = $null
        if (-not [Net.IPAddress]::TryParse($Parts[1], [ref]$Mask) -or
            $Mask.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork) { return $null }
        $Prefix = 0
        $ZeroSeen = $false
        foreach ($Byte in $Mask.GetAddressBytes()) {
            for ($Bit = 7; $Bit -ge 0; --$Bit) {
                if (($Byte -band (1 -shl $Bit)) -ne 0) {
                    if ($ZeroSeen) { return $null }
                    ++$Prefix
                } else { $ZeroSeen = $true }
            }
        }
    }
    try { return Get-Subnet $Parts[0] $Prefix }
    catch { return $null }
}

function Assert-Config($Config) {
    if ($null -eq $Config -or -not (Test-IPv4 ([string]$Config.lanAddress)) -or
        [int]$Config.port -lt 1 -or [int]$Config.port -gt 65535 -or
        [string]$Config.distribution -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$' -or
        [string]$Config.interfaceAlias -notmatch '^[^\x00-\x1F*?\[\]]{1,256}$' -or
        [string]$Config.subnet -cne (Get-Subnet ([string]$Config.lanAddress) ([int]$Config.prefixLength))) {
        throw 'Invalid ownership configuration; inspect the state and backups before proceeding.'
    }
}

function Get-DesiredConfig {
    $Lines = & wsl.exe -d $Distribution -u root --exec wslinfo --networking-mode
    if ($LASTEXITCODE -ne 0 -or ($Lines -join '').Trim() -ne 'mirrored') {
        throw "Distribution '$Distribution' must already use mirrored networking."
    }
    $Adapters = @(Get-NetAdapter -Name $InterfaceAlias | Where-Object { $_.Name -eq $InterfaceAlias })
    if ($Adapters.Count -ne 1 -or -not $Adapters[0].HardwareInterface -or $Adapters[0].Status -ne 'Up') {
        throw "Interface '$InterfaceAlias' must identify one active physical LAN adapter."
    }
    $Network = Get-NetIPConfiguration -InterfaceIndex $Adapters[0].InterfaceIndex
    if (-not $Network.IPv4DefaultGateway) { throw "Interface '$InterfaceAlias' has no IPv4 default gateway." }
    $Addresses = @(Get-NetIPAddress -InterfaceIndex $Adapters[0].InterfaceIndex -AddressFamily IPv4 |
        Where-Object { $_.AddressState -eq 'Preferred' -and -not $_.SkipAsSource -and (Test-IPv4 $_.IPAddress) })
    if ($Addresses.Count -ne 1) { throw "Interface '$InterfaceAlias' must have exactly one preferred usable IPv4 address." }
    return [pscustomobject][ordered]@{
        distribution = $Distribution; interfaceAlias = $Adapters[0].Name; lanAddress = $Addresses[0].IPAddress
        prefixLength = [int]$Addresses[0].PrefixLength
        subnet = Get-Subnet $Addresses[0].IPAddress $Addresses[0].PrefixLength
        port = $Port
    }
}

function Get-WindowsRule {
    $Rules = @(Get-NetFirewallRule -PolicyStore PersistentStore -Name $WindowsName -ErrorAction SilentlyContinue)
    if ($Rules.Count -eq 0) { return $null }
    if ($Rules.Count -ne 1) { throw "Multiple firewall rules use '$WindowsName'." }
    $Rule = $Rules[0]
    $Address = $Rule | Get-NetFirewallAddressFilter
    $Ports = $Rule | Get-NetFirewallPortFilter
    $Interface = $Rule | Get-NetFirewallInterfaceFilter
    return [pscustomobject][ordered]@{
        name = $Rule.Name; group = $Rule.Group; enabled = [string]$Rule.Enabled
        direction = [string]$Rule.Direction; action = [string]$Rule.Action; profiles = [string]$Rule.Profile
        protocol = [string]$Ports.Protocol; localPorts = @($Ports.LocalPort); remotePorts = @($Ports.RemotePort)
        localAddresses = @($Address.LocalAddress); remoteAddresses = @($Address.RemoteAddress)
        interfaceAlias = @($Interface.InterfaceAlias)
    }
}

function Get-HyperVRule {
    $Rules = @(Get-NetFirewallHyperVRule -PolicyStore PersistentStore -Name $HyperVName -ErrorAction SilentlyContinue)
    if ($Rules.Count -eq 0) { return $null }
    if ($Rules.Count -ne 1) { throw "Multiple Hyper-V firewall rules use '$HyperVName'." }
    $Rule = $Rules[0]
    return [pscustomobject][ordered]@{
        name = $Rule.Name; creatorId = [string]$Rule.VMCreatorId; enabled = [string]$Rule.Enabled
        direction = [string]$Rule.Direction; action = [string]$Rule.Action; profiles = [string]$Rule.Profiles
        protocol = [string]$Rule.Protocol; localPorts = @($Rule.LocalPorts); remotePorts = @($Rule.RemotePorts)
        localAddresses = @($Rule.LocalAddresses); remoteAddresses = @($Rule.RemoteAddresses)
    }
}

function Test-Rule($Rule, $Config, [string]$Kind) {
    if ($null -eq $Rule -or $null -eq $Config) { return $false }
    $Common = $Rule.enabled -eq 'True' -and $Rule.direction -eq 'Inbound' -and $Rule.action -eq 'Allow' -and
        $Rule.profiles -eq 'Any' -and $Rule.protocol -in @('TCP', '6') -and
        ($Rule.localPorts -join ',') -eq [string]$Config.port -and
        (($Rule.remotePorts -join ',') -in @('Any', '')) -and
        (Normalize-Subnet ($Rule.localAddresses -join ',')) -ceq ($Config.lanAddress + '/32') -and
        (Normalize-Subnet ($Rule.remoteAddresses -join ',')) -ceq $Config.subnet
    if ($Kind -eq 'Windows') {
        return $Common -and $Rule.name -ceq $WindowsName -and $Rule.group -ceq $WindowsGroup -and
            ($Rule.interfaceAlias -join ',') -eq $Config.interfaceAlias
    }
    return $Common -and $Rule.name -ceq $HyperVName -and [Guid]$Rule.creatorId -eq [Guid]$CreatorId
}

function Add-Rule($Config, [string]$Kind) {
    if ($Kind -eq 'Windows') {
        New-NetFirewallRule -PolicyStore PersistentStore -Name $WindowsName -DisplayName 'Triton Back LAN inference' -Group $WindowsGroup `
            -Enabled True -Direction Inbound -Action Allow -Profile Any -Protocol TCP -LocalPort $Config.port `
            -LocalAddress $Config.lanAddress -RemoteAddress $Config.subnet -InterfaceAlias $Config.interfaceAlias | Out-Null
    } else {
        New-NetFirewallHyperVRule -PolicyStore PersistentStore -Name $HyperVName -DisplayName 'Triton Back mirrored inference' `
            -VMCreatorId $CreatorId -Enabled True -Direction Inbound -Action Allow -Profiles Any -Protocol TCP `
            -LocalPorts $Config.port -LocalAddresses $Config.lanAddress -RemoteAddresses $Config.subnet | Out-Null
    }
}

function Remove-Rule([string]$Kind) {
    if ($Kind -eq 'Windows') { Remove-NetFirewallRule -PolicyStore PersistentStore -Name $WindowsName }
    else { Remove-NetFirewallHyperVRule -PolicyStore PersistentStore -Name $HyperVName }
}

function Write-State([string]$Text) {
    $Temporary = Join-Path $StateDirectory ('state.' + [Guid]::NewGuid().ToString('N') + '.tmp')
    [IO.File]::WriteAllText($Temporary, $Text, $Utf8)
    Move-Item -LiteralPath $Temporary -Destination $StateFile -Force
}

function Write-Report($Config, [bool]$Planned) {
    $Windows = Get-WindowsRule
    $HyperV = Get-HyperVRule
    [pscustomobject][ordered]@{
        action = $Action; planned = $Planned; networkingMode = 'mirrored'
        lanUrl = $(if ($null -ne $Config) { "http://$($Config.lanAddress):$($Config.port)" } else { $null })
        configuration = $Config; windowsRule = $Windows; hyperVRule = $HyperV
        windowsRuleMatches = (Test-Rule $Windows $Config 'Windows'); hyperVRuleMatches = (Test-Rule $HyperV $Config 'HyperV')
        windowsFirewallProfiles = $WindowsProfiles; hyperVFirewallProfiles = $HyperVProfiles; hyperVVMSetting = $VMSetting
        stateFile = $StateFile
        note = 'Global firewall settings are unchanged. Rules do not restrict sources while the corresponding firewall is disabled.'
    } | ConvertTo-Json -Depth 12
}

$PreviousStateText = if (Test-Path -LiteralPath $StateFile -PathType Leaf) { [IO.File]::ReadAllText($StateFile) } else { $null }
$State = if ($null -ne $PreviousStateText) { $PreviousStateText | ConvertFrom-Json } else { $null }
if ($null -ne $State -and ($State.owner -cne $Owner -or $State.schemaVersion -ne 1)) {
    throw "The state file is not owned by this script: $StateFile"
}
$OwnedConfigs = @()
if ($null -ne $State) {
    $OwnedConfigs = @(@($State.current, $State.pending) | Where-Object { $null -ne $_ })
    foreach ($Saved in $OwnedConfigs) { Assert-Config $Saved }
}
$Desired = if ($Action -ne 'Remove') { Get-DesiredConfig } elseif ($OwnedConfigs.Count -gt 0) { $OwnedConfigs[0] } else { $null }
$Rules = @{ Windows = Get-WindowsRule; HyperV = Get-HyperVRule }
$WindowsProfiles = @(Get-NetFirewallProfile | ForEach-Object {
    [pscustomobject]@{ name = [string]$_.Name; enabled = [string]$_.Enabled }
})
$HyperVProfiles = @(Get-NetFirewallHyperVProfile | Where-Object { [Guid]$_.Name -eq [Guid]$CreatorId } | ForEach-Object {
    [pscustomobject]@{ profile = [string]$_.Profile; enabled = [string]$_.Enabled }
})
$VMSetting = Get-NetFirewallHyperVVMSetting -Name $CreatorId | Select-Object Name,
    @{Name='enabled';Expression={[string]$_.Enabled}},
    @{Name='defaultInboundAction';Expression={[string]$_.DefaultInboundAction}}
if ($Action -eq 'Status') { Write-Report $Desired $false; return }
$PreviousConfigs = @{}
foreach ($Kind in @('Windows', 'HyperV')) {
    $PreviousConfigs[$Kind] = @($OwnedConfigs | Where-Object { Test-Rule $Rules[$Kind] $_ $Kind }) | Select-Object -First 1
    if ($null -ne $Rules[$Kind] -and $null -eq $PreviousConfigs[$Kind]) {
        throw "Refusing to change the unowned or externally modified $Kind firewall rule. Inspect '$StateFile'."
    }
}
$Target = if ($null -ne $Desired) { "$($Desired.lanAddress):$($Desired.port), subnet $($Desired.subnet)" } else { $StateFile }
if (-not $PSCmdlet.ShouldProcess($Target, "$Action owned Windows and Hyper-V firewall rules")) {
    Write-Report $Desired $true
    return
}
$Identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$Principal = New-Object Security.Principal.WindowsPrincipal($Identity)
if (-not $Principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Configure and Remove require an elevated PowerShell. Run as administrator; no changes were made.'
}
[IO.Directory]::CreateDirectory($StateDirectory) | Out-Null
$Timestamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmss.fffffffZ')
$BackupFile = Join-Path $StateDirectory ("backup-$Timestamp.json")
$Next = [ordered]@{ schemaVersion = 1; owner = $Owner; updatedAtUtc = $Timestamp
    current = $(if ($null -ne $State) { $State.current } else { $null })
    pending = $(if ($Action -eq 'Configure') { $Desired } elseif ($null -ne $State) { $State.pending } else { $null }) }
$Backup = [ordered]@{ owner = $Owner; createdAtUtc = $Timestamp; previousState = $State; plannedState = $Next
    windowsRule = $Rules.Windows; hyperVRule = $Rules.HyperV
    windowsFirewallProfiles = $WindowsProfiles; hyperVFirewallProfiles = $HyperVProfiles; hyperVVMSetting = $VMSetting }
[IO.File]::WriteAllText($BackupFile, ($Backup | ConvertTo-Json -Depth 16), $Utf8)
Write-State ($Next | ConvertTo-Json -Depth 12)
$Changed = @()
try {
    foreach ($Kind in @('Windows', 'HyperV')) {
        if ($Action -eq 'Configure' -and (Test-Rule $Rules[$Kind] $Desired $Kind)) { continue }
        if ($null -eq $Rules[$Kind] -and $Action -eq 'Remove') { continue }
        $Changed += $Kind
        if ($null -ne $Rules[$Kind]) { Remove-Rule $Kind }
        if ($Action -eq 'Configure') { Add-Rule $Desired $Kind }
    }
    if ($Action -eq 'Configure') {
        $Next.current = $Desired
        $Next.pending = $null
        Write-State ($Next | ConvertTo-Json -Depth 12)
    } else { Remove-Item -LiteralPath $StateFile -Force }
} catch {
    $Failure = $_
    try {
        $Managed = @($OwnedConfigs + @($Desired) | Where-Object { $null -ne $_ })
        foreach ($Kind in $Changed) {
            $Now = if ($Kind -eq 'Windows') { Get-WindowsRule } else { Get-HyperVRule }
            if ($null -ne $Now) {
                if (@($Managed | Where-Object { Test-Rule $Now $_ $Kind }).Count -eq 0) {
                    throw "$Kind rule changed externally; rollback will not overwrite it."
                }
                Remove-Rule $Kind
            }
            if ($null -ne $PreviousConfigs[$Kind]) { Add-Rule $PreviousConfigs[$Kind] $Kind }
        }
        if ($null -eq $PreviousStateText) { Remove-Item -LiteralPath $StateFile -Force }
        else { Write-State $PreviousStateText }
    } catch { throw "Rule update failed: $($Failure.Exception.Message). Rollback needs review: $($_.Exception.Message). Backup: $BackupFile" }
    throw "Rule update failed; previous owned rules restored: $($Failure.Exception.Message). Backup: $BackupFile"
}
Write-Report $Desired $false
