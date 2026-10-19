# Run: powershell.exe -NoProfile -ExecutionPolicy Bypass -File tests/scripts/test_lan_access.ps1
# pwsh accepts the same arguments.
# Only pure helper definitions are loaded; the network script's top level is never executed.
$ErrorActionPreference = 'Stop'
$Source = Join-Path (Join-Path $PSScriptRoot '..\..') 'scripts\windows\lan-access.ps1'
$Tokens = $null
$Errors = $null
$Ast = [System.Management.Automation.Language.Parser]::ParseFile($Source, [ref]$Tokens, [ref]$Errors)
if ($Errors.Count -gt 0) { throw ($Errors.Message -join '; ') }
foreach ($Name in @('Test-IPv4', 'Get-Subnet', 'Normalize-Subnet', 'Assert-Config', 'Test-Rule')) {
    $Definition = $Ast.Find({ param($Node)
        $Node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $Node.Name -eq $Name
    }, $false)
    if ($null -eq $Definition) { throw "Missing pure helper: $Name" }
    . ([ScriptBlock]::Create($Definition.Extent.Text))
}
$WindowsName = 'TritonBack-LAN-6577'
$WindowsGroup = 'Triton Back LAN'
$HyperVName = 'TritonBack-Mirrored-6577'
$CreatorId = '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}'
$Checks = 0
function Assert-True([bool]$Condition, [string]$Case) {
    if (-not $Condition) { throw "Failed: $Case" }
    ++$script:Checks
}
function Assert-Throws([ScriptBlock]$Operation, [string]$Case) {
    $Threw = $false
    try { & $Operation | Out-Null } catch { $Threw = $true }
    Assert-True $Threw $Case
}
function Copy-Object($Value) { return $Value | ConvertTo-Json -Depth 6 | ConvertFrom-Json }

foreach ($Address in @('192.168.30.200', '10.20.30.40', '172.20.0.2')) {
    Assert-True (Test-IPv4 $Address) "usable IPv4 $Address"
}
foreach ($Address in @('', '10.1', '192.168.030.200', '3232236033', '127.0.0.1',
    '169.254.1.1', '0.0.0.0', '224.0.0.1', '255.255.255.255', '::1', '192.168.30.999')) {
    Assert-True (-not (Test-IPv4 $Address)) "reject unusable/noncanonical IPv4 $Address"
}
foreach ($Case in @(
    @('192.168.30.200', 24, '192.168.30.0/24'), @('10.10.7.8', 20, '10.10.0.0/20'),
    @('192.168.30.201', 31, '192.168.30.200/31'), @('192.168.30.200', 32, '192.168.30.200/32'))) {
    Assert-True ((Get-Subnet $Case[0] $Case[1]) -ceq $Case[2]) "actual prefix subnet $($Case[0])/$($Case[1])"
}
foreach ($Value in @('192.168.30.0/24', '192.168.30.0/255.255.255.0')) {
    Assert-True ((Normalize-Subnet $Value) -ceq '192.168.30.0/24') "equivalent subnet $Value"
}
foreach ($Value in @('192.168.30.200', '192.168.30.200/32', '192.168.30.200/255.255.255.255')) {
    Assert-True ((Normalize-Subnet $Value) -ceq '192.168.30.200/32') "equivalent host $Value"
}
foreach ($Value in @('Any', '192.168.30.0/255.0.255.0', '192.168.30.0/33',
    '192.168.30.0/24/32', '192.168.30.0/24,10.0.0.0/8')) {
    Assert-True ($null -eq (Normalize-Subnet $Value)) "reject invalid/multiple subnets $Value"
}
Assert-Throws { Get-Subnet '192.168.30.200' 0 } 'reject unrestricted prefix'
Assert-Throws { Get-Subnet '192.168.30.200' 33 } 'reject oversized prefix'

$Config = [pscustomobject]@{ distribution = 'template-clone'; interfaceAlias = 'WLAN'
    lanAddress = '192.168.30.200'; prefixLength = 24; subnet = '192.168.30.0/24'; port = 6577 }
Assert-Config $Config
foreach ($Change in @(
    @{ field = 'lanAddress'; value = '127.0.0.1' }, @{ field = 'port'; value = 0 },
    @{ field = 'port'; value = 65536 }, @{ field = 'prefixLength'; value = 33 },
    @{ field = 'subnet'; value = '192.168.31.0/24' }, @{ field = 'interfaceAlias'; value = 'WLAN*' },
    @{ field = 'distribution'; value = 'template-clone;other' })) {
    $Invalid = Copy-Object $Config
    $Invalid.($Change.field) = $Change.value
    Assert-Throws { Assert-Config $Invalid } "reject invalid ownership configuration $($Change.field)=$($Change.value)"
}
$Base = [pscustomobject]@{ name = $WindowsName; group = $WindowsGroup; creatorId = $CreatorId
    enabled = 'True'; direction = 'Inbound'; action = 'Allow'; profiles = 'Any'; protocol = 'TCP'
    localPorts = @('6577'); remotePorts = @('Any'); localAddresses = @('192.168.30.200')
    remoteAddresses = @('192.168.30.0/24'); interfaceAlias = @('WLAN') }
foreach ($Kind in @('Windows', 'HyperV')) {
    $Rule = Copy-Object $Base
    if ($Kind -eq 'HyperV') { $Rule.name = $HyperVName }
    Assert-True (Test-Rule $Rule $Config $Kind) "owned $Kind rule"
    $Rule.remoteAddresses = @('192.168.30.0/255.255.255.0')
    $Rule.localAddresses = @('192.168.30.200/255.255.255.255')
    $Rule.protocol = '6'
    Assert-True (Test-Rule $Rule $Config $Kind) "normalized $Kind rule returned by Windows"
    foreach ($Change in @(
        @{ field = 'enabled'; value = 'False' }, @{ field = 'direction'; value = 'Outbound' },
        @{ field = 'action'; value = 'Block' }, @{ field = 'profiles'; value = 'Private' },
        @{ field = 'protocol'; value = 'UDP' }, @{ field = 'name'; value = 'SomeOtherRule' },
        @{ field = 'localPorts'; value = @('6578') }, @{ field = 'localPorts'; value = @('6577', '80') },
        @{ field = 'remotePorts'; value = @('443') }, @{ field = 'localAddresses'; value = @('Any') },
        @{ field = 'localAddresses'; value = @('192.168.30.201') },
        @{ field = 'localAddresses'; value = @('192.168.30.200', '10.20.30.40') },
        @{ field = 'remoteAddresses'; value = @('Any') },
        @{ field = 'remoteAddresses'; value = @('192.168.31.0/24') },
        @{ field = 'remoteAddresses'; value = @('192.168.30.0/24', '10.0.0.0/8') })) {
        $Invalid = Copy-Object $Rule
        $Invalid.($Change.field) = $Change.value
        Assert-True (-not (Test-Rule $Invalid $Config $Kind)) "reject changed $Kind rule $($Change.field)"
    }
}
$Invalid = Copy-Object $Base
$Invalid.interfaceAlias = @('Meta')
Assert-True (-not (Test-Rule $Invalid $Config 'Windows')) 'reject VPN interface substitution'
$Invalid = Copy-Object $Base
$Invalid.group = 'SomeOtherProject'
Assert-True (-not (Test-Rule $Invalid $Config 'Windows')) 'reject foreign Windows rule group'
$Invalid = Copy-Object $Base
$Invalid.name = $HyperVName
$Invalid.creatorId = '{00000000-0000-0000-0000-000000000001}'
Assert-True (-not (Test-Rule $Invalid $Config 'HyperV')) 'reject foreign VM creator'
Write-Output "LAN helper tests passed: $Checks checks; PowerShell $($PSVersionTable.PSVersion). No network or firewall commands executed."
