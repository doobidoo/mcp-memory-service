#Requires -Version 5.1
<#
.SYNOPSIS
    Shared configuration helper for MCP Memory Service Windows management scripts.

.DESCRIPTION
    Parses the project's .env file to derive the current server URL, so that
    scripts don't need to hardcode host/port/protocol. Also provides a helper
    to bypass self-signed certificate validation for HTTPS health checks.

    Dot-source this file from other scripts:
        . "$PSScriptRoot\lib\server-config.ps1"

.NOTES
    Relevant environment variables (read from .env):
        MCP_HTTP_HOST       (default: 127.0.0.1)
        MCP_HTTP_PORT       (default: 8000)
        MCP_HTTPS_ENABLED   (default: false)

    Resolution of ProjectRoot: assumes this file lives under
        <ProjectRoot>\scripts\service\windows\lib\server-config.ps1
#>

function Get-McpProjectRoot {
    <#
    .SYNOPSIS
        Resolves the project root based on this file's location.
    #>
    return (Get-Item "$PSScriptRoot\..\..\..\..").FullName
}

function Get-McpServerConfig {
    <#
    .SYNOPSIS
        Parses .env and returns a hashtable with Host, Port, HttpsEnabled,
        BaseUrl and HealthUrl. Falls back to defaults if .env is missing or
        a value cannot be parsed.
    #>
    param(
        [string]$ProjectRoot = (Get-McpProjectRoot)
    )

    $config = @{
        Host         = '127.0.0.1'
        Port         = 8000
        HttpsEnabled = $false
    }

    $EnvFile = Join-Path $ProjectRoot ".env"
    if (Test-Path $EnvFile) {
        Get-Content $EnvFile | ForEach-Object {
            $line = $_
            if ($line -match '^\s*#' -or $line -notmatch '=') { return }
            if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^#\r\n]*?)\s*(?:#.*)?$') {
                $key = $matches[1]
                $value = $matches[2].Trim().Trim('"').Trim("'")
                switch ($key) {
                    'MCP_HTTP_HOST'     { $config.Host = $value }
                    'MCP_HTTP_PORT'     {
                        $parsed = 0
                        if ([int]::TryParse($value, [ref]$parsed)) {
                            $config.Port = $parsed
                        }
                    }
                    'MCP_HTTPS_ENABLED' { $config.HttpsEnabled = ($value.ToLower() -eq 'true') }
                }
            }
        }
    }

    $scheme = if ($config.HttpsEnabled) { 'https' } else { 'http' }
    $displayHost = if ($config.Host -eq '0.0.0.0') { 'localhost' } else { $config.Host }
    $config.Scheme = $scheme
    $config.DisplayHost = $displayHost
    $config.BaseUrl = "${scheme}://${displayHost}:$($config.Port)"
    $config.HealthUrl = "$($config.BaseUrl)/api/health"
    $config.DashboardUrl = "$($config.BaseUrl)/"

    return $config
}

function Get-McpApiKey {
    <#
    .SYNOPSIS
        Reads MCP_API_KEY from the project's .env file.

    .DESCRIPTION
        Parses .env line-by-line and extracts MCP_API_KEY. Returns $null if
        the key is not defined. Used by scripts that need to call
        authenticated endpoints such as /api/server/status or
        /api/health/detailed (required since the v10.21.0 security hardening
        in GHSA-73hc-m4hx-79pj removed version/uptime from the public
        /api/health response).

    .OUTPUTS
        System.String - the API key, or $null if not found.
    #>
    param(
        [string]$ProjectRoot = (Get-McpProjectRoot)
    )

    $EnvFile = Join-Path $ProjectRoot ".env"
    if (-not (Test-Path $EnvFile)) {
        return $null
    }

    $apiKey = $null
    foreach ($line in Get-Content $EnvFile -Encoding UTF8) {
        if ($line -match '^\s*#' -or $line -notmatch '=') { continue }
        # Regex handles quoted values (which may contain '#') and unquoted values
        # (stripped of trailing comments). Capture groups: 1=double-quoted,
        # 2=single-quoted, 3=unquoted (no '#' or whitespace allowed).
        if ($line -match '^\s*MCP_API_KEY\s*=\s*(?:"([^"]*)"|''([^'']*)''|([^#\s]*))') {
            $apiKey = ($matches[1], $matches[2], $matches[3] | Where-Object { $_ -ne $null })[0]
            break
        }
    }

    if ([string]::IsNullOrWhiteSpace($apiKey)) {
        return $null
    }
    return $apiKey
}

function Enable-McpSelfSignedCertBypass {
    <#
    .SYNOPSIS
        Bypasses self-signed certificate validation for the current PowerShell
        session, so Invoke-WebRequest can talk to the HTTPS health endpoint.
        No-op on HTTP-only setups. Safe to call multiple times.
    #>
    if (-not ('TrustAllCertsPolicy' -as [type])) {
        Add-Type -TypeDefinition @"
using System.Net;
using System.Security.Cryptography.X509Certificates;
public class TrustAllCertsPolicy : ICertificatePolicy {
    public bool CheckValidationResult(ServicePoint sp, X509Certificate cert, WebRequest req, int problem) { return true; }
}
"@ -ErrorAction SilentlyContinue
    }
    [System.Net.ServicePointManager]::CertificatePolicy = New-Object TrustAllCertsPolicy
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor [System.Net.SecurityProtocolType]::Tls12
}
