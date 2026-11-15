/**
 * Version Checker Utility
 * Reads local version from __init__.py and checks PyPI for latest published version
 */

const fs = require('fs').promises;
const path = require('path');
const https = require('https');

/**
 * Read version from _version.py, falling back to __init__.py
 * @param {string} projectRoot - Path to project root directory
 * @returns {Promise<string|null>} Version string or null if not found
 */
async function readLocalVersion(projectRoot) {
    // _version.py holds the release version. __init__.py only imports it and keeps a
    // stale literal for the case where neither _version.py nor package metadata exist.
    const pkgDir = path.join(projectRoot, 'src', 'mcp_memory_service');
    for (const file of ['_version.py', '__init__.py']) {
        try {
            const content = await fs.readFile(path.join(pkgDir, file), 'utf8');

            // Match __version__ = "X.Y.Z" or __version__ = 'X.Y.Z'
            const versionMatch = content.match(/__version__\s*=\s*['"]([\d.]+)['"]/);

            if (versionMatch && versionMatch[1]) {
                return versionMatch[1];
            }
        } catch (error) {
            // Missing file: try the next one.
        }
    }
    return null;
}

/**
 * Fetch latest version from PyPI
 * @param {string} packageName - Name of the package on PyPI
 * @param {number} timeout - Request timeout in ms (default: 2000)
 * @returns {Promise<string|null>} Latest version string or null if error
 */
async function fetchPyPIVersion(packageName = 'mcp-memory-service', timeout = 2000) {
    return new Promise((resolve) => {
        const url = `https://pypi.org/pypi/${packageName}/json`;

        const timeoutId = setTimeout(() => {
            resolve(null);
        }, timeout);

        https.get(url, {
            headers: {
                'User-Agent': 'mcp-memory-service-hook'
            }
        }, (res) => {
            let data = '';

            res.on('data', (chunk) => {
                data += chunk;
            });

            res.on('end', () => {
                clearTimeout(timeoutId);
                try {
                    const parsed = JSON.parse(data);
                    const latestVersion = parsed?.info?.version;
                    resolve(latestVersion || null);
                } catch (error) {
                    resolve(null);
                }
            });
        }).on('error', () => {
            clearTimeout(timeoutId);
            resolve(null);
        });
    });
}

/**
 * Compare two semantic versions
 * @param {string} local - Local version (e.g., "8.39.1")
 * @param {string} pypi - PyPI version (e.g., "8.38.0")
 * @returns {number} -1 if local < pypi, 0 if equal, 1 if local > pypi
 */
function compareVersions(local, pypi) {
    const localParts = local.split('.').map(Number);
    const pypiParts = pypi.split('.').map(Number);

    for (let i = 0; i < Math.max(localParts.length, pypiParts.length); i++) {
        const localPart = localParts[i] || 0;
        const pypiPart = pypiParts[i] || 0;

        if (localPart < pypiPart) return -1;
        if (localPart > pypiPart) return 1;
    }

    return 0;
}

/**
 * Get version information with local and PyPI comparison
 * @param {string} projectRoot - Path to project root directory
 * @param {Object} options - Options for version check
 * @param {boolean} options.checkPyPI - Whether to check PyPI (default: true)
 * @param {number} options.timeout - PyPI request timeout in ms (default: 2000)
 * @returns {Promise<Object>} Version info object
 */
async function getVersionInfo(projectRoot, options = {}) {
    const { checkPyPI = true, timeout = 2000 } = options;

    const localVersion = await readLocalVersion(projectRoot);

    const result = {
        local: localVersion,
        pypi: null,
        comparison: null,
        status: 'unknown'
    };

    if (!localVersion) {
        result.status = 'error';
        return result;
    }

    if (checkPyPI) {
        const pypiVersion = await fetchPyPIVersion('mcp-memory-service', timeout);
        result.pypi = pypiVersion;

        if (pypiVersion) {
            const comparison = compareVersions(localVersion, pypiVersion);
            result.comparison = comparison;

            if (comparison === 0) {
                result.status = 'published';
            } else if (comparison > 0) {
                result.status = 'development';
            } else {
                result.status = 'outdated';
            }
        } else {
            result.status = 'local-only';
        }
    } else {
        result.status = 'local-only';
    }

    return result;
}

/**
 * Format version information for display
 * @param {Object} versionInfo - Version info from getVersionInfo()
 * @param {Object} colors - Console color codes
 * @returns {string} Formatted version string
 */
function formatVersionDisplay(versionInfo, colors) {
    const { local, pypi, status } = versionInfo;

    if (!local) {
        return `${colors.CYAN}📦 Version${colors.RESET} ${colors.DIM}→${colors.RESET} ${colors.GRAY}Unable to read version${colors.RESET}`;
    }

    let statusLabel = '';
    let pypiDisplay = '';

    switch (status) {
        case 'published':
            statusLabel = `${colors.GRAY}(published)${colors.RESET}`;
            break;
        case 'development':
            statusLabel = `${colors.GRAY}(local)${colors.RESET}`;
            pypiDisplay = pypi ? ` ${colors.DIM}•${colors.RESET} PyPI: ${colors.YELLOW}${pypi}${colors.RESET}` : '';
            break;
        case 'outdated':
            statusLabel = `${colors.RED}(outdated)${colors.RESET}`;
            pypiDisplay = pypi ? ` ${colors.DIM}•${colors.RESET} PyPI: ${colors.GREEN}${pypi}${colors.RESET}` : '';
            break;
        case 'local-only':
            statusLabel = `${colors.GRAY}(local)${colors.RESET}`;
            break;
        default:
            statusLabel = `${colors.GRAY}(unknown)${colors.RESET}`;
    }

    return `${colors.CYAN}📦 Version${colors.RESET} ${colors.DIM}→${colors.RESET} ${colors.BRIGHT}${local}${colors.RESET} ${statusLabel}${pypiDisplay}`;
}

module.exports = {
    readLocalVersion,
    fetchPyPIVersion,
    compareVersions,
    getVersionInfo,
    formatVersionDisplay
};
