/**
 * Git Context Analyzer
 * Analyzes git repository history and changelog to provide development context for memory retrieval
 */

const fs = require('fs').promises;
const path = require('path');
const { execSync } = require('child_process');

/**
 * Get recent commit history with detailed information
 */
async function getRecentCommits(workingDir, options = {}) {
    try {
        const {
            days = 14,
            maxCommits = 20,
            includeMerges = false
        } = options;
        
        // Build git log command
        let gitCommand = `git log --pretty=format:"%H|%aI|%s|%an" --max-count=${maxCommits}`;
        
        if (!includeMerges) {
            gitCommand += ' --no-merges';
        }
        
        // Add time filter
        const sinceDate = new Date();
        sinceDate.setDate(sinceDate.getDate() - days);
        gitCommand += ` --since="${sinceDate.toISOString()}"`;
        
        const output = execSync(gitCommand, { 
            cwd: path.resolve(workingDir), 
            encoding: 'utf8',
            timeout: 10000
        });
        
        if (!output.trim()) {
            return [];
        }
        
        const commits = output.trim().split('\n').map(line => {
            const [hash, date, message, author] = line.split('|');
            return {
                hash: hash?.substring(0, 8),
                fullHash: hash,
                date: new Date(date),
                message: message || '',
                author: author || '',
                daysSinceCommit: Math.floor((new Date() - new Date(date)) / (1000 * 60 * 60 * 24))
            };
        });
        
        // Get file changes for recent commits (last 5 commits for performance)
        const recentCommits = commits.slice(0, Math.min(5, commits.length));
        for (const commit of recentCommits) {
            try {
                const filesOutput = execSync(`git show --name-only --pretty="" ${commit.fullHash}`, {
                    cwd: path.resolve(workingDir),
                    encoding: 'utf8',
                    timeout: 5000
                });
                commit.files = filesOutput.trim().split('\n').filter(f => f.length > 0);
            } catch (error) {
                commit.files = [];
            }
        }
        
        return commits;
        
    } catch (error) {
        // Silently fail for non-git directories
        return [];
    }
}

/**
 * Parse CHANGELOG.md for recent entries
 */
async function parseChangelog(workingDir) {
    try {
        const changelogPath = path.join(workingDir, 'CHANGELOG.md');
        
        try {
            await fs.access(changelogPath);
        } catch {
            // Try alternative locations
            const altPaths = ['changelog.md', 'HISTORY.md', 'RELEASES.md'];
            let found = false;
            for (const altPath of altPaths) {
                try {
                    await fs.access(path.join(workingDir, altPath));
                    changelogPath = path.join(workingDir, altPath);
                    found = true;
                    break;
                } catch (error) {
                    // File doesn't exist or access error, continue to next path
                    console.debug(`[Git Analyzer] Could not access ${altPath}: ${error.message}`);
                }
            }
            if (!found) return null;
        }
        
        const content = await fs.readFile(changelogPath, 'utf8');
        
        // Parse changelog entries (assuming standard markdown format)
        const entries = [];
        const lines = content.split('\n');
        let currentVersion = null;
        let currentDate = null;
        let currentChanges = [];
        
        for (const line of lines) {
            // Match version headers: ## [1.0.0] - 2024-08-25 or ## v1.0.0
            const versionMatch = line.match(/^##\s*\[?v?([^\]]+)\]?\s*-?\s*(.*)$/);
            if (versionMatch) {
                // Save previous entry
                if (currentVersion && currentChanges.length > 0) {
                    entries.push({
                        version: currentVersion,
                        date: currentDate,
                        changes: currentChanges.slice(),
                        raw: currentChanges.join('\n')
                    });
                }
                
                currentVersion = versionMatch[1];
                currentDate = versionMatch[2] || null;
                currentChanges = [];
                continue;
            }
            
            // Collect changes under current version
            if (currentVersion && line.trim()) {
                // Skip section headers like "### Added", "### Fixed"
                if (!line.match(/^###\s/)) {
                    currentChanges.push(line.trim());
                }
            }
        }
        
        // Don't forget the last entry
        if (currentVersion && currentChanges.length > 0) {
            entries.push({
                version: currentVersion,
                date: currentDate,
                changes: currentChanges.slice(),
                raw: currentChanges.join('\n')
            });
        }
        
        // Return only recent entries (last 3 versions or entries from last 30 days)
        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - 30);
        
        const recentEntries = entries.slice(0, 3).filter(entry => {
            if (!entry.date) return true; // Include entries without dates
            try {
                const entryDate = new Date(entry.date);
                return entryDate >= cutoffDate;
            } catch {
                return true; // Include entries with unparseable dates
            }
        });
        
        return recentEntries.length > 0 ? recentEntries : null;
        
    } catch (error) {
        // Silently fail if changelog not found or not readable
        return null;
    }
}

/**
 * Extract development keywords from git history and changelog
 */
function extractDevelopmentKeywords(commits = [], changelogEntries = null) {
    const keywords = new Set();
    const themes = new Set();
    const filePatterns = new Set();
    
    // Extract from commit messages
    commits.forEach(commit => {
        const message = commit.message.toLowerCase();
        
        // Extract action keywords (feat, fix, refactor, etc.)
        const actionMatch = message.match(/^(feat|fix|refactor|docs|test|chore|improve|add|update|enhance)([:(]|\s)/);
        if (actionMatch) {
            keywords.add(actionMatch[1]);
        }
        
        // Extract key technical terms (avoid very common words)
        const technicalTerms = message.match(/\b(hook|memory|context|retrieval|phase|query|storage|backend|session|git|recent|scoring|config)\b/g);
        if (technicalTerms) {
            technicalTerms.forEach(term => keywords.add(term));
        }
        
        // Extract file-based themes
        if (commit.files) {
            commit.files.forEach(file => {
                const basename = path.basename(file, path.extname(file));
                if (basename.length > 2) {
                    filePatterns.add(basename);
                }
                
                // Extract directory themes
                const dir = path.dirname(file);
                if (dir !== '.' && dir !== '/' && !dir.startsWith('.')) {
                    themes.add(dir.split('/')[0]); // First directory level
                }
            });
        }
    });
    
    // Extract from changelog entries
    if (changelogEntries) {
        changelogEntries.forEach(entry => {
            const text = entry.raw.toLowerCase();
            
            // Extract technical keywords
            const changelogTerms = text.match(/\b(added|fixed|improved|enhanced|updated|removed|deprecated|breaking|feature|bug|performance|security)\b/g);
            if (changelogTerms) {
                changelogTerms.forEach(term => keywords.add(term));
            }
            
            // Extract version-specific themes
            if (entry.version) {
                themes.add(`v${entry.version}`);
                themes.add(`version-${entry.version}`);
            }
        });
    }
    
    return {
        keywords: Array.from(keywords).slice(0, 15), // Limit to prevent query bloat
        themes: Array.from(themes).slice(0, 10),
        filePatterns: Array.from(filePatterns).slice(0, 10),
        recentCommitMessages: commits.slice(0, 5).map(c => c.message)
    };
}

/**
 * Build git-aware search queries
 */
function buildGitContextQuery(projectContext, gitContext, userMessage = '') {
    try {
        const queries = [];
        const baseProject = projectContext.name || 'project';
        
        // Query 1: Recent development focus
        if (gitContext.keywords.length > 0) {
            const devKeywords = gitContext.keywords.slice(0, 8).join(' ');
            const recentQuery = `${baseProject} recent development ${devKeywords}`;
            queries.push({
                type: 'recent-development',
                semanticQuery: userMessage ? `${recentQuery} ${userMessage}` : recentQuery,
                weight: 1.0,
                source: 'git-commits'
            });
        }
        
        // Query 2: File-based context
        if (gitContext.filePatterns.length > 0) {
            const fileContext = gitContext.filePatterns.slice(0, 5).join(' ');
            const fileQuery = `${baseProject} ${fileContext} implementation changes`;
            queries.push({
                type: 'file-context',
                semanticQuery: userMessage ? `${fileQuery} ${userMessage}` : fileQuery,
                weight: 0.8,
                source: 'git-files'
            });
        }
        
        // Query 3: Version/theme context
        if (gitContext.themes.length > 0) {
            const themeContext = gitContext.themes.slice(0, 5).join(' ');
            const themeQuery = `${baseProject} ${themeContext} features decisions`;
            queries.push({
                type: 'theme-context', 
                semanticQuery: userMessage ? `${themeQuery} ${userMessage}` : themeQuery,
                weight: 0.6,
                source: 'git-themes'
            });
        }
        
        // Query 4: Commit message context (most recent)
        if (gitContext.recentCommitMessages.length > 0) {
            const recentMessage = gitContext.recentCommitMessages[0];
            const commitQuery = `${baseProject} ${recentMessage}`;
            queries.push({
                type: 'commit-context',
                semanticQuery: userMessage ? `${commitQuery} ${userMessage}` : commitQuery,
                weight: 0.9,
                source: 'recent-commit'
            });
        }
        
        return queries;
        
    } catch (error) {
        // Return empty queries on error
        return [];
    }
}

/**
 * Get current git branch information
 */
function getCurrentGitInfo(workingDir) {
    try {
        const branch = execSync('git rev-parse --abbrev-ref HEAD', {
            cwd: path.resolve(workingDir),
            encoding: 'utf8',
            timeout: 3000
        }).trim();
        
        const lastCommit = execSync('git log -1 --pretty=format:"%h %s"', {
            cwd: path.resolve(workingDir),
            encoding: 'utf8',
            timeout: 3000
        }).trim();
        
        const hasChanges = execSync('git status --porcelain', {
            cwd: path.resolve(workingDir),
            encoding: 'utf8',
            timeout: 3000
        }).trim().length > 0;
        
        return {
            branch,
            lastCommit,
            hasUncommittedChanges: hasChanges,
            isGitRepo: true
        };
        
    } catch (error) {
        return {
            branch: null,
            lastCommit: null,
            hasUncommittedChanges: false,
            isGitRepo: false
        };
    }
}

/**
 * Main function to analyze git context for memory retrieval
 */
async function analyzeGitContext(workingDir, options = {}) {
    try {
        const {
            commitLookback = 14,
            maxCommits = 20,
            includeChangelog = true,
            verbose = false
        } = options;
        
        // Get basic git info
        const gitInfo = getCurrentGitInfo(workingDir);
        if (!gitInfo.isGitRepo) {
            return null;
        }
        
        // Get recent commits
        const commits = await getRecentCommits(workingDir, {
            days: commitLookback,
            maxCommits
        });
        
        // Parse changelog if enabled
        const changelogEntries = includeChangelog ? await parseChangelog(workingDir) : null;
        
        // Extract development context
        const developmentKeywords = extractDevelopmentKeywords(commits, changelogEntries);
        
        const context = {
            gitInfo,
            commits: commits.slice(0, 10), // Limit for performance
            changelogEntries,
            developmentKeywords,
            analysisTimestamp: new Date().toISOString(),
            repositoryActivity: {
                recentCommitCount: commits.length,
                activeDays: Math.max(1, Math.min(commitLookback, commits.length > 0 ? commits[0].daysSinceCommit : commitLookback)),
                hasChangelog: changelogEntries !== null,
                developmentIntensity: commits.length > 5 ? 'high' : commits.length > 2 ? 'medium' : 'low'
            }
        };
        
        if (verbose) {
            console.log(`[Git Analyzer] Analyzed ${commits.length} commits, ${changelogEntries?.length || 0} changelog entries`);
            console.log(`[Git Analyzer] Keywords: ${developmentKeywords.keywords.join(', ')}`);
        }
        
        return context;
        
    } catch (error) {
        if (options.verbose) {
            console.warn(`[Git Analyzer] Error analyzing context: ${error.message}`);
        }
        return null;
    }
}

module.exports = {
    analyzeGitContext,
    getRecentCommits,
    parseChangelog,
    extractDevelopmentKeywords,
    buildGitContextQuery,
    getCurrentGitInfo
};

// Direct execution support for testing
if (require.main === module) {
    // Test the git analyzer
    analyzeGitContext(process.cwd(), { verbose: true })
        .then(context => {
            if (context) {
                console.log('\n=== GIT CONTEXT ANALYSIS ===');
                console.log(`Repository: ${context.gitInfo.branch} (${context.commits.length} recent commits)`);
                console.log(`Development keywords: ${context.developmentKeywords.keywords.join(', ')}`);
                console.log(`File patterns: ${context.developmentKeywords.filePatterns.join(', ')}`);
                console.log(`Themes: ${context.developmentKeywords.themes.join(', ')}`);
                
                if (context.changelogEntries) {
                    console.log(`Changelog entries: ${context.changelogEntries.length}`);
                    context.changelogEntries.forEach(entry => {
                        console.log(`  - ${entry.version} (${entry.changes.length} changes)`);
                    });
                }
                
                // Test query building
                const queries = buildGitContextQuery({ name: 'test-project' }, context.developmentKeywords);
                console.log(`\nGenerated ${queries.length} git-aware queries:`);
                queries.forEach((query, idx) => {
                    console.log(`  ${idx + 1}. [${query.type}] ${query.semanticQuery}`);
                });
                
            } else {
                console.log('No git context available');
            }
        })
        .catch(error => console.error('Git analysis failed:', error));
}