/**
 * Extract real project data for MCP Memory Service video
 *
 * This script extracts metrics, code snippets, and other data from the
 * main project directory and generates a TypeScript file with the data.
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

interface ProjectData {
  version: string;
  testCount: number;
  fileCount: number;
  locCount: number;
  codeSnippets: Record<string, string>;
  gitStats: {
    commits: number;
    contributors: number;
    lastRelease: string;
  };
  performance: {
    cacheBoost: string;
    avgReadTime: string;
    tokenReduction: string;
  };
}

async function extractProjectData(): Promise<ProjectData> {
  // Navigate to project root (two levels up from video directory)
  const rootDir = path.resolve(__dirname, '../..');

  console.log('📂 Extracting data from:', rootDir);

  // Extract version from pyproject.toml
  const pyprojectPath = path.join(rootDir, 'pyproject.toml');
  const pyproject = fs.readFileSync(pyprojectPath, 'utf-8');
  const versionMatch = pyproject.match(/version = "(.+)"/);
  const version = versionMatch?.[1] || 'unknown';
  console.log('✅ Version:', version);

  // Count tests (using grep instead of pytest for speed)
  let testCount = 0;
  try {
    const testOutput = execSync(
      'grep -r "def test_" tests/ | wc -l',
      { cwd: rootDir, encoding: 'utf-8' }
    );
    testCount = parseInt(testOutput.trim());
    console.log('✅ Test count:', testCount);
  } catch (error) {
    console.warn('⚠️  Could not count tests, using default');
    testCount = 968; // Fallback from docs
  }

  // Count Python files
  const fileOutput = execSync(
    'find src -name "*.py" | wc -l',
    { cwd: rootDir, encoding: 'utf-8' }
  );
  const fileCount = parseInt(fileOutput.trim());
  console.log('✅ File count:', fileCount);

  // Count lines of code
  const locOutput = execSync(
    'find src -name "*.py" -exec cat {} \\; | wc -l',
    { cwd: rootDir, encoding: 'utf-8' }
  );
  const locCount = parseInt(locOutput.trim());
  console.log('✅ Lines of code:', locCount);

  // Extract code snippets
  console.log('📝 Extracting code snippets...');

  const codeSnippets: Record<string, string> = {};

  // BaseStorage interface (lines 10-25)
  const baseStoragePath = path.join(
    rootDir,
    'src/mcp_memory_service/storage/base.py'
  );
  if (fs.existsSync(baseStoragePath)) {
    const baseStorage = fs.readFileSync(baseStoragePath, 'utf-8').split('\n');
    codeSnippets.baseStorage = baseStorage.slice(10, 25).join('\n');
  }

  // Factory pattern
  const factoryPath = path.join(
    rootDir,
    'src/mcp_memory_service/storage/factory.py'
  );
  if (fs.existsSync(factoryPath)) {
    const factory = fs.readFileSync(factoryPath, 'utf-8').split('\n');
    codeSnippets.factoryPattern = factory.slice(15, 30).join('\n');
  }

  // Claude Desktop config from README
  const readmePath = path.join(rootDir, 'README.md');
  if (fs.existsSync(readmePath)) {
    const readme = fs.readFileSync(readmePath, 'utf-8');
    const configMatch = readme.match(/```json\n([\s\S]+?)\n```/);
    if (configMatch) {
      codeSnippets.claudeConfig = configMatch[1];
    }
  }

  console.log('✅ Code snippets extracted:', Object.keys(codeSnippets).length);

  // Git stats
  console.log('📊 Extracting git stats...');

  const commits = execSync('git rev-list --count HEAD', {
    cwd: rootDir,
    encoding: 'utf-8',
  }).trim();

  const lastRelease = execSync('git describe --tags --abbrev=0', {
    cwd: rootDir,
    encoding: 'utf-8',
  }).trim();

  console.log('✅ Git stats:', commits, 'commits, last release:', lastRelease);

  // Performance metrics from docs
  const performance = {
    cacheBoost: '534,628x',
    avgReadTime: '5ms',
    tokenReduction: '90%',
  };

  return {
    version,
    testCount,
    fileCount,
    locCount,
    codeSnippets,
    gitStats: {
      commits: parseInt(commits),
      contributors: 3, // Could extract from git log --format='%aN'
      lastRelease,
    },
    performance,
  };
}

// Run extraction and save to TypeScript file
extractProjectData()
  .then((data) => {
    const outputPath = path.join(__dirname, '../src/data/realData.ts');
    const content = `/**
 * Real project data extracted from MCP Memory Service repository
 * Generated on: ${new Date().toISOString()}
 * DO NOT EDIT MANUALLY - Run npm run extract-data to regenerate
 */

export const projectData = ${JSON.stringify(data, null, 2)} as const;
`;

    fs.writeFileSync(outputPath, content);
    console.log('✅ Project data written to:', outputPath);
    console.log('🎉 Data extraction complete!');
  })
  .catch((error) => {
    console.error('❌ Error extracting project data:', error);
    process.exit(1);
  });
