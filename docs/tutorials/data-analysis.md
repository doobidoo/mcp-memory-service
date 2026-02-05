# Data Analysis Examples

This guide demonstrates how to extract insights, patterns, and visualizations from your MCP Memory Service data, transforming stored knowledge into actionable intelligence.

## 🎯 Overview

The MCP Memory Service can be used not just for storage and retrieval, but as a powerful analytics platform for understanding knowledge patterns, usage trends, and information relationships. This guide shows practical examples of data analysis techniques that reveal valuable insights about your knowledge base.

## 📊 Types of Analysis

### 1. Temporal Analysis
Understanding when and how your knowledge base grows over time.

### 2. Content Analysis
Analyzing what types of information are stored and how they're organized.

### 3. Usage Pattern Analysis
Identifying how information is accessed and utilized.

### 4. Quality Analysis
Measuring the health and organization of your knowledge base.

### 5. Relationship Analysis
Discovering connections and patterns between different pieces of information.

## 📈 Temporal Distribution Analysis

### Basic Time-Based Queries

**Monthly Distribution:**
```javascript
// Retrieve memories by time period
const januaryMemories = await recall_memory({
  "query": "memories from january 2025",
  "n_results": 50
});

const juneMemories = await recall_memory({
  "query": "memories from june 2025",
  "n_results": 50
});

// Analyze patterns
console.log(`January: ${januaryMemories.length} memories`);
console.log(`June: ${juneMemories.length} memories`);
```

**Weekly Activity Patterns:**
```javascript
// Get recent activity
const lastWeek = await recall_memory({
  "query": "memories from last week",
  "n_results": 25
});

const thisWeek = await recall_memory({
  "query": "memories from this week",
  "n_results": 25
});

// Compare activity levels
const weeklyGrowth = ((thisWeek.length - lastWeek.length) / lastWeek.length) * 100;
console.log(`Weekly growth rate: ${weeklyGrowth.toFixed(1)}%`);
```

### Advanced Temporal Analysis

**Memory Creation Frequency:**
```javascript
// Process temporal data for visualization
function analyzeMemoryDistribution(memories) {
  const monthlyDistribution = {};

  memories.forEach(memory => {
    // Extract date from timestamp
    const date = new Date(memory.timestamp);
    const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;

    if (!monthlyDistribution[monthKey]) {
      monthlyDistribution[monthKey] = {
        count: 0,
        memories: []
      };
    }

    monthlyDistribution[monthKey].count++;
    monthlyDistribution[monthKey].memories.push(memory);
  });

  return monthlyDistribution;
}

// Convert to chart data
function prepareChartData(distribution) {
  return Object.entries(distribution)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([month, data]) => {
      const [year, monthNum] = month.split('-');
      const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
      const monthName = monthNames[parseInt(monthNum) - 1];

      return {
        month: `${monthName} ${year}`,
        count: data.count,
        monthKey: month,
        memories: data.memories
      };
    });
}
```

**Project Lifecycle Analysis:**
```javascript
// Analyze project phases through memory patterns
async function analyzeProjectLifecycle(projectTag) {
  const projectMemories = await search_by_tag({
    "tags": [projectTag]
  });

  // Group by status tags
  const phases = {
    planning: [],
    development: [],
    testing: [],
    deployment: [],
    maintenance: []
  };

  projectMemories.forEach(memory => {
    const tags = memory.tags || [];

    if (tags.includes('planning') || tags.includes('design')) {
      phases.planning.push(memory);
    } else if (tags.includes('development') || tags.includes('implementation')) {
      phases.development.push(memory);
    } else if (tags.includes('testing') || tags.includes('debugging')) {
      phases.testing.push(memory);
    } else if (tags.includes('deployment') || tags.includes('production')) {
      phases.deployment.push(memory);
    } else if (tags.includes('maintenance') || tags.includes('optimization')) {
      phases.maintenance.push(memory);
    }
  });

  return phases;
}

// Usage example
const mcpLifecycle = await analyzeProjectLifecycle('mcp-memory-service');
console.log('Project phases:', {
  planning: mcpLifecycle.planning.length,
  development: mcpLifecycle.development.length,
  testing: mcpLifecycle.testing.length,
  deployment: mcpLifecycle.deployment.length,
  maintenance: mcpLifecycle.maintenance.length
});
```

## 🏷️ Tag Analysis

### Tag Frequency Analysis

**Most Used Tags:**
```javascript
async function analyzeTagFrequency() {
  // Get all memories (you may need to paginate for large datasets)
  const allMemories = await retrieve_memory({
    "query": "all memories",
    "n_results": 500
  });

  const tagFrequency = {};

  allMemories.forEach(memory => {
    const tags = memory.tags || [];
    tags.forEach(tag => {
      tagFrequency[tag] = (tagFrequency[tag] || 0) + 1;
    });
  });

  // Sort by frequency
  const sortedTags = Object.entries(tagFrequency)
    .sort(([,a], [,b]) => b - a)
    .slice(0, 20); // Top 20 tags

  return sortedTags;
}

// Generate insights
const topTags = await analyzeTagFrequency();
console.log('Most used tags:');
topTags.forEach(([tag, count]) => {
  console.log(`${tag}: ${count} memories`);
});
```

**Tag Co-occurrence Analysis:**
```javascript
function analyzeTagRelationships(memories) {
  const cooccurrence = {};

  memories.forEach(memory => {
    const tags = memory.tags || [];

    // For each pair of tags in the memory
    for (let i = 0; i < tags.length; i++) {
      for (let j = i + 1; j < tags.length; j++) {
        const pair = [tags[i], tags[j]].sort().join(' + ');
        cooccurrence[pair] = (cooccurrence[pair] || 0) + 1;
      }
    }
  });

  // Find most common tag combinations
  return Object.entries(cooccurrence)
    .sort(([,a], [,b]) => b - a)
    .slice(0, 10);
}

// Usage
const tagRelationships = analyzeTagRelationships(allMemories);
console.log('Common tag combinations:');
tagRelationships.forEach(([pair, count]) => {
  console.log(`${pair}: ${count} times`);
});
```

### Tag Category Analysis

**Category Distribution:**
```javascript
function categorizeTagsByType(tags) {
  const categories = {
    projects: [],
    technologies: [],
    activities: [],
    status: [],
    content: [],
    temporal: [],
    other: []
  };

  // Define patterns for each category
  const patterns = {
    projects: /^(mcp-memory-service|memory-dashboard|github-integration)/,
    technologies: /^(python|react|typescript|chromadb|git|docker)/,
    activities: /^(testing|debugging|development|documentation|deployment)/,
    status: /^(resolved|in-progress|blocked|verified|completed)/,
    content: /^(concept|architecture|tutorial|reference|example)/,
    temporal: /^(january|february|march|april|may|june|q1|q2|2025)/
  };

  tags.forEach(([tag, count]) => {
    let categorized = false;

    for (const [category, pattern] of Object.entries(patterns)) {
      if (pattern.test(tag)) {
        categories[category].push([tag, count]);
        categorized = true;
        break;
      }
    }

    if (!categorized) {
      categories.other.push([tag, count]);
    }
  });

  return categories;
}

// Analyze tag distribution by category
const tagCategories = categorizeTagsByType(topTags);
console.log('Tags by category:');
Object.entries(tagCategories).forEach(([category, tags]) => {
  console.log(`${category}: ${tags.length} unique tags`);
});
```

## 📋 Content Quality Analysis

### Tagging Quality Assessment

**Untagged Memory Detection:**
```javascript
async function findUntaggedMemories() {
  // Search for potentially untagged content
  const candidates = await retrieve_memory({
    "query": "test simple basic example memory",
    "n_results": 50
  });

  const untagged = candidates.filter(memory => {
    const tags = memory.tags || [];
    return tags.length === 0 ||
           (tags.length === 1 && ['test', 'memory', 'note'].includes(tags[0]));
  });

  return {
    total: candidates.length,
    untagged: untagged.length,
    percentage: (untagged.length / candidates.length) * 100,
    examples: untagged.slice(0, 5)
  };
}

// Quality assessment
const qualityReport = await findUntaggedMemories();
console.log(`Tagging quality: ${(100 - qualityReport.percentage).toFixed(1)}% properly tagged`);
```

**Tag Consistency Analysis:**
```javascript
function analyzeTagConsistency(memories) {
  const patterns = {};
  const inconsistencies = [];

  memories.forEach(memory => {
    const content = memory.content;
    const tags = memory.tags || [];

    // Look for common content patterns
    if (content.includes('issue') || content.includes('bug')) {
      const hasIssueTag = tags.some(tag => tag.includes('issue') || tag.includes('bug'));
      if (!hasIssueTag) {
        inconsistencies.push({
          type: 'missing-issue-tag',
          memory: memory.content.substring(0, 100),
          tags: tags
        });
      }
    }

    if (content.includes('test') || content.includes('TEST')) {
      const hasTestTag = tags.includes('test') || tags.includes('testing');
      if (!hasTestTag) {
        inconsistencies.push({
          type: 'missing-test-tag',
          memory: memory.content.substring(0, 100),
          tags: tags
        });
      }
    }
  });

  return {
    totalMemories: memories.length,
    inconsistencies: inconsistencies.length,
    consistencyScore: ((memories.length - inconsistencies.length) / memories.length) * 100,
    examples: inconsistencies.slice(0, 5)
  };
}
```

## 📊 Visualization Examples

### Memory Distribution Chart Data

**Prepare data for visualization:**
```javascript
function prepareDistributionData(memories) {
  const distribution = analyzeMemoryDistribution(memories);
  const chartData = prepareChartData(distribution);

  // Add additional metrics
  const total = chartData.reduce((sum, item) => sum + item.count, 0);
  const average = total / chartData.length;

  // Identify peaks and valleys
  const peak = chartData.reduce((max, item) =>
    item.count > max.count ? item : max, chartData[0]);
  const valley = chartData.reduce((min, item) =>
    item.count < min.count ? item : min, chartData[0]);

  return {
    chartData,
    metrics: {
      total,
      average: Math.round(average * 10) / 10,
      peak: { month: peak.month, count: peak.count },
      valley: { month: valley.month, count: valley.count },
      growth: calculateGrowthRate(chartData)
    }
  };
}

function calculateGrowthRate(chartData) {
  if (chartData.length < 2) return 0;

  const first = chartData[0].count;
  const last = chartData[chartData.length - 1].count;

  return ((last - first) / first) * 100;
}
```

### Activity Heatmap Data

**Generate activity patterns:**
```javascript
function generateActivityHeatmap(memories) {
  const heatmapData = {};

  memories.forEach(memory => {
    const date = new Date(memory.timestamp);
    const dayOfWeek = date.getDay(); // 0 = Sunday
    const hour = date.getHours();

    const key = `${dayOfWeek}-${hour}`;
    heatmapData[key] = (heatmapData[key] || 0) + 1;
  });

  // Convert to matrix format for visualization
  const matrix = [];
  const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

  for (let day = 0; day < 7; day++) {
    const dayData = [];
    for (let hour = 0; hour < 24; hour++) {
      const key = `${day}-${hour}`;
      dayData.push({
        day: days[day],
        hour: hour,
        value: heatmapData[key] || 0
      });
    }
    matrix.push(dayData);
  }

  return matrix;
}
```

## 🔍 Advanced Analytics

### Semantic Similarity Analysis

**Find related memories:**
```javascript
async function findRelatedMemories(targetMemory, threshold = 0.7) {
  // Use semantic search to find similar content
  const related = await retrieve_memory({
    "query": targetMemory.content.substring(0, 200),
    "n_results": 20
  });

  // Filter by relevance score (if available)
  const highlyRelated = related.filter(memory =>
    memory.relevanceScore > threshold &&
    memory.content_hash !== targetMemory.content_hash
  );

  return highlyRelated;
}

// Build knowledge graph data
async function buildKnowledgeGraph(memories) {
  const nodes = [];
  const edges = [];

  for (const memory of memories.slice(0, 50)) { // Limit for performance
    nodes.push({
      id: memory.content_hash,
      label: memory.content.substring(0, 50) + '...',
      tags: memory.tags || [],
      group: memory.tags?.[0] || 'untagged'
    });

    const related = await findRelatedMemories(memory, 0.8);

    related.forEach(relatedMemory => {
      edges.push({
        from: memory.content_hash,
        to: relatedMemory.content_hash,
        weight: relatedMemory.relevanceScore || 0.5
      });
    });
  }

  return { nodes, edges };
}
```

### Trend Analysis

**Identify emerging patterns:**
```javascript
function analyzeTrends(memories, timeWindow = 30) {
  const now = new Date();
  const cutoff = new Date(now - timeWindow * 24 * 60 * 60 * 1000);

  const recentMemories = memories.filter(memory =>
    new Date(memory.timestamp) > cutoff
  );

  const historicalMemories = memories.filter(memory =>
    new Date(memory.timestamp) <= cutoff
  );

  // Analyze tag frequency changes
  const recentTags = getTagFrequency(recentMemories);
  const historicalTags = getTagFrequency(historicalMemories);

  const trends = [];

  Object.entries(recentTags).forEach(([tag, recentCount]) => {
    const historicalCount = historicalTags[tag] || 0;
    const change = recentCount - historicalCount;
    const changePercent = historicalCount > 0 ?
      (change / historicalCount) * 100 : 100;

    if (Math.abs(changePercent) > 50) { // Significant change
      trends.push({
        tag,
        trend: changePercent > 0 ? 'increasing' : 'decreasing',
        change: changePercent,
        recentCount,
        historicalCount
      });
    }
  });

  return trends.sort((a, b) => Math.abs(b.change) - Math.abs(a.change));
}

function getTagFrequency(memories) {
  const frequency = {};
  memories.forEach(memory => {
    (memory.tags || []).forEach(tag => {
      frequency[tag] = (frequency[tag] || 0) + 1;
    });
  });
  return frequency;
}
```

## 📋 Analysis Workflows

### Daily Analytics Routine

```javascript
async function runDailyAnalytics() {
  console.log('🔍 Daily Memory Analytics Report');
  console.log('================================');

  // 1. Recent activity
  const todayMemories = await recall_memory({
    "query": "memories from today",
    "n_results": 50
  });
  console.log(`📊 Memories added today: ${todayMemories.length}`);

  // 2. Tag quality check
  const qualityReport = await findUntaggedMemories();
  console.log(`🏷️  Tagging quality: ${(100 - qualityReport.percentage).toFixed(1)}%`);

  // 3. Most active projects
  const topTags = await analyzeTagFrequency();
  const topProjects = topTags.filter(([tag]) =>
    tag.includes('project') || tag.includes('service')
  ).slice(0, 3);
  console.log('🚀 Most active projects:', topProjects);

  // 4. Database health
  const health = await check_database_health();
  console.log(`💾 Database health: ${health.status}`);

  console.log('\n✅ Daily analytics complete');
}
```

### Weekly Analysis Report

```javascript
async function generateWeeklyReport() {
  const weekMemories = await recall_memory({
    "query": "memories from last week",
    "n_results": 100
  });

  const report = {
    summary: {
      totalMemories: weekMemories.length,
      date: new Date().toISOString().split('T')[0]
    },

    topCategories: analyzeTagFrequency(weekMemories),

    qualityMetrics: await findUntaggedMemories(),

    trends: analyzeTrends(weekMemories, 7),

    recommendations: generateRecommendations(weekMemories)
  };

  // Store report as memory
  await store_memory({
    "content": `Weekly Analytics Report - ${report.summary.date}: ${JSON.stringify(report, null, 2)}`,
    "metadata": {
      "tags": ["analytics", "weekly-report", "metrics", "summary"],
      "type": "analytics-report"
    }
  });

  return report;
}

function generateRecommendations(memories) {
  const recommendations = [];

  // Tag consistency recommendations
  const untagged = memories.filter(m => (m.tags || []).length === 0);
  if (untagged.length > 0) {
    recommendations.push({
      type: 'tagging',
      priority: 'high',
      message: `${untagged.length} memories need tagging`
    });
  }

  // Content organization recommendations
  const testMemories = memories.filter(m =>
    m.content.toLowerCase().includes('test') &&
    !(m.tags || []).includes('test')
  );
  if (testMemories.length > 0) {
    recommendations.push({
      type: 'organization',
      priority: 'medium',
      message: `${testMemories.length} test memories need proper categorization`
    });
  }

  return recommendations;
}
```

## 🎯 Practical Implementation

### Setting Up Analytics Pipeline

**1. Create analysis script:**
```javascript
// analytics.js
const MemoryAnalytics = {
  async runFullAnalysis() {
    const results = {
      temporal: await this.analyzeTemporalDistribution(),
      tags: await this.analyzeTagUsage(),
      quality: await this.assessQuality(),
      trends: await this.identifyTrends()
    };

    return results;
  },

  async generateVisualizationData() {
    const memories = await this.getAllMemories();
    return prepareDistributionData(memories);
  }
};
```

**2. Schedule regular analysis:**
```javascript
// Run analytics and store results
async function scheduledAnalysis() {
  const results = await MemoryAnalytics.runFullAnalysis();

  await store_memory({
    "content": `Automated Analytics Report: ${JSON.stringify(results, null, 2)}`,
    "metadata": {
      "tags": ["automated-analytics", "system-analysis", "metrics"],
      "type": "analytics-report"
    }
  });
}

// Run weekly
setInterval(scheduledAnalysis, 7 * 24 * 60 * 60 * 1000);
```

## 📊 Export and Integration

### Data Export for External Tools

**CSV Export:**
```javascript
function exportToCSV(memories) {
  const headers = ['Timestamp', 'Content_Preview', 'Tags', 'Type'];
  const rows = memories.map(memory => [
    memory.timestamp,
    memory.content.substring(0, 100).replace(/,/g, ';'),
    (memory.tags || []).join(';'),
    memory.type || 'unknown'
  ]);

  const csv = [headers, ...rows]
    .map(row => row.map(field => `"${field}"`).join(','))
    .join('\n');

  return csv;
}
```

**JSON Export for Visualization Tools:**
```javascript
function exportForVisualization(memories) {
  return {
    metadata: {
      total: memories.length,
      exported: new Date().toISOString(),
      schema_version: '1.0'
    },

    temporal_data: prepareDistributionData(memories),

    tag_analysis: analyzeTagFrequency(memories),

    relationships: buildKnowledgeGraph(memories),

    quality_metrics: assessQuality(memories)
  };
}
```

---

*These analysis examples demonstrate the power of treating your MCP Memory Service as not just storage, but as a comprehensive analytics platform for understanding and optimizing your knowledge management workflows.*
