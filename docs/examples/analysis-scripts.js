/**
 * Memory Analysis Scripts
 *
 * A collection of JavaScript functions for analyzing and extracting insights
 * from MCP Memory Service data. These scripts demonstrate practical approaches
 * to memory data analysis, pattern recognition, and visualization preparation.
 *
 * Usage: Import individual functions or use as reference for building
 * custom analysis pipelines.
 */

// =============================================================================
// TEMPORAL ANALYSIS FUNCTIONS
// =============================================================================

/**
 * Analyze memory distribution over time periods
 * @param {Array} memories - Array of memory objects with timestamps
 * @returns {Object} Distribution data organized by time periods
 */
function analyzeTemporalDistribution(memories) {
  const distribution = {
    monthly: {},
    weekly: {},
    daily: {},
    hourly: {}
  };

  memories.forEach(memory => {
    const date = new Date(memory.timestamp);

    // Monthly distribution
    const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
    if (!distribution.monthly[monthKey]) {
      distribution.monthly[monthKey] = [];
    }
    distribution.monthly[monthKey].push(memory);

    // Weekly distribution (week of year)
    const weekKey = `${date.getFullYear()}-W${getWeekNumber(date)}`;
    if (!distribution.weekly[weekKey]) {
      distribution.weekly[weekKey] = [];
    }
    distribution.weekly[weekKey].push(memory);

    // Daily distribution (day of week)
    const dayKey = date.toLocaleDateString('en-US', { weekday: 'long' });
    if (!distribution.daily[dayKey]) {
      distribution.daily[dayKey] = [];
    }
    distribution.daily[dayKey].push(memory);

    // Hourly distribution
    const hourKey = date.getHours();
    if (!distribution.hourly[hourKey]) {
      distribution.hourly[hourKey] = [];
    }
    distribution.hourly[hourKey].push(memory);
  });

  return distribution;
}

/**
 * Calculate week number for a given date
 * @param {Date} date - Date object
 * @returns {number} Week number
 */
function getWeekNumber(date) {
  const firstDayOfYear = new Date(date.getFullYear(), 0, 1);
  const pastDaysOfYear = (date - firstDayOfYear) / 86400000;
  return Math.ceil((pastDaysOfYear + firstDayOfYear.getDay() + 1) / 7);
}

/**
 * Prepare temporal data for chart visualization
 * @param {Object} distribution - Distribution object from analyzeTemporalDistribution
 * @param {string} period - Time period ('monthly', 'weekly', 'daily', 'hourly')
 * @returns {Array} Chart-ready data array
 */
function prepareTemporalChartData(distribution, period = 'monthly') {
  const data = distribution[period];

  const chartData = Object.entries(data)
    .map(([key, memories]) => ({
      period: formatPeriodLabel(key, period),
      count: memories.length,
      memories: memories,
      key: key
    }))
    .sort((a, b) => a.key.localeCompare(b.key));

  return chartData;
}

/**
 * Format period labels for display
 * @param {string} key - Period key
 * @param {string} period - Period type
 * @returns {string} Formatted label
 */
function formatPeriodLabel(key, period) {
  switch (period) {
    case 'monthly':
      const [year, month] = key.split('-');
      const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
      return `${monthNames[parseInt(month) - 1]} ${year}`;

    case 'weekly':
      return key; // Already formatted as YYYY-WXX

    case 'daily':
      return key; // Day names are already formatted

    case 'hourly':
      const hour = parseInt(key);
      return `${hour}:00`;

    default:
      return key;
  }
}

// =============================================================================
// TAG ANALYSIS FUNCTIONS
// =============================================================================

/**
 * Analyze tag usage frequency and patterns
 * @param {Array} memories - Array of memory objects
 * @returns {Object} Tag analysis results
 */
function analyzeTagUsage(memories) {
  const tagFrequency = {};
  const tagCombinations = {};
  const categoryDistribution = {};

  memories.forEach(memory => {
    const tags = memory.tags || [];

    // Tag frequency analysis
    tags.forEach(tag => {
      tagFrequency[tag] = (tagFrequency[tag] || 0) + 1;

      // Categorize tags
      const category = categorizeTag(tag);
      if (!categoryDistribution[category]) {
        categoryDistribution[category] = {};
      }
      categoryDistribution[category][tag] = (categoryDistribution[category][tag] || 0) + 1;
    });

    // Tag combination analysis
    if (tags.length > 1) {
      for (let i = 0; i < tags.length; i++) {
        for (let j = i + 1; j < tags.length; j++) {
          const combo = [tags[i], tags[j]].sort().join(' + ');
          tagCombinations[combo] = (tagCombinations[combo] || 0) + 1;
        }
      }
    }
  });

  return {
    frequency: Object.entries(tagFrequency)
      .sort(([,a], [,b]) => b - a),
    combinations: Object.entries(tagCombinations)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 20), // Top 20 combinations
    categories: categoryDistribution,
    totalTags: Object.keys(tagFrequency).length,
    averageTagsPerMemory: memories.reduce((sum, m) => sum + (m.tags?.length || 0), 0) / memories.length
  };
}

/**
 * Categorize a tag based on common patterns
 * @param {string} tag - Tag to categorize
 * @returns {string} Category name
 */
function categorizeTag(tag) {
  const patterns = {
    'projects': /^(mcp-memory-service|memory-dashboard|github-integration|mcp-protocol)/,
    'technologies': /^(python|react|typescript|chromadb|git|docker|aws|npm)/,
    'activities': /^(testing|debugging|development|documentation|deployment|maintenance)/,
    'status': /^(resolved|in-progress|blocked|verified|completed|experimental)/,
    'content-types': /^(concept|architecture|tutorial|reference|example|guide)/,
    'temporal': /^(january|february|march|april|may|june|q1|q2|2025)/,
    'priorities': /^(urgent|high-priority|low-priority|critical)/
  };

  for (const [category, pattern] of Object.entries(patterns)) {
    if (pattern.test(tag)) {
      return category;
    }
  }

  return 'other';
}

/**
 * Find tag inconsistencies and suggest improvements
 * @param {Array} memories - Array of memory objects
 * @returns {Object} Consistency analysis results
 */
function analyzeTagConsistency(memories) {
  const inconsistencies = [];
  const suggestions = [];
  const patterns = {};

  memories.forEach((memory, index) => {
    const content = memory.content || '';
    const tags = memory.tags || [];

    // Common content patterns that should have corresponding tags
    const contentPatterns = {
      'test': /\b(test|testing|TEST)\b/i,
      'bug': /\b(bug|issue|error|problem)\b/i,
      'debug': /\b(debug|debugging|fix|fixed)\b/i,
      'documentation': /\b(document|guide|tutorial|readme)\b/i,
      'concept': /\b(concept|idea|design|architecture)\b/i,
      'implementation': /\b(implement|implementation|develop|development)\b/i
    };

    Object.entries(contentPatterns).forEach(([expectedTag, pattern]) => {
      if (pattern.test(content)) {
        const hasRelatedTag = tags.some(tag =>
          tag.includes(expectedTag) ||
          expectedTag.includes(tag.split('-')[0])
        );

        if (!hasRelatedTag) {
          inconsistencies.push({
            memoryIndex: index,
            type: 'missing-tag',
            expectedTag: expectedTag,
            content: content.substring(0, 100) + '...',
            currentTags: tags
          });
        }
      }
    });

    // Check for overly generic tags
    const genericTags = ['test', 'memory', 'note', 'temp', 'example'];
    const hasGenericOnly = tags.length > 0 &&
      tags.every(tag => genericTags.includes(tag));

    if (hasGenericOnly) {
      suggestions.push({
        memoryIndex: index,
        type: 'improve-specificity',
        suggestion: 'Replace generic tags with specific categories',
        currentTags: tags,
        content: content.substring(0, 100) + '...'
      });
    }
  });

  return {
    inconsistencies,
    suggestions,
    consistencyScore: ((memories.length - inconsistencies.length) / memories.length) * 100,
    totalIssues: inconsistencies.length + suggestions.length
  };
}

// =============================================================================
// CONTENT ANALYSIS FUNCTIONS
// =============================================================================

/**
 * Analyze content patterns and themes
 * @param {Array} memories - Array of memory objects
 * @returns {Object} Content analysis results
 */
function analyzeContentPatterns(memories) {
  const themes = {};
  const contentTypes = {};
  const wordFrequency = {};
  const lengthDistribution = {};

  memories.forEach(memory => {
    const content = memory.content || '';
    const words = extractKeywords(content);
    const contentType = detectContentType(content);

    // Theme analysis based on keywords
    words.forEach(word => {
      wordFrequency[word] = (wordFrequency[word] || 0) + 1;
    });

    // Content type distribution
    contentTypes[contentType] = (contentTypes[contentType] || 0) + 1;

    // Length distribution
    const lengthCategory = categorizeContentLength(content.length);
    lengthDistribution[lengthCategory] = (lengthDistribution[lengthCategory] || 0) + 1;
  });

  // Extract top themes from word frequency
  const topWords = Object.entries(wordFrequency)
    .sort(([,a], [,b]) => b - a)
    .slice(0, 50);

  return {
    themes: extractThemes(topWords),
    contentTypes,
    lengthDistribution,
    wordFrequency: topWords,
    averageLength: memories.reduce((sum, m) => sum + (m.content?.length || 0), 0) / memories.length
  };
}

/**
 * Extract keywords from content
 * @param {string} content - Memory content
 * @returns {Array} Array of keywords
 */
function extractKeywords(content) {
  const stopWords = new Set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
  ]);

  return content
    .toLowerCase()
    .replace(/[^\w\s-]/g, ' ') // Remove punctuation except hyphens
    .split(/\s+/)
    .filter(word =>
      word.length > 2 &&
      !stopWords.has(word) &&
      !word.match(/^\d+$/) // Exclude pure numbers
    );
}

/**
 * Detect content type based on patterns
 * @param {string} content - Memory content
 * @returns {string} Content type
 */
function detectContentType(content) {
  const patterns = {
    'code': /```|function\s*\(|class\s+\w+|import\s+\w+/,
    'documentation': /^#+\s|README|GUIDE|TUTORIAL/i,
    'issue': /issue|bug|error|problem|fix|resolved/i,
    'concept': /concept|idea|design|architecture|approach/i,
    'test': /test|testing|verify|validation|TEST/i,
    'configuration': /config|setup|installation|environment/i,
    'analysis': /analysis|report|summary|statistics|metrics/i
  };

  for (const [type, pattern] of Object.entries(patterns)) {
    if (pattern.test(content)) {
      return type;
    }
  }

  return 'general';
}

/**
 * Categorize content by length
 * @param {number} length - Content length in characters
 * @returns {string} Length category
 */
function categorizeContentLength(length) {
  if (length < 100) return 'very-short';
  if (length < 500) return 'short';
  if (length < 1500) return 'medium';
  if (length < 3000) return 'long';
  return 'very-long';
}

/**
 * Extract themes from word frequency data
 * @param {Array} topWords - Array of [word, frequency] pairs
 * @returns {Object} Organized themes
 */
function extractThemes(topWords) {
  const themeCategories = {
    technology: ['python', 'react', 'typescript', 'chromadb', 'git', 'docker', 'api', 'database'],
    development: ['development', 'implementation', 'code', 'programming', 'build', 'deploy'],
    testing: ['test', 'testing', 'debug', 'debugging', 'verification', 'quality'],
    project: ['project', 'service', 'system', 'application', 'platform', 'tool'],
    process: ['process', 'workflow', 'methodology', 'procedure', 'approach', 'strategy']
  };

  const themes = {};
  const wordMap = new Map(topWords);

  Object.entries(themeCategories).forEach(([theme, keywords]) => {
    themes[theme] = keywords
      .filter(keyword => wordMap.has(keyword))
      .map(keyword => ({ word: keyword, frequency: wordMap.get(keyword) }))
      .sort((a, b) => b.frequency - a.frequency);
  });

  return themes;
}

// =============================================================================
// QUALITY ANALYSIS FUNCTIONS
// =============================================================================

/**
 * Assess overall memory quality and organization
 * @param {Array} memories - Array of memory objects
 * @returns {Object} Quality assessment results
 */
function assessMemoryQuality(memories) {
  const metrics = {
    tagging: assessTaggingQuality(memories),
    content: assessContentQuality(memories),
    organization: assessOrganizationQuality(memories),
    searchability: assessSearchabilityQuality(memories)
  };

  // Calculate overall quality score
  const overallScore = Object.values(metrics)
    .reduce((sum, metric) => sum + metric.score, 0) / Object.keys(metrics).length;

  return {
    overallScore: Math.round(overallScore),
    metrics,
    recommendations: generateQualityRecommendations(metrics),
    totalMemories: memories.length
  };
}

/**
 * Assess tagging quality
 * @param {Array} memories - Array of memory objects
 * @returns {Object} Tagging quality assessment
 */
function assessTaggingQuality(memories) {
  let taggedCount = 0;
  let wellTaggedCount = 0;
  let totalTags = 0;

  memories.forEach(memory => {
    const tags = memory.tags || [];
    totalTags += tags.length;

    if (tags.length > 0) {
      taggedCount++;

      // Well-tagged: has 3+ tags from different categories
      if (tags.length >= 3) {
        const categories = new Set(tags.map(tag => categorizeTag(tag)));
        if (categories.size >= 2) {
          wellTaggedCount++;
        }
      }
    }
  });

  const taggedPercentage = (taggedCount / memories.length) * 100;
  const wellTaggedPercentage = (wellTaggedCount / memories.length) * 100;
  const averageTagsPerMemory = totalTags / memories.length;

  let score = 0;
  if (taggedPercentage >= 90) score += 40;
  else if (taggedPercentage >= 70) score += 30;
  else if (taggedPercentage >= 50) score += 20;

  if (wellTaggedPercentage >= 70) score += 30;
  else if (wellTaggedPercentage >= 50) score += 20;
  else if (wellTaggedPercentage >= 30) score += 10;

  if (averageTagsPerMemory >= 4) score += 30;
  else if (averageTagsPerMemory >= 3) score += 20;
  else if (averageTagsPerMemory >= 2) score += 10;

  return {
    score,
    taggedPercentage: Math.round(taggedPercentage),
    wellTaggedPercentage: Math.round(wellTaggedPercentage),
    averageTagsPerMemory: Math.round(averageTagsPerMemory * 10) / 10,
    issues: {
      untagged: memories.length - taggedCount,
      poorlyTagged: taggedCount - wellTaggedCount
    }
  };
}

/**
 * Assess content quality
 * @param {Array} memories - Array of memory objects
 * @returns {Object} Content quality assessment
 */
function assessContentQuality(memories) {
  let substantialContent = 0;
  let hasDescription = 0;
  let totalLength = 0;

  memories.forEach(memory => {
    const content = memory.content || '';
    totalLength += content.length;

    if (content.length >= 50) {
      substantialContent++;
    }

    if (content.length >= 200) {
      hasDescription++;
    }
  });

  const substantialPercentage = (substantialContent / memories.length) * 100;
  const descriptivePercentage = (hasDescription / memories.length) * 100;
  const averageLength = totalLength / memories.length;

  let score = 0;
  if (substantialPercentage >= 90) score += 50;
  else if (substantialPercentage >= 70) score += 35;
  else if (substantialPercentage >= 50) score += 20;

  if (descriptivePercentage >= 60) score += 30;
  else if (descriptivePercentage >= 40) score += 20;
  else if (descriptivePercentage >= 20) score += 10;

  if (averageLength >= 300) score += 20;
  else if (averageLength >= 150) score += 10;

  return {
    score,
    substantialPercentage: Math.round(substantialPercentage),
    descriptivePercentage: Math.round(descriptivePercentage),
    averageLength: Math.round(averageLength),
    issues: {
      tooShort: memories.length - substantialContent,
      lackingDescription: memories.length - hasDescription
    }
  };
}

/**
 * Assess organization quality
 * @param {Array} memories - Array of memory objects
 * @returns {Object} Organization quality assessment
 */
function assessOrganizationQuality(memories) {
  const tagAnalysis = analyzeTagUsage(memories);
  const categories = Object.keys(tagAnalysis.categories);
  const topTags = tagAnalysis.frequency.slice(0, 10);

  // Check for balanced tag distribution
  const tagDistribution = tagAnalysis.frequency.map(([, count]) => count);
  const maxUsage = Math.max(...tagDistribution);
  const minUsage = Math.min(...tagDistribution);
  const distributionBalance = minUsage / maxUsage;

  let score = 0;

  // Category diversity
  if (categories.length >= 5) score += 30;
  else if (categories.length >= 3) score += 20;
  else if (categories.length >= 2) score += 10;

  // Tag usage balance
  if (distributionBalance >= 0.3) score += 25;
  else if (distributionBalance >= 0.2) score += 15;
  else if (distributionBalance >= 0.1) score += 5;

  // Consistent tag combinations
  if (tagAnalysis.combinations.length >= 10) score += 25;
  else if (tagAnalysis.combinations.length >= 5) score += 15;

  // Avoid over-concentration
  const topTagUsagePercentage = (topTags[0]?.[1] || 0) / memories.length * 100;
  if (topTagUsagePercentage <= 30) score += 20;
  else if (topTagUsagePercentage <= 40) score += 10;

  return {
    score,
    categoryCount: categories.length,
    tagDistributionBalance: Math.round(distributionBalance * 100),
    topTagUsagePercentage: Math.round(topTagUsagePercentage),
    consistentCombinations: tagAnalysis.combinations.length,
    issues: {
      fewCategories: categories.length < 3,
      imbalancedDistribution: distributionBalance < 0.2,
      overConcentration: topTagUsagePercentage > 40
    }
  };
}

/**
 * Assess searchability quality
 * @param {Array} memories - Array of memory objects
 * @returns {Object} Searchability quality assessment
 */
function assessSearchabilityQuality(memories) {
  const contentAnalysis = analyzeContentPatterns(memories);
  const tagAnalysis = analyzeTagUsage(memories);

  // Calculate searchability metrics
  const keywordDiversity = Object.keys(contentAnalysis.wordFrequency).length;
  const tagDiversity = tagAnalysis.totalTags;
  const averageTagsPerMemory = tagAnalysis.averageTagsPerMemory;

  let score = 0;

  // Keyword diversity
  if (keywordDiversity >= 100) score += 25;
  else if (keywordDiversity >= 50) score += 15;
  else if (keywordDiversity >= 25) score += 5;

  // Tag diversity
  if (tagDiversity >= 50) score += 25;
  else if (tagDiversity >= 30) score += 15;
  else if (tagDiversity >= 15) score += 5;

  // Tag coverage
  if (averageTagsPerMemory >= 4) score += 25;
  else if (averageTagsPerMemory >= 3) score += 15;
  else if (averageTagsPerMemory >= 2) score += 5;

  // Content type diversity
  const contentTypes = Object.keys(contentAnalysis.contentTypes).length;
  if (contentTypes >= 5) score += 25;
  else if (contentTypes >= 3) score += 15;
  else if (contentTypes >= 2) score += 5;

  return {
    score,
    keywordDiversity,
    tagDiversity,
    averageTagsPerMemory: Math.round(averageTagsPerMemory * 10) / 10,
    contentTypeDiversity: contentTypes,
    issues: {
      lowKeywordDiversity: keywordDiversity < 25,
      lowTagDiversity: tagDiversity < 15,
      poorTagCoverage: averageTagsPerMemory < 2
    }
  };
}

/**
 * Generate quality improvement recommendations
 * @param {Object} metrics - Quality metrics object
 * @returns {Array} Array of recommendations
 */
function generateQualityRecommendations(metrics) {
  const recommendations = [];

  // Tagging recommendations
  if (metrics.tagging.taggedPercentage < 90) {
    recommendations.push({
      category: 'tagging',
      priority: 'high',
      issue: `${metrics.tagging.issues.untagged} memories are untagged`,
      action: 'Run memory maintenance session to tag untagged memories',
      expectedImprovement: 'Improve searchability and organization'
    });
  }

  if (metrics.tagging.averageTagsPerMemory < 3) {
    recommendations.push({
      category: 'tagging',
      priority: 'medium',
      issue: 'Low average tags per memory',
      action: 'Add more specific and categorical tags to existing memories',
      expectedImprovement: 'Better categorization and discoverability'
    });
  }

  // Content recommendations
  if (metrics.content.substantialPercentage < 80) {
    recommendations.push({
      category: 'content',
      priority: 'medium',
      issue: `${metrics.content.issues.tooShort} memories have minimal content`,
      action: 'Expand brief memories with more context and details',
      expectedImprovement: 'Increased information value and searchability'
    });
  }

  // Organization recommendations
  if (metrics.organization.categoryCount < 3) {
    recommendations.push({
      category: 'organization',
      priority: 'high',
      issue: 'Limited tag category diversity',
      action: 'Implement standardized tag schema with multiple categories',
      expectedImprovement: 'Better knowledge organization structure'
    });
  }

  if (metrics.organization.tagDistributionBalance < 20) {
    recommendations.push({
      category: 'organization',
      priority: 'medium',
      issue: 'Imbalanced tag usage distribution',
      action: 'Review and balance tag usage across content types',
      expectedImprovement: 'More consistent knowledge organization'
    });
  }

  // Searchability recommendations
  if (metrics.searchability.tagDiversity < 30) {
    recommendations.push({
      category: 'searchability',
      priority: 'medium',
      issue: 'Limited tag vocabulary',
      action: 'Expand tag vocabulary with more specific and varied tags',
      expectedImprovement: 'Enhanced search precision and recall'
    });
  }

  return recommendations.sort((a, b) => {
    const priorityOrder = { 'high': 3, 'medium': 2, 'low': 1 };
    return priorityOrder[b.priority] - priorityOrder[a.priority];
  });
}

// =============================================================================
// VISUALIZATION DATA PREPARATION
// =============================================================================

/**
 * Prepare comprehensive data package for visualizations
 * @param {Array} memories - Array of memory objects
 * @returns {Object} Complete visualization data package
 */
function prepareVisualizationData(memories) {
  const temporal = analyzeTemporalDistribution(memories);
  const tags = analyzeTagUsage(memories);
  const content = analyzeContentPatterns(memories);
  const quality = assessMemoryQuality(memories);

  return {
    metadata: {
      totalMemories: memories.length,
      analysisDate: new Date().toISOString(),
      dataVersion: '1.0'
    },

    // Chart data for different visualizations
    charts: {
      temporalDistribution: prepareTemporalChartData(temporal, 'monthly'),
      weeklyPattern: prepareTemporalChartData(temporal, 'weekly'),
      dailyPattern: prepareTemporalChartData(temporal, 'daily'),
      hourlyPattern: prepareTemporalChartData(temporal, 'hourly'),

      tagFrequency: tags.frequency.slice(0, 20).map(([tag, count]) => ({
        tag,
        count,
        category: categorizeTag(tag)
      })),

      tagCombinations: tags.combinations.slice(0, 10).map(([combo, count]) => ({
        combination: combo,
        count,
        tags: combo.split(' + ')
      })),

      contentTypes: Object.entries(content.contentTypes).map(([type, count]) => ({
        type,
        count,
        percentage: Math.round((count / memories.length) * 100)
      })),

      contentLengths: Object.entries(content.lengthDistribution).map(([category, count]) => ({
        category,
        count,
        percentage: Math.round((count / memories.length) * 100)
      }))
    },

    // Summary statistics
    statistics: {
      temporal: {
        peakMonth: findPeakPeriod(temporal.monthly),
        mostActiveDay: findPeakPeriod(temporal.daily),
        mostActiveHour: findPeakPeriod(temporal.hourly)
      },

      tags: {
        totalUniqueTags: tags.totalTags,
        averageTagsPerMemory: Math.round(tags.averageTagsPerMemory * 10) / 10,
        mostUsedTag: tags.frequency[0],
        categoryDistribution: Object.keys(tags.categories).length
      },

      content: {
        averageLength: Math.round(content.averageLength),
        mostCommonType: Object.entries(content.contentTypes)
          .sort(([,a], [,b]) => b - a)[0],
        keywordCount: Object.keys(content.wordFrequency).length
      },

      quality: {
        overallScore: quality.overallScore,
        taggedPercentage: quality.metrics.tagging.taggedPercentage,
        organizationScore: quality.metrics.organization.score,
        recommendationCount: quality.recommendations.length
      }
    },

    // Raw analysis data for advanced processing
    rawData: {
      temporal,
      tags,
      content,
      quality
    }
  };
}

/**
 * Find peak period from distribution data
 * @param {Object} distribution - Distribution object
 * @returns {Object} Peak period information
 */
function findPeakPeriod(distribution) {
  const entries = Object.entries(distribution);
  if (entries.length === 0) return null;

  const peak = entries.reduce((max, [period, memories]) =>
    memories.length > max.count ? { period, count: memories.length } : max,
    { period: null, count: 0 }
  );

  return peak;
}

// =============================================================================
// EXPORT FUNCTIONS
// =============================================================================

/**
 * Export analysis results to various formats
 * @param {Object} analysisData - Complete analysis data
 * @param {string} format - Export format ('json', 'csv', 'summary')
 * @returns {string} Formatted export data
 */
function exportAnalysisData(analysisData, format = 'json') {
  switch (format) {
    case 'json':
      return JSON.stringify(analysisData, null, 2);

    case 'csv':
      return exportToCSV(analysisData);

    case 'summary':
      return generateSummaryReport(analysisData);

    default:
      throw new Error(`Unsupported export format: ${format}`);
  }
}

/**
 * Export key metrics to CSV format
 * @param {Object} analysisData - Analysis data
 * @returns {string} CSV formatted data
 */
function exportToCSV(analysisData) {
  const csvSections = [];

  // Temporal data
  csvSections.push('TEMPORAL DISTRIBUTION');
  csvSections.push('Month,Count');
  analysisData.charts.temporalDistribution.forEach(item => {
    csvSections.push(`${item.period},${item.count}`);
  });
  csvSections.push('');

  // Tag frequency
  csvSections.push('TAG FREQUENCY');
  csvSections.push('Tag,Count,Category');
  analysisData.charts.tagFrequency.forEach(item => {
    csvSections.push(`${item.tag},${item.count},${item.category}`);
  });
  csvSections.push('');

  // Content types
  csvSections.push('CONTENT TYPES');
  csvSections.push('Type,Count,Percentage');
  analysisData.charts.contentTypes.forEach(item => {
    csvSections.push(`${item.type},${item.count},${item.percentage}%`);
  });

  return csvSections.join('\n');
}

/**
 * Generate a human-readable summary report
 * @param {Object} analysisData - Analysis data
 * @returns {string} Summary report
 */
function generateSummaryReport(analysisData) {
  const stats = analysisData.statistics;
  const quality = analysisData.rawData.quality;

  return `
MEMORY ANALYSIS SUMMARY REPORT
Generated: ${new Date().toLocaleDateString()}

DATABASE OVERVIEW:
- Total Memories: ${analysisData.metadata.totalMemories}
- Overall Quality Score: ${stats.quality.overallScore}/100
- Tagged Memories: ${stats.quality.taggedPercentage}%

TEMPORAL PATTERNS:
- Peak Activity: ${stats.temporal.peakMonth?.period} (${stats.temporal.peakMonth?.count} memories)
- Most Active Day: ${stats.temporal.mostActiveDay?.period}
- Most Active Hour: ${stats.temporal.mostActiveHour?.period}:00

TAG ANALYSIS:
- Unique Tags: ${stats.tags.totalUniqueTags}
- Average Tags per Memory: ${stats.tags.averageTagsPerMemory}
- Most Used Tag: ${stats.tags.mostUsedTag?.[0]} (${stats.tags.mostUsedTag?.[1]} uses)
- Tag Categories: ${stats.tags.categoryDistribution}

CONTENT INSIGHTS:
- Average Length: ${stats.content.averageLength} characters
- Most Common Type: ${stats.content.mostCommonType?.[0]}
- Unique Keywords: ${stats.content.keywordCount}

QUALITY RECOMMENDATIONS:
${quality.recommendations.slice(0, 3).map(rec =>
  `- ${rec.priority.toUpperCase()}: ${rec.action}`
).join('\n')}

For detailed analysis, use the full JSON export or visualization tools.
`.trim();
}

// =============================================================================
// MAIN ANALYSIS PIPELINE
// =============================================================================

/**
 * Run complete analysis pipeline on memory data
 * @param {Array} memories - Array of memory objects
 * @returns {Object} Complete analysis results
 */
async function runCompleteAnalysis(memories) {
  console.log('Starting comprehensive memory analysis...');

  const startTime = Date.now();

  try {
    // Run all analysis functions
    const results = prepareVisualizationData(memories);

    const endTime = Date.now();
    const duration = endTime - startTime;

    console.log(`Analysis complete in ${duration}ms`);
    console.log(`Analyzed ${memories.length} memories`);
    console.log(`Overall quality score: ${results.statistics.quality.overallScore}/100`);

    return {
      ...results,
      meta: {
        analysisDuration: duration,
        analysisTimestamp: new Date().toISOString(),
        version: '1.0'
      }
    };

  } catch (error) {
    console.error('Analysis failed:', error);
    throw error;
  }
}

// Export all functions for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    // Temporal analysis
    analyzeTemporalDistribution,
    prepareTemporalChartData,

    // Tag analysis
    analyzeTagUsage,
    analyzeTagConsistency,
    categorizeTag,

    // Content analysis
    analyzeContentPatterns,
    detectContentType,
    extractKeywords,

    // Quality analysis
    assessMemoryQuality,
    generateQualityRecommendations,

    // Visualization
    prepareVisualizationData,

    // Export utilities
    exportAnalysisData,
    generateSummaryReport,

    // Main pipeline
    runCompleteAnalysis
  };
}

/**
 * Usage Examples:
 *
 * // Basic usage with MCP Memory Service data
 * const memories = await retrieve_memory({ query: "all memories", n_results: 500 });
 * const analysis = await runCompleteAnalysis(memories);
 *
 * // Specific analyses
 * const temporalData = analyzeTemporalDistribution(memories);
 * const tagAnalysis = analyzeTagUsage(memories);
 * const qualityReport = assessMemoryQuality(memories);
 *
 * // Export results
 * const jsonExport = exportAnalysisData(analysis, 'json');
 * const csvExport = exportAnalysisData(analysis, 'csv');
 * const summary = exportAnalysisData(analysis, 'summary');
 *
 * // Prepare data for React charts
 * const chartData = prepareVisualizationData(memories);
 * // Use chartData.charts.temporalDistribution with the React component
 */
