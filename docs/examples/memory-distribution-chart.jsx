import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

/**
 * Memory Distribution Chart Component
 *
 * A comprehensive visualization component for displaying monthly memory storage
 * distribution with insights, statistics, and interactive features.
 *
 * Features:
 * - Responsive bar chart with monthly distribution
 * - Custom tooltips with percentages
 * - Statistics cards for key metrics
 * - Automatic insights generation
 * - Professional styling and layout
 *
 * Usage:
 * 1. Install dependencies: npm install recharts
 * 2. Import and use: <MemoryDistributionChart data={yourData} />
 * 3. Or use with sample data as shown below
 */

const MemoryDistributionChart = ({ data = null, title = "Memory Storage Distribution by Month" }) => {
  // Sample data based on real MCP Memory Service analysis
  // Replace with actual data from your analytics pipeline
  const defaultData = [
    { month: "Jan 2025", count: 50, monthKey: "2025-01" },
    { month: "Feb 2025", count: 15, monthKey: "2025-02" },
    { month: "Mar 2025", count: 8, monthKey: "2025-03" },
    { month: "Apr 2025", count: 12, monthKey: "2025-04" },
    { month: "May 2025", count: 4, monthKey: "2025-05" },
    { month: "Jun 2025", count: 45, monthKey: "2025-06" }
  ];

  const monthlyData = data || defaultData;
  const totalMemories = monthlyData.reduce((sum, item) => sum + item.count, 0);

  // Calculate statistics
  const peakMonth = monthlyData.reduce((max, item) =>
    item.count > max.count ? item : max, monthlyData[0]);
  const averagePerMonth = (totalMemories / monthlyData.length).toFixed(1);

  // Find most recent month with data
  const recentMonth = monthlyData[monthlyData.length - 1];

  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const percentage = ((data.count / totalMemories) * 100).toFixed(1);

      return (
        <div className="bg-white p-3 border border-gray-300 rounded-lg shadow-lg">
          <p className="font-semibold text-gray-800">{label}</p>
          <p className="text-blue-600">
            <span className="font-medium">Memories: </span>
            {data.count}
          </p>
          <p className="text-gray-600">
            <span className="font-medium">Percentage: </span>
            {percentage}%
          </p>
        </div>
      );
    }
    return null;
  };

  // Custom label function for bars
  const renderCustomLabel = (entry) => {
    if (entry.count > 5) { // Only show labels for bars with more than 5 memories
      return entry.count;
    }
    return null;
  };

  // Generate insights based on data patterns
  const generateInsights = () => {
    const insights = [];

    // Peak activity insight
    const peakPercentage = ((peakMonth.count / totalMemories) * 100).toFixed(1);
    insights.push(`Peak activity in ${peakMonth.month} (${peakPercentage}% of total memories)`);

    // Recent activity insight
    const recentPercentage = ((recentMonth.count / totalMemories) * 100).toFixed(1);
    if (recentMonth.count > averagePerMonth) {
      insights.push(`High recent activity: ${recentMonth.month} above average`);
    }

    // Growth pattern insight
    const firstMonth = monthlyData[0];
    const lastMonth = monthlyData[monthlyData.length - 1];
    if (lastMonth.count > firstMonth.count * 0.8) {
      insights.push(`Sustained activity: Recent months maintain high productivity`);
    }

    return insights;
  };

  const insights = generateInsights();

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gray-50 rounded-lg">
      {/* Header Section */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          {title}
        </h2>
        <p className="text-gray-600">
          Total memories analyzed: <span className="font-semibold text-blue-600">{totalMemories}</span> memories
        </p>
      </div>

      {/* Main Chart */}
      <div className="bg-white p-4 rounded-lg shadow-sm mb-6">
        <ResponsiveContainer width="100%" height={400}>
          <BarChart
            data={monthlyData}
            margin={{
              top: 20,
              right: 30,
              left: 20,
              bottom: 5,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="month"
              tick={{ fontSize: 12 }}
              tickLine={{ stroke: '#d1d5db' }}
              axisLine={{ stroke: '#d1d5db' }}
            />
            <YAxis
              tick={{ fontSize: 12 }}
              tickLine={{ stroke: '#d1d5db' }}
              axisLine={{ stroke: '#d1d5db' }}
              label={{
                value: 'Number of Memories',
                angle: -90,
                position: 'insideLeft',
                style: { textAnchor: 'middle', fontSize: '12px', fill: '#6b7280' }
              }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Bar
              dataKey="count"
              name="Memories Stored"
              fill="#3b82f6"
              radius={[4, 4, 0, 0]}
              label={renderCustomLabel}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-blue-50 p-4 rounded-lg">
          <h3 className="font-semibold text-blue-800 mb-2">Peak Month</h3>
          <p className="text-lg font-bold text-blue-600">{peakMonth.month}</p>
          <p className="text-sm text-blue-600">
            {peakMonth.count} memories ({((peakMonth.count / totalMemories) * 100).toFixed(1)}%)
          </p>
        </div>

        <div className="bg-green-50 p-4 rounded-lg">
          <h3 className="font-semibold text-green-800 mb-2">Recent Activity</h3>
          <p className="text-lg font-bold text-green-600">{recentMonth.month}</p>
          <p className="text-sm text-green-600">
            {recentMonth.count} memories ({((recentMonth.count / totalMemories) * 100).toFixed(1)}%)
          </p>
        </div>

        <div className="bg-amber-50 p-4 rounded-lg">
          <h3 className="font-semibold text-amber-800 mb-2">Average/Month</h3>
          <p className="text-lg font-bold text-amber-600">{averagePerMonth}</p>
          <p className="text-sm text-amber-600">memories per month</p>
        </div>
      </div>

      {/* Insights Section */}
      <div className="bg-white p-4 rounded-lg shadow-sm">
        <h3 className="font-semibold text-gray-800 mb-3">📊 Data Insights</h3>
        <div className="space-y-2">
          {insights.map((insight, index) => (
            <div key={index} className="flex items-start">
              <span className="text-blue-500 mr-2">•</span>
              <p className="text-sm text-gray-600">{insight}</p>
            </div>
          ))}
        </div>

        <div className="mt-4 pt-4 border-t border-gray-200">
          <p className="text-xs text-gray-500">
            <strong>Analysis Pattern:</strong> This distribution shows typical software development
            lifecycle phases - high initial activity (project setup), consolidation periods,
            and renewed intensive development phases.
          </p>
        </div>
      </div>
    </div>
  );
};

export default MemoryDistributionChart;

/**
 * Usage Examples:
 *
 * 1. Basic Usage (with sample data):
 * <MemoryDistributionChart />
 *
 * 2. With Custom Data:
 * const myData = [
 *   { month: "Jan 2025", count: 25, monthKey: "2025-01" },
 *   { month: "Feb 2025", count: 30, monthKey: "2025-02" },
 *   // ... more data
 * ];
 * <MemoryDistributionChart data={myData} title="My Project Analysis" />
 *
 * 3. Integration with MCP Memory Service:
 *
 * async function loadMemoryData() {
 *   const memories = await recall_memory({
 *     "query": "memories from this year",
 *     "n_results": 500
 *   });
 *
 *   // Process memories into chart format
 *   const processedData = processMemoriesForChart(memories);
 *   return processedData;
 * }
 *
 * function processMemoriesForChart(memories) {
 *   const monthlyDistribution = {};
 *
 *   memories.forEach(memory => {
 *     const date = new Date(memory.timestamp);
 *     const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
 *
 *     if (!monthlyDistribution[monthKey]) {
 *       monthlyDistribution[monthKey] = 0;
 *     }
 *     monthlyDistribution[monthKey]++;
 *   });
 *
 *   return Object.entries(monthlyDistribution)
 *     .sort(([a], [b]) => a.localeCompare(b))
 *     .map(([month, count]) => {
 *       const [year, monthNum] = month.split('-');
 *       const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
 *                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
 *       const monthName = monthNames[parseInt(monthNum) - 1];
 *
 *       return {
 *         month: `${monthName} ${year}`,
 *         count: count,
 *         monthKey: month
 *       };
 *     });
 * }
 *
 * Dependencies:
 * npm install recharts
 *
 * For Tailwind CSS styling, ensure you have Tailwind configured in your project.
 */
