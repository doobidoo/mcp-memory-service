/**
 * Mock data for MCP Memory Service video visualizations
 *
 * This file contains generated/mocked data for animations and visualizations.
 * Real project data is extracted separately via extract-project-data.ts
 */

/**
 * 3D positions for vector embeddings visualization (AI/ML section)
 * Generates 30 spheres in 3 clusters representing semantic similarity
 */
export const mockEmbeddings = Array.from({ length: 30 }, (_, i) => {
  const cluster = Math.floor(i / 10);
  const clusterCenters = [
    [-4, 2, 1],
    [0, -1, 0],
    [4, 1, -2],
  ];
  const center = clusterCenters[cluster];

  return {
    id: `mem_${i}`,
    position: [
      center[0] + (Math.random() - 0.5) * 2.5,
      center[1] + (Math.random() - 0.5) * 2.5,
      center[2] + (Math.random() - 0.5) * 2.5,
    ] as [number, number, number],
    color: ['#8B5CF6', '#EC4899', '#10B981'][cluster],
    size: 0.3 + Math.random() * 0.2,
  };
});

/**
 * Response time comparison data for backend performance chart
 */
export const backendComparison = [
  { name: 'SQLite-Vec', time: 5, color: '#10B981' },
  { name: 'Hybrid', time: 5, color: '#3B82F6' },
  { name: 'Cloudflare', time: 45, color: '#8B5CF6' },
];

/**
 * Dashboard mock memories for UI preview (Developer Experience section)
 */
export const mockMemories = [
  {
    content: 'User prefers dark mode in all applications',
    tags: ['preference', 'ui'],
    quality: 0.92,
    timestamp: '2 hours ago',
  },
  {
    content: 'Project uses TypeScript with strict mode enabled',
    tags: ['project', 'config'],
    quality: 0.87,
    timestamp: '1 day ago',
  },
  {
    content: 'API rate limit is 100 requests per minute',
    tags: ['api', 'limits'],
    quality: 0.95,
    timestamp: '3 days ago',
  },
  {
    content: 'Database backup runs daily at 2 AM UTC',
    tags: ['ops', 'schedule'],
    quality: 0.89,
    timestamp: '1 week ago',
  },
];

/**
 * Architecture layer data for diagram visualization
 */
export const architectureLayers = [
  {
    title: 'MCP Server Layer',
    items: ['35 Tools', 'Global Caching', 'Client Detection'],
    color: '#3B82F6',
  },
  {
    title: 'Storage Strategy',
    items: ['SQLite-Vec', 'Cloudflare', 'Hybrid'],
    color: '#8B5CF6',
  },
  {
    title: 'Service Layer',
    items: ['Memory Service', 'Quality System', 'Consolidation'],
    color: '#10B981',
  },
];

/**
 * Design pattern badges for architecture section
 */
export const designPatterns = [
  'Strategy',
  'Singleton',
  'Orchestrator',
  'Factory',
  'Observer',
];
