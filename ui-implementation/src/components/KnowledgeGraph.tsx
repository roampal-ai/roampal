import React, { useEffect, useState, useRef } from 'react';
import { apiFetch } from '../utils/fetch';
import { ROAMPAL_CONFIG } from '../config/roampal';

interface GraphNode {
  id: string;
  label: string;
  type: string;
  source?: 'routing' | 'content' | 'both' | 'action';  // v0.2.1: Triple KG system
  best_collection?: string;
  success_rate?: number;
  usage_count?: number;
  mentions?: number;  // v0.2.0: From content KG
  hybridScore?: number; // Calculated: √usage × √(quality + 0.1)
  last_used?: string;   // ISO timestamp
  created_at?: string;  // ISO timestamp
}

interface GraphEdge {
  source: string;
  target: string;
  weight?: number;
  type?: string;
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

interface ConceptDefinition {
  id: string;
  label: string;
  definition?: string;
  type?: string;
  best_collection?: string;
  success_rate?: number;
  usage_count?: number;
  related_concepts?: string[];
  last_used?: string;   // ISO timestamp
  created_at?: string;  // ISO timestamp

  // Enhanced detail fields
  total_searches?: number;
  outcome_breakdown?: {
    worked: number;
    failed: number;
    partial: number;
  };
  collections_breakdown?: Record<string, {
    successes: number;
    failures: number;
    total: number;
  }>;
  related_concepts_with_stats?: Array<{
    concept: string;
    co_occurrence: number;
    success_together: number;
    failure_together: number;
    success_rate: number;
  }>;
  context_snippet?: string;
  confidence?: number;
}

interface KnowledgeGraphProps {
  searchQuery?: string;
}

// Visualization constants
const MAX_NODES_DISPLAYED = 20; // Limit to fit on screen without scroll
const NODE_BASE_RADIUS = 5;      // Minimum node size (px)
const NODE_SCORE_MULTIPLIER = 5; // Scale factor for hybrid score (increased for early-stage visibility)

const KnowledgeGraph: React.FC<KnowledgeGraphProps> = ({ searchQuery = '' }) => {
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] });
  const [filteredData, setFilteredData] = useState<GraphData>({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedConcept, setSelectedConcept] = useState<ConceptDefinition | null>(null);
  const [conceptDefinition, setConceptDefinition] = useState<string>('');
  const [loadingDefinition, setLoadingDefinition] = useState(false);
  const [sortBy, setSortBy] = useState<'hybrid' | 'recent' | 'oldest'>('hybrid');
  const [timeFilter, setTimeFilter] = useState<'all' | 'today' | 'week' | 'session'>('all');
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const nodesRef = useRef<Map<string, { x: number; y: number; vx: number; vy: number }>>(new Map());

  useEffect(() => {
    // Track session start time for "This Session" filter
    if (!sessionStorage.getItem('kg_session_start')) {
      sessionStorage.setItem('kg_session_start', new Date().toISOString());
    }

    fetchGraphData();
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  const fetchGraphData = async () => {
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/memory/knowledge-graph`);
      if (!response.ok) throw new Error('Failed to fetch graph data');
      const data = await response.json();

      // Calculate hybrid score for each node: √usage × √(quality + 0.1)
      // Formula rewards both high usage AND high success rate
      const nodesWithScore = (data.nodes || []).map((node: any) => ({
        ...node,
        hybridScore: Math.sqrt(node.usage_count || 0) * Math.sqrt((node.success_rate || 0) + 0.1)
      }));

      // Store all nodes (will filter/sort later based on user controls)
      setGraphData({ nodes: nodesWithScore, edges: data.edges || [] });
      setLoading(false);
      if (graphData.nodes && graphData.nodes.length > 0) {
        initializeNodePositions(graphData.nodes);
      }
    } catch (err) {
      setError('Failed to load knowledge graph');
      setLoading(false);
    }
  };

  // Apply time filter and sorting
  const applyFilters = (data: GraphData) => {
    if (!data.nodes || data.nodes.length === 0) {
      setFilteredData({ nodes: [], edges: [] });
      return;
    }

    const now = new Date();
    const sessionStart = new Date(sessionStorage.getItem('kg_session_start') || now);

    // Step 1: Apply time filter
    const filteredByTime = data.nodes.filter((node: GraphNode) => {
      if (timeFilter === 'all') return true;
      if (!node.last_used) return false;

      const lastUsed = new Date(node.last_used);

      if (timeFilter === 'today') {
        return lastUsed.toDateString() === now.toDateString();
      } else if (timeFilter === 'week') {
        const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        return lastUsed >= weekAgo;
      } else if (timeFilter === 'session') {
        return lastUsed >= sessionStart;
      }
      return true;
    });

    // Step 2: Sort based on selected method
    const sorted = [...filteredByTime].sort((a: GraphNode, b: GraphNode) => {
      if (sortBy === 'hybrid') {
        return (b.hybridScore || 0) - (a.hybridScore || 0);
      } else if (sortBy === 'recent') {
        const aTime = a.last_used ? new Date(a.last_used).getTime() : 0;
        const bTime = b.last_used ? new Date(b.last_used).getTime() : 0;
        return bTime - aTime;
      } else if (sortBy === 'oldest') {
        const aTime = a.created_at ? new Date(a.created_at).getTime() : Date.now();
        const bTime = b.created_at ? new Date(b.created_at).getTime() : Date.now();
        return aTime - bTime;
      }
      return 0;
    });

    // Step 3: Take top 20
    const topNodes = sorted.slice(0, MAX_NODES_DISPLAYED);

    // Step 4: Filter edges to only show connections between displayed nodes
    const topNodeIds = new Set(topNodes.map((n: GraphNode) => n.id));
    const filteredEdges = data.edges.filter((edge: GraphEdge) =>
      topNodeIds.has(edge.source) && topNodeIds.has(edge.target)
    );

    const result = { nodes: topNodes, edges: filteredEdges };
    setFilteredData(result);

    // Initialize node positions if this is the first render
    if (topNodes.length > 0) {
      initializeNodePositions(topNodes);
    }
  };

  // Re-filter when sort or time filter changes
  useEffect(() => {
    applyFilters(graphData);
  }, [sortBy, timeFilter, graphData]);

  // Apply search filter
  useEffect(() => {
    if (!searchQuery) {
      applyFilters(graphData);
      if (graphData.nodes && graphData.nodes.length > 0) {
        initializeNodePositions(graphData.nodes);
      }
      return;
    }

    const query = searchQuery.toLowerCase();
    const filteredNodes = graphData.nodes.filter(node =>
      node.label.toLowerCase().includes(query) ||
      node.type?.toLowerCase().includes(query) ||
      node.best_collection?.toLowerCase().includes(query)
    );

    const nodeIds = new Set(filteredNodes.map(n => n.id));
    const filteredEdges = graphData.edges.filter(edge =>
      nodeIds.has(edge.source) || nodeIds.has(edge.target)
    );

    // Add connected nodes to make the graph more meaningful
    filteredEdges.forEach(edge => {
      if (!nodeIds.has(edge.source)) {
        const sourceNode = graphData.nodes.find(n => n.id === edge.source);
        if (sourceNode) {
          filteredNodes.push(sourceNode);
          nodeIds.add(edge.source);
        }
      }
      if (!nodeIds.has(edge.target)) {
        const targetNode = graphData.nodes.find(n => n.id === edge.target);
        if (targetNode) {
          filteredNodes.push(targetNode);
          nodeIds.add(edge.target);
        }
      }
    });

    const filtered = { nodes: filteredNodes, edges: filteredEdges };
    setFilteredData(filtered);
    if (filteredNodes.length > 0) {
      initializeNodePositions(filteredNodes);
    }
  }, [searchQuery, graphData]);

  // Helper function to format concept info from the routing graph
  const formatConceptInfo = (concept: GraphNode): string => {
    // Show what the KG actually tracks - routing and performance data
    const collection = concept.best_collection || 'general';
    const successRate = Math.round((concept.success_rate || 0) * 100);
    const usageCount = concept.usage_count || 0;
    const hybridScore = (concept.hybridScore || 0).toFixed(2);

    return `💡 Pattern: "${concept.label}"\n\n` +
           `Quality Score: ${successRate}% (conversations where this helped)\n` +
           `Usage: ${usageCount}x referenced\n` +
           `Best Memory Type: ${collection}\n` +
           `Hybrid Score: ${hybridScore} (determines node size)\n\n` +
           `The system learned this pattern from past conversations. Higher quality scores mean this approach consistently works well. ` +
           `Node size reflects both quality AND usage frequency.`;
  };


  const fetchConceptDefinition = async (concept: GraphNode) => {
    setSelectedConcept(concept as ConceptDefinition);
    setLoadingDefinition(true);

    try {
      // Fetch actual definition from the API
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/memory/knowledge-graph/concept/${concept.id}/definition`);
      if (response.ok) {
        const data = await response.json();
        console.log('[KG] API response for', concept.id, ':', data);

        // Check if we got a useful definition (not generic fallback)
        const isGenericFallback = data.definition && data.definition.includes("is a tracked concept representing");

        if (data.definition && data.definition !== "Unable to retrieve definition" && !isGenericFallback) {
          // Use the definition from memory search
          setConceptDefinition(data.definition);
        } else if (data.collections_breakdown && Object.keys(data.collections_breakdown).length > 0) {
          // Build summary from actual routing data
          const totalSearches = data.total_searches || 0;
          const worked = data.outcome_breakdown?.worked || 0;
          const failed = data.outcome_breakdown?.failed || 0;
          const successRate = totalSearches > 0 ? Math.round((worked / (worked + failed)) * 100) : 0;

          let summary = `This concept has been used in ${totalSearches} memory searches.\n\n`;
          summary += `**Performance**: ${worked} worked, ${failed} failed (${successRate}% success rate)\n\n`;
          summary += `**Best performing collection**: ${data.best_collection || 'unknown'}\n\n`;
          summary += `The system routes queries mentioning "${concept.label}" based on learned patterns from past interactions.`;

          setConceptDefinition(summary);
        } else {
          // Ultimate fallback
          setConceptDefinition(formatConceptInfo(concept));
        }

        // Update selected concept with additional data from API
        setSelectedConcept({
          ...concept,
          related_concepts: data.related_concepts || [],
          related_concepts_with_stats: data.related_concepts_with_stats || [],
          sources: data.sources || [],
          metadata: data.metadata || {},
          // Enhanced detail fields from backend
          total_searches: data.total_searches,
          outcome_breakdown: data.outcome_breakdown,
          collections_breakdown: data.collections_breakdown,
          context_snippet: data.context_snippet,
          confidence: data.confidence
        } as ConceptDefinition);
      } else {
        // No definition from KG
        setConceptDefinition(formatConceptInfo(concept));
      }
    } catch (error) {
      // Fallback when API fails
      setConceptDefinition(formatConceptInfo(concept));
    }

    setLoadingDefinition(false);
  };

  const initializeNodePositions = (nodes: GraphNode[]) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Use CSS dimensions for positioning (not scaled canvas dimensions)
    const rect = canvas.getBoundingClientRect();
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const radius = Math.min(centerX, centerY) * 0.6;

    nodes.forEach((node, index) => {
      const angle = (2 * Math.PI * index) / nodes.length;
      nodesRef.current.set(node.id, {
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
        vx: 0,
        vy: 0
      });
    });

    startAnimation();
  };

  const startAnimation = () => {
    const animate = () => {
      updatePhysics();
      draw();
      animationRef.current = requestAnimationFrame(animate);
    };
    animate();
  };

  const updatePhysics = () => {
    const nodes = nodesRef.current;
    const damping = 0.95;
    const repulsion = 50;
    const attraction = 0.001;

    // Apply forces
    nodes.forEach((node1, id1) => {
      // Center attraction
      const canvas = canvasRef.current;
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        node1.vx += (centerX - node1.x) * attraction;
        node1.vy += (centerY - node1.y) * attraction;
      }

      // Node repulsion
      nodes.forEach((node2, id2) => {
        if (id1 !== id2) {
          const dx = node1.x - node2.x;
          const dy = node1.y - node2.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = repulsion / (dist * dist);
          node1.vx += (dx / dist) * force;
          node1.vy += (dy / dist) * force;
        }
      });

      // Apply velocity
      node1.vx *= damping;
      node1.vy *= damping;
      node1.x += node1.vx;
      node1.y += node1.vy;
    });
  };

  const draw = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    // Get logical dimensions
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    // Clear canvas with visible background
    ctx.fillStyle = '#18181b';
    ctx.fillRect(0, 0, width, height);

    // Draw border to make canvas visible
    ctx.strokeStyle = '#27272a';
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, width - 1, height - 1);

    // Enable better text rendering
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    // Draw edges with weights
    filteredData.edges.forEach(edge => {
      const source = nodesRef.current.get(edge.source);
      const target = nodesRef.current.get(edge.target);
      if (source && target) {
        // Edge line
        const weight = edge.weight || 1;
        ctx.strokeStyle = `rgba(39, 39, 42, ${Math.min(weight, 1)})`;
        ctx.lineWidth = Math.max(1, weight * 2);
        ctx.beginPath();
        ctx.moveTo(source.x, source.y);
        ctx.lineTo(target.x, target.y);
        ctx.stroke();

        // Edge label (relationship strength)
        if (edge.weight && edge.weight > 0.5) {
          const midX = (source.x + target.x) / 2;
          const midY = (source.y + target.y) / 2;
          ctx.fillStyle = '#52525b';
          ctx.font = '10px monospace';
          ctx.textAlign = 'center';
          ctx.fillText(`${Math.round(edge.weight * 100)}%`, midX, midY - 3);
        }
      }
    });

    // Draw nodes
    filteredData.nodes.forEach(node => {
      const pos = nodesRef.current.get(node.id);
      if (!pos) return;

      // Node circle - size based on hybrid score (usage × quality)
      const hybridScore = node.hybridScore || 0;
      const radius = NODE_BASE_RADIUS + hybridScore * NODE_SCORE_MULTIPLIER;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, 2 * Math.PI);

      // Color based on source (v0.2.1: Triple KG system)
      const source = node.source || 'routing';  // Default to routing for backward compatibility
      if (source === 'action') {
        // Orange: Action effectiveness patterns (context|action|collection)
        ctx.fillStyle = '#f97316';  // Orange-500
      } else if (source === 'both') {
        // Purple: Exists in BOTH routing + content graphs
        ctx.fillStyle = '#a855f7';  // Purple-500
      } else if (source === 'content') {
        // Green: Content KG only (memory-based entities)
        ctx.fillStyle = '#22c55e';  // Green-500
      } else {
        // Blue: Routing KG only (query-based patterns)
        ctx.fillStyle = '#3b82f6';  // Blue-500
      }
      ctx.fill();

      ctx.strokeStyle = '#52525b';
      ctx.stroke();

      // Node label
      ctx.fillStyle = '#e4e4e7';
      ctx.font = 'bold 12px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(node.label.toUpperCase(), pos.x, pos.y - radius - 5);

      // Collection type
      if (node.best_collection) {
        ctx.fillStyle = '#71717a';
        ctx.font = '10px monospace';
        ctx.fillText(`[${node.best_collection}]`, pos.x, pos.y + radius + 15);
      }

      // Success rate in node center
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 10px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${Math.round((node.success_rate || 0) * 100)}%`, pos.x, pos.y);

      // Usage count below
      if (node.usage_count) {
        ctx.fillStyle = '#71717a';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(`${node.usage_count}x`, pos.x, pos.y + 8);
      }
    });

    // Legend - positioned at bottom of canvas (v0.2.1: Triple KG system)
    const legendY = height - 50;

    // Semi-transparent background for legend
    ctx.fillStyle = 'rgba(24, 24, 27, 0.8)';
    ctx.fillRect(5, legendY - 5, 350, 45);

    // Title on its own line
    ctx.fillStyle = '#a1a1aa';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('Entity Source:', 10, legendY + 8);

    // Color boxes on the next line with more vertical space
    const boxY = legendY + 20;

    // Blue (routing)
    ctx.fillStyle = '#3b82f6';
    ctx.fillRect(10, boxY, 12, 12);
    ctx.fillStyle = '#a1a1aa';
    ctx.fillText('Query', 26, boxY + 9);

    // Green (content)
    ctx.fillStyle = '#22c55e';
    ctx.fillRect(80, boxY, 12, 12);
    ctx.fillStyle = '#a1a1aa';
    ctx.fillText('Memory', 96, boxY + 9);

    // Purple (both)
    ctx.fillStyle = '#a855f7';
    ctx.fillRect(165, boxY, 12, 12);
    ctx.fillStyle = '#a1a1aa';
    ctx.fillText('Both', 181, boxY + 9);

    // Orange (action effectiveness)
    ctx.fillStyle = '#f97316';
    ctx.fillRect(225, boxY, 12, 12);
    ctx.fillStyle = '#a1a1aa';
    ctx.fillText('Action', 241, boxY + 9);
  };

  useEffect(() => {
    let resizeTimeout: NodeJS.Timeout;

    const handleResize = (forceReinit = false) => {
      const canvas = canvasRef.current;
      const container = canvas?.parentElement;
      if (canvas && container) {
        const rect = container.getBoundingClientRect();
        const width = rect.width || 400;
        const height = rect.height || 300;

        // Use device pixel ratio for high-DPI displays
        const dpr = window.devicePixelRatio || 1;

        // Set display size (css pixels)
        canvas.style.width = width + 'px';
        canvas.style.height = height + 'px';

        // Set actual size in memory (scaled for DPI)
        canvas.width = width * dpr;
        canvas.height = height * dpr;

        // Scale context to match device pixel ratio
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.scale(dpr, dpr);
        }

        console.log('Canvas resized to:', width, 'x', height, 'DPR:', dpr);

        // Reinitialize positions on initial load or when forced
        if ((filteredData.nodes.length > 0 && nodesRef.current.size === 0) || forceReinit) {
          initializeNodePositions(filteredData.nodes);
        } else {
          draw();
        }
      }
    };

    const debouncedResize = () => {
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(() => handleResize(true), 300);
    };

    // Initial sizing with forced reinit
    handleResize(true);
    setTimeout(() => handleResize(true), 100);
    setTimeout(() => handleResize(true), 500);

    // Use ResizeObserver to catch panel dragging (not just window resize)
    const canvas = canvasRef.current;
    const container = canvas?.parentElement;
    let resizeObserver: ResizeObserver | null = null;

    if (container) {
      resizeObserver = new ResizeObserver(() => {
        debouncedResize();
      });
      resizeObserver.observe(container);
    }

    window.addEventListener('resize', debouncedResize);
    return () => {
      clearTimeout(resizeTimeout);
      window.removeEventListener('resize', debouncedResize);
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  }, [filteredData]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-zinc-500">Loading knowledge graph...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-red-500">{error}</div>
      </div>
    );
  }

  const hasNoData = !filteredData.nodes || filteredData.nodes.length === 0;

  const getSubtitle = () => {
    const sortText = sortBy === 'hybrid' ? 'most important' :
                     sortBy === 'recent' ? 'most recent' : 'oldest';
    const timeText = timeFilter === 'all' ? 'concepts' :
                     timeFilter === 'today' ? 'concepts from today' :
                     timeFilter === 'week' ? 'concepts from this week' :
                     'concepts from this session';
    return `Top 20 ${sortText} ${timeText} • Node size = usage × quality`;
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header - Improved hierarchy */}
      <div className="mb-3 flex-shrink-0">
        <h3 className="text-sm font-semibold text-zinc-300">Knowledge Graph</h3>
        <p className="text-xs text-zinc-500 mt-0.5">
          {getSubtitle()}
        </p>
      </div>

      {/* Filter Controls - Icon-based with hierarchy */}
      <div className="mb-4 flex gap-3 items-center flex-shrink-0">
        {/* Sort Dropdown - Primary control */}
        <div className="flex-[1.2] flex items-center gap-2">
          <svg className="w-3.5 h-3.5 text-zinc-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4h13M3 8h9m-9 4h6m4 0l4-4m0 0l4 4m-4-4v12" />
          </svg>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="flex-1 bg-zinc-800 text-zinc-200 text-xs px-3 py-2 rounded-lg border border-zinc-700 hover:border-zinc-600 focus:outline-none focus:border-cyan-600 transition-colors"
            title="Sort concepts by importance, recency, or age"
          >
            <option value="hybrid">Importance</option>
            <option value="recent">Recent</option>
            <option value="oldest">Oldest</option>
          </select>
        </div>

        {/* Time Filter Dropdown - Secondary control */}
        <div className="flex-1 flex items-center gap-2">
          <svg className="w-3.5 h-3.5 text-zinc-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <select
            value={timeFilter}
            onChange={(e) => setTimeFilter(e.target.value as any)}
            className="flex-1 bg-zinc-800 text-zinc-200 text-xs px-3 py-2 rounded-lg border border-zinc-700 hover:border-zinc-600 focus:outline-none focus:border-cyan-600 transition-colors"
            title="Filter by time period"
          >
            <option value="all">All Time</option>
            <option value="today">Today</option>
            <option value="week">This Week</option>
            <option value="session">This Session</option>
          </select>
        </div>
      </div>

      {/* Canvas Area */}
      <div className="flex-1 relative min-h-[300px]">
        {hasNoData ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center space-y-4">
            <svg className="w-16 h-16 text-zinc-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1}
                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <div className="text-center">
              <p className="text-sm text-zinc-500">No concepts found</p>
              <p className="text-xs text-zinc-600 mt-1">
                {timeFilter === 'session' ? 'Search memory and provide feedback ("that worked!", "thanks!") to build the graph' :
                 timeFilter === 'today' ? 'No activity today yet' :
                 timeFilter === 'week' ? 'No activity this week' :
                 'Search memory and provide feedback to create concepts'}
              </p>
            </div>
          </div>
        ) : (
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full"
          />
        )}
      </div>

      {/* Concepts Summary - Improved clarity */}
      {!hasNoData && (
        <div className="mt-4 p-3 bg-zinc-900 rounded-lg border border-zinc-800 flex-shrink-0">
          <h4 className="text-xs font-medium text-zinc-300 mb-2">
            Active Concepts ({filteredData.nodes?.length || 0}) <span className="text-zinc-500 font-normal">• Click to view details</span>
          </h4>
        <div className="max-h-32 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent">
          <div className="flex flex-wrap gap-1.5">
            {filteredData.nodes.map(node => (
              <button
                key={node.id}
                onClick={() => fetchConceptDefinition(node)}
                className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-md bg-zinc-800 text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100 transition-all cursor-pointer border border-zinc-700/50 hover:border-zinc-600"
                title={`Click to see routing info | Success: ${Math.round((node.success_rate || 0) * 100)}% | Used: ${node.usage_count || 0}x`}
              >
                <div className={`w-2 h-2 rounded-full flex-shrink-0 ${
                  node.success_rate && node.success_rate > 0.7 ? 'bg-green-500' :
                  node.success_rate && node.success_rate > 0.4 ? 'bg-amber-500' : 'bg-red-500'
                }`} />
                {node.label}
              </button>
            ))}
          </div>
        </div>
        </div>
      )}

      {/* Concept Definition Modal */}
      {selectedConcept && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedConcept(null)}
        >
          {/* Backdrop */}
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />

          {/* Modal */}
          <div
            className="relative bg-zinc-900 rounded-2xl border border-zinc-800 shadow-2xl p-6 max-w-md w-full max-h-[85vh] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close button */}
            <button
              onClick={() => setSelectedConcept(null)}
              className="absolute top-4 right-4 text-zinc-500 hover:text-zinc-300 transition-colors"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>

            {/* Header */}
            <div className="mb-4">
              <div className="flex items-center gap-3 mb-2">
                <div className={`w-4 h-4 rounded-full ${
                  selectedConcept.success_rate && selectedConcept.success_rate > 0.7 ? 'bg-green-500' :
                  selectedConcept.success_rate && selectedConcept.success_rate > 0.4 ? 'bg-amber-500' : 'bg-red-500'
                }`} />
                <h3 className="text-lg font-semibold text-zinc-100">
                  {selectedConcept.label}
                </h3>
              </div>

              {/* Stats */}
              <div className="flex gap-3 text-xs text-zinc-500">
                {selectedConcept.best_collection && (
                  <span className="flex items-center gap-1 px-2 py-1 bg-zinc-800 rounded-md">
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                    </svg>
                    {selectedConcept.best_collection}
                  </span>
                )}
                {selectedConcept.success_rate !== undefined && (
                  <span className="flex items-center gap-1 px-2 py-1 bg-zinc-800 rounded-md">
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    {Math.round(selectedConcept.success_rate * 100)}% success
                  </span>
                )}
                {selectedConcept.usage_count !== undefined && (
                  <span className="flex items-center gap-1 px-2 py-1 bg-zinc-800 rounded-md">
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    {selectedConcept.usage_count}x used
                  </span>
                )}
              </div>

              {/* Timestamp info */}
              {(selectedConcept.last_used || selectedConcept.created_at) && (
                <div className="text-xs text-zinc-500 mt-2">
                  {selectedConcept.last_used && (
                    <span>Last used: {new Date(selectedConcept.last_used).toLocaleString()}</span>
                  )}
                  {selectedConcept.last_used && selectedConcept.created_at && <span> • </span>}
                  {selectedConcept.created_at && (
                    <span>Created: {new Date(selectedConcept.created_at).toLocaleString()}</span>
                  )}
                </div>
              )}
            </div>

            {/* Concept Info - Scrollable - MINIMAL DESIGN */}
            <div className="space-y-3 overflow-y-auto flex-1 pr-2">

              {/* Loading State */}
              {loadingDefinition && (
                <div className="text-sm text-zinc-500 animate-pulse p-4 text-center">Loading concept data...</div>
              )}

              {!loadingDefinition && (
                <>
                  {/* Learned routing behavior header */}
                  <div className="text-sm text-zinc-400 mb-4 px-4 py-2 bg-zinc-800/30 rounded-lg">
                    <span className="text-zinc-200 font-semibold">Learned routing behavior</span> for queries containing "{selectedConcept.label}"
                  </div>

                  {/* Search Strategy */}
                  {selectedConcept.collections_breakdown && Object.keys(selectedConcept.collections_breakdown).length > 0 && (
                    <div className="p-4 bg-zinc-800/50 rounded-lg">
                      <h4 className="text-xs font-medium text-zinc-400 mb-3 flex items-center gap-1.5">
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                        </svg>
                        Collections Searched
                      </h4>
                      <div className="space-y-2">
                        {Object.entries(selectedConcept.collections_breakdown)
                          .sort(([, a], [, b]) => {
                            const totalA = a.successes + a.failures;
                            const totalB = b.successes + b.failures;
                            const rateA = totalA > 0 ? a.successes / totalA : 0;
                            const rateB = totalB > 0 ? b.successes / totalB : 0;
                            return rateB - rateA; // Sort by success rate descending
                          })
                          .map(([collection, data], idx) => {
                            const totalWithFeedback = data.successes + data.failures;
                            const successRate = totalWithFeedback > 0 ? Math.round((data.successes / totalWithFeedback) * 100) : 0;
                            const isBest = collection === selectedConcept.best_collection;
                            return (
                              <div key={collection} className="flex items-center justify-between text-xs">
                                <div className="flex items-center gap-2">
                                  <span className="text-zinc-300">{collection}</span>
                                  {isBest && <span className="text-yellow-400">⭐</span>}
                                </div>
                                <div className="text-zinc-500">
                                  {totalWithFeedback > 0 ? `${totalWithFeedback} with feedback, ${successRate}% success` : `${data.total} tries (no feedback yet)`}
                                </div>
                              </div>
                            );
                          })}
                      </div>
                    </div>
                  )}

                  {/* Track Record */}
                  {selectedConcept.outcome_breakdown && (
                    <div className="p-4 bg-zinc-800/50 rounded-lg">
                      <h4 className="text-xs font-medium text-zinc-400 mb-2 flex items-center gap-1.5">
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 4 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                        Track Record
                      </h4>
                      <div className="space-y-2">
                        <div className="flex items-center gap-4 text-sm">
                          <span className="text-green-400">✓ {selectedConcept.outcome_breakdown.worked} worked</span>
                          <span className="text-red-400">✗ {selectedConcept.outcome_breakdown.failed} failed</span>
                          {(selectedConcept.outcome_breakdown.worked + selectedConcept.outcome_breakdown.failed) > 0 && (
                            <span className="text-zinc-400">
                              → {Math.round((selectedConcept.outcome_breakdown.worked / (selectedConcept.outcome_breakdown.worked + selectedConcept.outcome_breakdown.failed)) * 100)}% success
                            </span>
                          )}
                        </div>
                        {selectedConcept.outcome_breakdown.partial > 0 && (
                          <div className="text-xs text-zinc-500">
                            Plus {selectedConcept.outcome_breakdown.partial} partial result{selectedConcept.outcome_breakdown.partial !== 1 ? 's' : ''} (still useful data, just not counted in success rate)
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Often appears with */}
                  {selectedConcept.related_concepts_with_stats && selectedConcept.related_concepts_with_stats.length > 0 && (
                    <div className="p-4 bg-zinc-800/50 rounded-lg">
                      <h4 className="text-xs font-medium text-zinc-400 mb-3 flex items-center gap-1.5">
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                        </svg>
                        Often appears with
                      </h4>
                      <div className="space-y-1.5">
                        {selectedConcept.related_concepts_with_stats
                          .sort((a, b) => b.success_rate - a.success_rate)
                          .slice(0, 5)
                          .map((rel, idx) => (
                            <div key={idx} className="flex items-center justify-between text-xs">
                              <span className="text-zinc-300">{rel.concept}</span>
                              <span className="text-zinc-500">
                                {Math.round(rel.success_rate * 100)}% success together
                              </span>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Footer - Fixed at bottom */}
            <div className="mt-4 pt-4 border-t border-zinc-800 flex justify-end flex-shrink-0">
              <button
                onClick={() => setSelectedConcept(null)}
                className="px-4 py-2 text-xs font-medium text-zinc-300 bg-zinc-800 hover:bg-zinc-700 rounded-lg transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default KnowledgeGraph;