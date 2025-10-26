import React, { useEffect, useState, useRef } from 'react';
import { apiFetch } from '../utils/fetch';

interface GraphNode {
  id: string;
  label: string;
  type: string;
  best_collection?: string;
  success_rate?: number;
  usage_count?: number;
  hybridScore?: number; // Calculated: √usage × √(quality + 0.1)
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
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const nodesRef = useRef<Map<string, { x: number; y: number; vx: number; vy: number }>>(new Map());

  useEffect(() => {
    fetchGraphData();
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  const fetchGraphData = async () => {
    try {
      const response = await apiFetch('http://localhost:8000/api/memory/knowledge-graph');
      if (!response.ok) throw new Error('Failed to fetch graph data');
      const data = await response.json();

      // Calculate hybrid score for each node: √usage × √(quality + 0.1)
      // Formula rewards both high usage AND high success rate
      const nodesWithScore = (data.nodes || []).map((node: any) => ({
        ...node,
        hybridScore: Math.sqrt(node.usage_count || 0) * Math.sqrt((node.success_rate || 0) + 0.1)
      }));

      // Sort by hybrid score (best first), take top N
      const topNodes = nodesWithScore
        .sort((a: any, b: any) => b.hybridScore - a.hybridScore)
        .slice(0, MAX_NODES_DISPLAYED);

      // Filter edges to only show connections between displayed nodes
      const topNodeIds = new Set(topNodes.map((n: any) => n.id));
      const filteredEdges = (data.edges || []).filter((edge: any) =>
        topNodeIds.has(edge.source) && topNodeIds.has(edge.target)
      );

      const graphData = {
        nodes: topNodes,
        edges: filteredEdges
      };
      setGraphData(graphData);
      setFilteredData(graphData);
      setLoading(false);
      if (graphData.nodes && graphData.nodes.length > 0) {
        initializeNodePositions(graphData.nodes);
      }
    } catch (err) {
      setError('Failed to load knowledge graph');
      setLoading(false);
    }
  };

  // Apply search filter
  useEffect(() => {
    if (!searchQuery) {
      setFilteredData(graphData);
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
      const response = await apiFetch(`http://localhost:8000/api/memory/knowledge-graph/concept/${concept.id}/definition`);
      if (response.ok) {
        const data = await response.json();

        // Check if the definition is the error message
        if (data.definition && data.definition !== "Unable to retrieve definition") {
          setConceptDefinition(data.definition);
        } else {
          // No definition available from KG
          setConceptDefinition(formatConceptInfo(concept));
        }

        // Update selected concept with additional data
        setSelectedConcept({
          ...concept,
          related_concepts: data.related_concepts || [],
          sources: data.sources || [],
          metadata: data.metadata || {}
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

      // Color based on success rate
      const successRate = node.success_rate || 0;
      if (successRate > 0.7) {
        ctx.fillStyle = '#22c55e'; // Green
      } else if (successRate > 0.4) {
        ctx.fillStyle = '#f59e0b'; // Orange
      } else {
        ctx.fillStyle = '#ef4444'; // Red
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

    // Legend - positioned at bottom of canvas
    const legendY = height - 50;

    // Semi-transparent background for legend
    ctx.fillStyle = 'rgba(24, 24, 27, 0.8)';
    ctx.fillRect(5, legendY - 5, 200, 45);

    // Title on its own line
    ctx.fillStyle = '#a1a1aa';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('Success Rate:', 10, legendY + 8);

    // Color boxes on the next line with more vertical space
    const boxY = legendY + 20;

    // Green
    ctx.fillStyle = '#22c55e';
    ctx.fillRect(10, boxY, 12, 12);
    ctx.fillStyle = '#a1a1aa';
    ctx.fillText('>70%', 26, boxY + 9);

    // Orange
    ctx.fillStyle = '#f59e0b';
    ctx.fillRect(65, boxY, 12, 12);
    ctx.fillStyle = '#a1a1aa';
    ctx.fillText('40-70%', 81, boxY + 9);

    // Red
    ctx.fillStyle = '#ef4444';
    ctx.fillRect(130, boxY, 12, 12);
    ctx.fillStyle = '#a1a1aa';
    ctx.fillText('<40%', 146, boxY + 9);
  };

  useEffect(() => {
    const handleResize = () => {
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

        // Reinitialize node positions if needed
        if (nodesRef.current.size === 0 && filteredData.nodes.length > 0) {
          initializeNodePositions(filteredData.nodes);
        } else {
          draw();
        }
      }
    };

    // Multiple attempts to ensure proper sizing
    handleResize();
    setTimeout(handleResize, 100);
    setTimeout(handleResize, 500);

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
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

  if (!filteredData.nodes || filteredData.nodes.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-4">
        <svg className="w-16 h-16 text-zinc-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1}
            d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
        <div className="text-center">
          <p className="text-sm text-zinc-500">No knowledge graph data yet</p>
          <p className="text-xs text-zinc-600 mt-1">Concepts will appear as the system learns</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="mb-2 flex-shrink-0">
        <h3 className="text-xs font-medium text-zinc-500">Knowledge Graph - Top 20 Patterns</h3>
        <p className="text-xs text-zinc-600">
          Larger nodes = better quality + more usage
        </p>
      </div>
      <div className="flex-1 relative min-h-[300px]">
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
          style={{ cursor: 'grab' }}
        />
      </div>

      {/* Concepts Summary */}
      <div className="mt-3 p-3 bg-zinc-900 rounded-lg border border-zinc-800 flex-shrink-0">
        <h4 className="text-xs font-medium text-zinc-400 mb-2">
          Active Concepts ({filteredData.nodes?.length || 0})
        </h4>
        <div className="max-h-32 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent">
          <div className="flex flex-wrap gap-1">
            {filteredData.nodes.map(node => (
              <button
                key={node.id}
                onClick={() => fetchConceptDefinition(node)}
                className="flex items-center gap-1 px-2 py-1 text-xs rounded-md bg-zinc-800 text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100 transition-all cursor-pointer"
                title={`Click to see routing info | Success: ${Math.round((node.success_rate || 0) * 100)}% | Used: ${node.usage_count || 0}x`}
              >
                <div className={`w-2 h-2 rounded-full ${
                  node.success_rate && node.success_rate > 0.7 ? 'bg-green-500' :
                  node.success_rate && node.success_rate > 0.4 ? 'bg-amber-500' : 'bg-red-500'
                }`} />
                {node.label}
              </button>
            ))}
          </div>
        </div>
      </div>

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
            className="relative bg-zinc-900 rounded-2xl border border-zinc-800 shadow-2xl p-6 max-w-md w-full"
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
            </div>

            {/* Concept Info */}
            <div className="space-y-3">
              <div className="p-4 bg-zinc-800/50 rounded-lg">
                <h4 className="text-xs font-medium text-zinc-400 mb-2">Memory Routing Info</h4>
                {loadingDefinition ? (
                  <div className="text-sm text-zinc-500 animate-pulse">Loading concept info...</div>
                ) : (
                  <p className="text-sm text-zinc-300 leading-relaxed whitespace-pre-line">
                    {conceptDefinition || 'Fetching concept info...'}
                  </p>
                )}
              </div>

              {/* Related Concepts */}
              {selectedConcept.related_concepts && selectedConcept.related_concepts.length > 0 && (
                <div className="p-4 bg-zinc-800/50 rounded-lg">
                  <h4 className="text-xs font-medium text-zinc-400 mb-2">Related Concepts</h4>
                  <div className="flex flex-wrap gap-1">
                    {selectedConcept.related_concepts.map((concept, idx) => (
                      <span key={idx} className="px-2 py-1 text-xs bg-zinc-700 rounded-md text-zinc-300">
                        {concept}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="mt-6 flex justify-end">
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