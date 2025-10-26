"""
Enhanced Memory Visualization Router
Shows the new outcome-based memory system with collections and KG
"""

import logging
import json
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Request, Query
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

# Add a simpler endpoint that the UI expects
@router.get("/knowledge-graph")
async def get_knowledge_graph(request: Request):
    """Simple endpoint for UI compatibility - returns graph data"""
    return await get_kg_concepts(request)


@router.get("/stats")
async def get_memory_stats(request: Request):
    """Get statistics for all memory collections"""
    try:
        # Get memory collections
        memory_collections = getattr(request.app.state, 'memory_collections', None)
        if not memory_collections:
            return {'error': 'Memory collections not initialized'}

        # Get collection statistics (using the correct method name)
        stats = memory_collections.get_stats() if hasattr(memory_collections, 'get_stats') else {}

        # Get outcome tracker stats
        outcome_tracker = getattr(request.app.state, 'outcome_tracker', None)
        outcome_stats = {}
        if outcome_tracker:
            try:
                patterns = await outcome_tracker.get_best_patterns(min_attempts=1, min_success_rate=0.0)
                outcome_stats = {
                    'total_patterns': len(patterns),
                    'successful_patterns': len([p for p in patterns if p.get('success_rate', 0) > 0.7]),
                    'failed_patterns': len([p for p in patterns if p.get('success_rate', 0) < 0.3])
                }
            except Exception as e:
                logger.error(f"Error getting outcome stats: {e}")

        # Get KG stats
        kg_stats = {}
        kg_router = getattr(request.app.state, 'kg_router', None)
        if kg_router:
            kg_stats = kg_router.get_graph_summary()

        # Get decay scheduler stats
        decay_stats = {}
        decay_scheduler = getattr(request.app.state, 'decay_scheduler', None)
        if decay_scheduler:
            decay_stats = decay_scheduler.get_stats()

        return {
            'collections': stats,
            'outcomes': outcome_stats,
            'knowledge_graph': kg_stats,
            'decay': decay_stats,
            'status': 'active'
        }

    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        return {
            'error': str(e),
            'status': 'error'
        }


@router.get("/collections/{collection_type}")
@router.get("/enhanced/collections/{collection_type}")  # Alias for UI compatibility
async def get_collection_memories(
    request: Request,
    collection_type: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """Get memories from a specific collection"""
    try:
        memory_collections = getattr(request.app.state, 'memory_collections', None)
        if not memory_collections:
            return {'memories': [], 'total': 0, 'error': 'Memory collections not initialized'}

        # Map conversations to history for backward compatibility
        actual_collection = "history" if collection_type == "conversations" else collection_type

        # For working collection, get ALL items then sort and paginate
        # This ensures we show the most recent items first
        if actual_collection == "working":
            # Get ALL items from the collection to sort properly
            all_results = await memory_collections.search(
                query="",  # Get all
                collections=[actual_collection],
                limit=1000,  # Get many items to sort
                offset=0,
                return_metadata=True
            )

            # Handle both dict and list return types
            if isinstance(all_results, dict):
                all_items = all_results.get('results', [])
                total_count = all_results.get('total', len(all_items))
            else:
                all_items = all_results if isinstance(all_results, list) else []
                total_count = len(all_items)

            # Sort ALL items by timestamp (most recent first)
            all_items.sort(
                key=lambda item: item.get('metadata', {}).get('timestamp', ''),
                reverse=True
            )

            # Now apply pagination to the sorted results
            start_idx = offset
            end_idx = offset + limit
            collection_results = all_items[start_idx:end_idx]
        else:
            # For other collections, use the normal search
            results = await memory_collections.search(
                query="",  # Get all
                collections=[actual_collection],
                limit=limit,
                offset=offset,
                return_metadata=True
            )

            # Handle both dict and list return types
            if isinstance(results, dict):
                collection_results = results.get('results', [])
                total_count = results.get('total', len(collection_results))
            else:
                collection_results = results if isinstance(results, list) else []
                total_count = len(collection_results)

        memories = []
        for item in collection_results:
            metadata = item.get('metadata', {})
            memory = {
                'id': item.get('id', item.get('doc_id')),
                'content': item.get('content', item.get('text', '')),
                'metadata': metadata,
                'score': metadata.get('score', item.get('score', 0.5)),
                'collection': collection_type,
                'timestamp': metadata.get('timestamp', metadata.get('upload_timestamp'))  # Flatten timestamp for frontend
            }

            # Add outcome data if available
            if collection_type == "patterns":
                success_rate = metadata.get('success_rate', 0)
                attempts = metadata.get('attempts', 0)
                memory['outcome'] = {
                    'success_rate': success_rate,
                    'attempts': attempts,
                    'status': 'successful' if success_rate > 0.7 else 'learning'
                }

            memories.append(memory)

        return {
            'memories': memories,
            'total': total_count,
            'collection': collection_type,
            'offset': offset,
            'limit': limit
        }

    except Exception as e:
        logger.error(f"Error getting collection memories: {e}")
        return {'memories': [], 'total': 0, 'error': str(e)}


@router.get("/knowledge-graph/concepts")
async def get_kg_concepts(request: Request):
    """
    Get concept graph data for visualization

    NOTE: Concept-to-concept relationships are derived from problem_categories.
    The system tracks routing patterns and problem groupings, but not explicit
    concept co-occurrence. Edges are inferred from concepts appearing in the
    same problem category.
    """
    try:
        # Try to get UnifiedMemorySystem first
        memory = getattr(request.app.state, 'memory', None)
        if memory and hasattr(memory, 'knowledge_graph'):
            # Use the knowledge graph from UnifiedMemorySystem
            graph = memory.knowledge_graph
        else:
            # Fallback to kg_router for backward compatibility
            kg_router = getattr(request.app.state, 'kg_router', None)
            if not kg_router:
                return {'concepts': [], 'relationships': []}

            # Load the knowledge graph
            if kg_router.graph_path.exists():
                with open(kg_router.graph_path, 'r') as f:
                    graph = json.load(f)
            else:
                graph = {'concepts': {}, 'relationships': {}, 'success_patterns': {}}

        # Format for visualization
        nodes = []
        edges = []

        # Add concept nodes from routing_patterns (UnifiedMemorySystem format)
        for concept, pattern in graph.get('routing_patterns', {}).items():
            nodes.append({
                'id': concept,
                'label': concept,
                'type': 'concept',
                'best_collection': pattern.get('best_collection'),
                'success_rate': pattern.get('success_rate', 0),
                'usage_count': sum(
                    coll_data.get('total', 0)
                    for coll_data in pattern.get('collections_used', {}).values()
                )
            })

        # Also add from success_patterns if present (backward compatibility)
        for concept, pattern in graph.get('success_patterns', {}).items():
            if concept not in [n['id'] for n in nodes]:  # Don't duplicate
                nodes.append({
                    'id': concept,
                    'label': concept,
                    'type': 'concept',
                    'best_collection': pattern.get('best_collection'),
                    'success_rate': pattern.get('success_rate', 0),
                    'usage_count': pattern.get('usage_count', 0)
                })

        # Use actual tracked concept relationships from the knowledge graph
        for rel_key, rel_data in graph.get('relationships', {}).items():
            # Relationship key format: "concept1|concept2" (sorted alphabetically)
            concepts = rel_key.split('|')
            if len(concepts) == 2:
                concept_a, concept_b = concepts

                # Check if both concepts exist as nodes
                if concept_a in [n['id'] for n in nodes] and concept_b in [n['id'] for n in nodes]:
                    # Calculate success rate from tracked data
                    co_occur = rel_data.get('co_occurrence', 0)
                    success = rel_data.get('success_together', 0)
                    failure = rel_data.get('failure_together', 0)
                    total_outcomes = success + failure
                    success_rate = (success / total_outcomes) if total_outcomes > 0 else 0.5

                    edges.append({
                        'source': concept_a,
                        'target': concept_b,
                        'weight': co_occur,  # Co-occurrence count
                        'success_rate': success_rate  # Actual tracked success rate
                    })

        return {
            'nodes': nodes,
            'edges': edges,
            'total_concepts': len(nodes),
            'total_relationships': len(edges)
        }

    except Exception as e:
        logger.error(f"Error getting KG concepts: {e}")
        return {'nodes': [], 'edges': [], 'error': str(e)}


@router.get("/knowledge-graph/data")
async def get_kg_data(request: Request):
    """Get raw knowledge graph data"""
    try:
        memory = getattr(request.app.state, 'memory', None)
        if memory and hasattr(memory, 'knowledge_graph'):
            kg = memory.knowledge_graph

            # Calculate summary stats
            total_concepts = len(kg.get('routing_patterns', {}))
            total_solutions = len(kg.get('problem_solutions', {}))

            return {
                'routing_patterns': kg.get('routing_patterns', {}),
                'problem_solutions': kg.get('problem_solutions', {}),
                'solution_patterns': kg.get('solution_patterns', {}),
                'failure_patterns': kg.get('failure_patterns', {}),
                'stats': {
                    'total_concepts': total_concepts,
                    'total_solutions': total_solutions,
                    'total_failure_patterns': len(kg.get('failure_patterns', {}))
                }
            }
        return {'error': 'Knowledge graph not available'}
    except Exception as e:
        logger.error(f"Error getting KG data: {e}")
        return {'error': str(e)}


@router.get("/knowledge-graph/concept/{concept_id}/definition")
async def get_concept_definition(
    request: Request,
    concept_id: str
):
    """Get definition and details for a specific concept"""
    try:
        memory_collections = getattr(request.app.state, 'memory_collections', None)
        if not memory_collections:
            return {'definition': f'{concept_id} is a tracked concept in the system.', 'related_concepts': []}

        # URL decode the concept ID (in case it has spaces or special chars)
        from urllib.parse import unquote
        concept_id = unquote(concept_id)

        # Search for the concept in patterns and conversations
        results = await memory_collections.search(
            query=concept_id,
            collections=["patterns", "conversations", "books", "working"],
            limit=10,
            return_metadata=True  # Get dictionary format with metadata
        )

        # Extract definition from search results
        definition = None
        related_concepts = []
        best_content = None
        best_score = 0

        # Handle the results - they come as a dict with 'results' key
        items_list = results.get("results", []) if isinstance(results, dict) else results
        for item in items_list:
            content = item.get("content", "")
            score = item.get("score", 0)

            # Keep track of the best matching content
            if score > best_score:
                best_score = score
                best_content = content

            # Extract related concepts from metadata
            metadata = item.get("metadata", {})
            if "concepts" in metadata:
                for concept in metadata["concepts"]:
                    if concept != concept_id and concept not in related_concepts:
                        related_concepts.append(concept)

        # Try to create a meaningful definition
        if best_content:
            # Clean and format the content as a definition
            content_lower = best_content.lower()
            concept_lower = concept_id.lower()

            # Look for sentences containing the concept
            sentences = best_content.replace('\n', '. ').split('.')
            relevant_sentences = []

            for sentence in sentences:
                if concept_lower in sentence.lower():
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 20:  # Ensure it's meaningful
                        relevant_sentences.append(sentence)

            if relevant_sentences:
                # Take the most descriptive sentences
                definition = '. '.join(relevant_sentences[:2])
                if not definition.endswith('.'):
                    definition += '.'
            else:
                # If no direct sentences, use the best content snippet
                if len(best_content) > 200:
                    definition = best_content[:200] + "..."
                else:
                    definition = best_content

        if not definition or len(definition) < 30:
            # Generate a more contextual definition based on the concept name
            concept_words = concept_id.lower().replace('_', ' ').replace('-', ' ')

            # Common programming/technical concepts
            if 'error' in concept_words or 'exception' in concept_words:
                definition = f"{concept_id} refers to an error condition or exception that may occur during program execution."
            elif 'fix' in concept_words or 'patch' in concept_words:
                definition = f"{concept_id} represents a solution or correction applied to resolve an issue or bug in the code."
            elif 'test' in concept_words:
                definition = f"{concept_id} relates to testing procedures or test cases used to verify code functionality."
            elif 'api' in concept_words:
                definition = f"{concept_id} pertains to API interactions, endpoints, or integration patterns."
            elif 'data' in concept_words or 'database' in concept_words:
                definition = f"{concept_id} involves data management, storage, or database operations."
            elif 'function' in concept_words or 'method' in concept_words:
                definition = f"{concept_id} represents a function or method implementation in the codebase."
            elif 'config' in concept_words or 'setting' in concept_words:
                definition = f"{concept_id} relates to configuration settings or environment parameters."
            elif 'model' in concept_words:
                definition = f"{concept_id} represents a data model or machine learning model used in the system."
            elif 'memory' in concept_words:
                definition = f"{concept_id} pertains to memory management, storage, or retrieval operations."
            elif 'pattern' in concept_words:
                definition = f"{concept_id} represents a design pattern or recurring solution approach."
            else:
                # Generic but more informative fallback
                definition = f"{concept_id} is a tracked concept representing a specific pattern, technique, or component that has been identified through system usage and learning."

        return {
            "concept": concept_id,
            "definition": definition,
            "related_concepts": related_concepts[:5]  # Limit to 5 related concepts
        }

    except Exception as e:
        logger.error(f"Error getting concept definition: {e}")
        return {'definition': 'Unable to retrieve definition', 'related_concepts': [], 'error': str(e)}


@router.get("/patterns/performance")
async def get_pattern_performance(request: Request):
    """Get pattern performance metrics"""
    try:
        outcome_tracker = getattr(request.app.state, 'outcome_tracker', None)
        if not outcome_tracker:
            return {'patterns': []}

        patterns = await outcome_tracker.get_best_patterns(min_attempts=1, min_success_rate=0.0)

        # Sort by success rate
        patterns.sort(key=lambda x: x.get('success_rate', 0), reverse=True)

        # Format for UI
        formatted = []
        for pattern in patterns[:20]:  # Top 20
            formatted.append({
                'problem': pattern.get('problem', 'Unknown'),
                'solution': pattern.get('solution', 'Unknown'),
                'success_rate': pattern.get('success_rate', 0),
                'attempts': pattern.get('attempts', 0),
                'last_used': pattern.get('last_used', 'Never'),
                'status': 'top_performer' if pattern.get('success_rate', 0) > 0.8 else 'normal'
            })

        return {
            'patterns': formatted,
            'total': len(patterns)
        }

    except Exception as e:
        logger.error(f"Error getting pattern performance: {e}")
        return {'patterns': [], 'error': str(e)}


@router.get("/decay/schedule")
async def get_decay_schedule(request: Request):
    """Get decay scheduler information"""
    try:
        decay_scheduler = getattr(request.app.state, 'decay_scheduler', None)
        if not decay_scheduler:
            return {'status': 'not_initialized'}

        stats = decay_scheduler.get_stats()

        return {
            'status': 'running' if stats['running'] else 'stopped',
            'last_run': stats.get('last_run', 'Never'),
            'next_run': stats.get('next_run', 'Unknown'),
            'config': {
                'conversation_ttl_days': stats['config']['conversation_ttl_days'],
                'working_memory_ttl_hours': stats['config']['working_memory_ttl_hours'],
                'pattern_failure_threshold': stats['config']['pattern_failure_threshold'],
                'check_interval_hours': stats['config']['decay_check_interval_hours']
            }
        }

    except Exception as e:
        logger.error(f"Error getting decay schedule: {e}")
        return {'status': 'error', 'error': str(e)}


@router.post("/decay/force")
async def force_decay(request: Request, collection_type: Optional[str] = None):
    """Force immediate decay for testing"""
    try:
        decay_scheduler = getattr(request.app.state, 'decay_scheduler', None)
        if not decay_scheduler:
            return {'status': 'error', 'message': 'Decay scheduler not initialized'}

        await decay_scheduler.force_cleanup(collection_type)

        return {
            'status': 'success',
            'message': f"Forced decay for {collection_type or 'all collections'}"
        }

    except Exception as e:
        logger.error(f"Error forcing decay: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/search")
async def search_memories(
    request: Request,
    query: str,
    collections: Optional[str] = None  # Comma-separated list
):
    """Search across memory collections"""
    try:
        memory_collections = getattr(request.app.state, 'memory_collections', None)
        if not memory_collections:
            return {'results': [], 'error': 'Memory collections not initialized'}

        # Parse collection types
        collection_list = collections.split(',') if collections else None

        # Search
        results = await memory_collections.search(
            query=query,
            collections=collection_list,
            limit=20,
            return_metadata=True
        )

        # Format results - handle both dict and list return types
        formatted_results = []
        if isinstance(results, dict):
            # New format with metadata
            for item in results.get('results', []):
                # Use stored score if available, otherwise convert distance to relevance
                stored_score = item.get('metadata', {}).get('score', None)
                if stored_score is not None:
                    # Use the actual stored confidence score
                    relevance_score = stored_score
                else:
                    # Convert distance to relevance: smaller distance = higher relevance
                    # Use inverse distance formula to keep scores in 0-1 range
                    distance = item.get('distance', 1.0)
                    relevance_score = 1.0 / (1.0 + distance)

                formatted_results.append({
                    'content': item.get('content', item.get('text', '')),
                    'collection': item.get('collection', item.get('collection_type', 'unknown')),
                    'score': relevance_score,
                    'metadata': item.get('metadata', {})
                })
        elif isinstance(results, list):
            # Old format (list of dicts)
            for item in results:
                if isinstance(item, dict):
                    # Use stored score if available, otherwise convert distance to relevance
                    stored_score = item.get('metadata', {}).get('score', None)
                    if stored_score is not None:
                        relevance_score = stored_score
                    else:
                        distance = item.get('distance', 1.0)
                        relevance_score = 1.0 / (1.0 + distance)

                    formatted_results.append({
                        'content': item.get('content', item.get('text', '')),
                        'collection': item.get('collection', item.get('collection_type', 'unknown')),
                        'score': relevance_score,
                        'metadata': item.get('metadata', {})
                    })
                else:
                    # Fallback for unexpected format
                    logger.warning(f"Unexpected item type in search results: {type(item)}")
                    continue

        # Sort by score
        formatted_results.sort(key=lambda x: x['score'], reverse=True)

        return {
            'results': formatted_results,
            'query': query,
            'total': len(formatted_results)
        }

    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        return {'results': [], 'error': str(e)}


@router.post("/feedback")
async def record_memory_feedback(
    request: Request,
    doc_id: str = Query(..., description="Memory document ID"),
    outcome: str = Query(..., description="Outcome: 'worked', 'failed', 'partial', or 'unknown'"),
    confidence: float = Query(0.8, description="Confidence score (0.0-1.0)"),
    context: Optional[str] = Query(None, description="Additional context")
):
    """
    Record explicit user feedback on a memory's usefulness.
    This helps the system learn which memories are valuable.
    """
    try:
        memory = getattr(request.app.state, 'memory', None) or getattr(request.app.state, 'memory_collections', None)
        if not memory:
            return {'status': 'error', 'message': 'Memory system not initialized'}

        # Validate outcome
        valid_outcomes = ['worked', 'failed', 'partial', 'unknown']
        if outcome not in valid_outcomes:
            return {
                'status': 'error',
                'message': f"Invalid outcome. Must be one of: {', '.join(valid_outcomes)}"
            }

        # Validate confidence
        if not 0.0 <= confidence <= 1.0:
            return {'status': 'error', 'message': 'Confidence must be between 0.0 and 1.0'}

        # Record the outcome
        await memory.record_outcome(
            doc_id=doc_id,
            outcome=outcome,
            context={
                "confidence": confidence,
                "user_feedback": True,
                "additional_context": context
            }
        )

        logger.info(f"Recorded feedback for {doc_id}: {outcome} (confidence: {confidence})")

        return {
            'status': 'success',
            'doc_id': doc_id,
            'outcome': outcome,
            'confidence': confidence,
            'message': 'Feedback recorded successfully'
        }

    except Exception as e:
        logger.error(f"Error recording feedback: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}
