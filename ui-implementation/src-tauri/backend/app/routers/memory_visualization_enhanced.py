"""
Enhanced Memory Visualization Router
Shows the outcome-based memory system with collections
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Request, Query

logger = logging.getLogger(__name__)

router = APIRouter()


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

        # Get decay scheduler stats
        decay_stats = {}
        decay_scheduler = getattr(request.app.state, 'decay_scheduler', None)
        if decay_scheduler:
            decay_stats = decay_scheduler.get_stats()

        return {
            'collections': stats,
            'outcomes': outcome_stats,
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
                'uses': metadata.get('uses', 0),  # v0.3.0: Flatten uses for frontend
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


