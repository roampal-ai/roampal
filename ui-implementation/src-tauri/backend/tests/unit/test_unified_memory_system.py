"""
Unit Tests for UnifiedMemorySystem Facade

Tests the refactored facade that coordinates all services.
"""

import sys
from pathlib import Path
backend_dir = Path(__file__).parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from modules.memory.unified_memory_system import UnifiedMemorySystem
from modules.memory.config import MemoryConfig


class TestUnifiedMemorySystemInit:
    """Test initialization."""

    def test_init_creates_data_dir(self, tmp_path):
        """Should create data directory."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        assert (tmp_path / "data").exists()

    def test_init_with_custom_config(self, tmp_path):
        """Should use custom config."""
        config = MemoryConfig(promotion_score_threshold=0.8)
        ums = UnifiedMemorySystem(
            data_dir=str(tmp_path / "data"),
            config=config
        )
        assert ums.config.promotion_score_threshold == 0.8
        assert ums.PROMOTION_SCORE_THRESHOLD == 0.8

    def test_init_conversation_id(self, tmp_path):
        """Should generate conversation ID."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        assert ums.conversation_id is not None
        assert len(ums.conversation_id) > 0

    def test_init_not_initialized(self, tmp_path):
        """Should not be initialized until initialize() called."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        assert not ums.initialized


class TestInitialize:
    """Test initialization process."""

    @pytest.fixture
    def mock_adapter_factory(self):
        """Create mock adapter factory."""
        def factory(name):
            adapter = MagicMock()
            adapter.initialize = AsyncMock()
            adapter.list_all_ids = MagicMock(return_value=[])
            adapter.get_fragment = MagicMock(return_value=None)
            adapter.query_vectors = AsyncMock(return_value=[])
            adapter.upsert_vectors = AsyncMock()
            return adapter
        return factory

    @pytest.fixture
    def ums(self, tmp_path, mock_adapter_factory):
        """Create UMS with mocked adapters."""
        ums = UnifiedMemorySystem(
            data_dir=str(tmp_path / "data"),
            chromadb_adapter_factory=mock_adapter_factory
        )
        return ums

    @pytest.mark.asyncio
    async def test_initialize_creates_collections(self, ums):
        """Should create all collections."""
        await ums.initialize()

        assert "books" in ums.collections
        assert "working" in ums.collections
        assert "history" in ums.collections
        assert "patterns" in ums.collections
        assert "memory_bank" in ums.collections

    @pytest.mark.asyncio
    async def test_initialize_creates_services(self, ums):
        """Should initialize all services."""
        await ums.initialize()

        assert ums._scoring_service is not None
        assert ums._routing_service is not None
        assert ums._search_service is not None
        assert ums._promotion_service is not None
        assert ums._outcome_service is not None
        assert ums._memory_bank_service is not None
        assert ums._context_service is not None

    @pytest.mark.asyncio
    async def test_initialize_only_once(self, ums):
        """Should only initialize once."""
        await ums.initialize()
        first_search = ums._search_service

        await ums.initialize()
        assert ums._search_service is first_search


class TestStore:
    """Test store functionality."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with all mocks."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        ums.initialized = True

        # Mock collections
        working = MagicMock()
        working.upsert_vectors = AsyncMock()
        ums.collections = {"working": working}

        # Mock embedding
        ums._embedding_service = MagicMock()
        ums._embedding_service.embed_text = AsyncMock(return_value=[0.1] * 384)

        return ums

    @pytest.mark.asyncio
    async def test_store_generates_doc_id(self, mock_ums):
        """Should generate document ID."""
        doc_id = await mock_ums.store("test text")

        assert doc_id.startswith("working_")
        mock_ums.collections["working"].upsert_vectors.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_embeds_text(self, mock_ums):
        """Should embed the text."""
        await mock_ums.store("test text")

        mock_ums._embedding_service.embed_text.assert_called_with("test text")

    @pytest.mark.asyncio
    async def test_store_with_metadata(self, mock_ums):
        """Should include custom metadata."""
        await mock_ums.store(
            "test text",
            metadata={"custom": "value"}
        )

        call_args = mock_ums.collections["working"].upsert_vectors.call_args
        metadata = call_args[1]["metadatas"][0]

        assert metadata["custom"] == "value"
        assert metadata["text"] == "test text"


class TestSearch:
    """Test search functionality."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with search mock."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        ums.initialized = True

        # Mock search service
        ums._search_service = MagicMock()
        ums._search_service.search = AsyncMock(return_value=[
            {"content": "result1", "metadata": {}},
            {"content": "result2", "metadata": {}}
        ])

        return ums

    @pytest.mark.asyncio
    async def test_search_delegates_to_service(self, mock_ums):
        """Should delegate to search service."""
        results = await mock_ums.search("test query")

        mock_ums._search_service.search.assert_called_once()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_passes_collections(self, mock_ums):
        """Should pass collections to search service."""
        await mock_ums.search(
            "test query",
            collections=["patterns", "history"]
        )

        call_args = mock_ums._search_service.search.call_args
        assert call_args[1]["collections"] == ["patterns", "history"]


class TestRecordOutcome:
    """Test outcome recording."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with outcome mock."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        ums.initialized = True

        # Mock outcome service
        ums._outcome_service = MagicMock()
        ums._outcome_service.record_outcome = AsyncMock(return_value={"score": 0.7})

        return ums

    @pytest.mark.asyncio
    async def test_record_outcome_delegates(self, mock_ums):
        """Should delegate to outcome service."""
        await mock_ums.record_outcome(
            doc_id="working_123",
            outcome="worked"
        )

        mock_ums._outcome_service.record_outcome.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_outcome_with_reason(self, mock_ums):
        """Should pass failure reason."""
        await mock_ums.record_outcome(
            doc_id="working_123",
            outcome="failed",
            failure_reason="Test failure"
        )

        call_args = mock_ums._outcome_service.record_outcome.call_args
        assert call_args[1]["failure_reason"] == "Test failure"


class TestMemoryBankAPI:
    """Test memory bank API."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with memory bank mock."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        ums.initialized = True

        # Mock memory bank service
        ums._memory_bank_service = MagicMock()
        ums._memory_bank_service.store = AsyncMock(return_value="memory_bank_123")
        ums._memory_bank_service.update = AsyncMock(return_value="memory_bank_123")
        ums._memory_bank_service.archive = AsyncMock(return_value=True)
        ums._memory_bank_service.search = AsyncMock(return_value=[])
        ums._memory_bank_service.restore = AsyncMock(return_value=True)
        ums._memory_bank_service.delete = AsyncMock(return_value=True)

        return ums

    @pytest.mark.asyncio
    async def test_store_memory_bank(self, mock_ums):
        """Should delegate to memory bank service."""
        doc_id = await mock_ums.store_memory_bank(
            text="User prefers dark mode",
            tags=["preference"]
        )

        assert doc_id == "memory_bank_123"
        mock_ums._memory_bank_service.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_memory_bank(self, mock_ums):
        """Should delegate update."""
        await mock_ums.update_memory_bank(
            doc_id="memory_bank_123",
            new_text="Updated text"
        )

        mock_ums._memory_bank_service.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_archive_memory_bank(self, mock_ums):
        """Should delegate archive."""
        result = await mock_ums.archive_memory_bank("memory_bank_123")
        assert result is True

    @pytest.mark.asyncio
    async def test_search_memory_bank(self, mock_ums):
        """Should delegate search."""
        await mock_ums.search_memory_bank(query="test")
        mock_ums._memory_bank_service.search.assert_called_once()


class TestContextAPI:
    """Test context analysis API."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with context mock."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        ums.initialized = True

        # Mock context service
        ums._context_service = MagicMock()
        ums._context_service.analyze_conversation_context = AsyncMock(return_value={
            "relevant_patterns": [],
            "past_outcomes": [],
            "topic_continuity": [],
            "proactive_insights": []
        })

        return ums

    @pytest.mark.asyncio
    async def test_analyze_context(self, mock_ums):
        """Should delegate to context service."""
        context = await mock_ums.analyze_conversation_context(
            current_message="test",
            recent_conversation=[],
            conversation_id="conv123"
        )

        assert "relevant_patterns" in context
        mock_ums._context_service.analyze_conversation_context.assert_called_once()


class TestPromotionAPI:
    """Test promotion API."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with promotion mock."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        ums.initialized = True

        # Mock promotion service
        ums._promotion_service = MagicMock()
        ums._promotion_service.promote_valuable_working_memory = AsyncMock(return_value=5)
        ums._promotion_service.cleanup_old_working_memory = AsyncMock(return_value=3)

        return ums

    @pytest.mark.asyncio
    async def test_promote_valuable(self, mock_ums):
        """Should delegate to promotion service."""
        count = await mock_ums.promote_valuable_working_memory()

        assert count == 5
        mock_ums._promotion_service.promote_valuable_working_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_old(self, mock_ums):
        """Should delegate cleanup."""
        count = await mock_ums.cleanup_old_working_memory()

        assert count == 3


class TestSessionManagement:
    """Test session/conversation management."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with session mocks."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        ums.initialized = True

        # Mock promotion service
        ums._promotion_service = MagicMock()
        ums._promotion_service.promote_valuable_working_memory = AsyncMock(return_value=0)

        return ums

    @pytest.mark.asyncio
    async def test_switch_conversation(self, mock_ums):
        """Should switch conversation ID."""
        old_id = mock_ums.conversation_id

        # Pass explicit new ID to avoid timestamp collision
        new_id = await mock_ums.switch_conversation("new_conv_123")

        assert new_id == "new_conv_123"
        assert mock_ums.conversation_id == new_id
        assert mock_ums.message_count == 0

    @pytest.mark.asyncio
    async def test_switch_conversation_promotes(self, mock_ums):
        """Should promote valuable memories when switching."""
        await mock_ums.switch_conversation()

        mock_ums._promotion_service.promote_valuable_working_memory.assert_called_once()

    def test_increment_message_count(self, mock_ums):
        """Should increment message count."""
        initial = mock_ums.message_count

        mock_ums.increment_message_count()

        assert mock_ums.message_count == initial + 1


class TestCleanup:
    """Test cleanup functionality."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        """Create UMS with cleanup mocks."""
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        ums.initialized = True

        # Mock collections
        working = MagicMock()
        working.cleanup = AsyncMock()
        ums.collections = {"working": working}

        return ums

    @pytest.mark.asyncio
    async def test_cleanup_closes_collections(self, mock_ums):
        """Should cleanup collections."""
        await mock_ums.cleanup()

        mock_ums.collections["working"].cleanup.assert_called_once()


class TestV032LatencyFixes:
    """v0.3.2: Chat-path latency fixes."""

    @pytest.fixture
    def mock_ums(self, tmp_path):
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        ums.initialized = True
        ums._memory_bank_service = MagicMock()
        ums._memory_bank_service.get_always_inject = MagicMock(return_value=[])
        ums._format_context_injection = MagicMock(return_value="")
        return ums

    @pytest.mark.asyncio
    async def test_get_context_for_injection_parallelizes_lanes(self, mock_ums):
        """Summary and fact lanes must run concurrently, not sequentially."""
        import asyncio

        in_flight = 0
        max_in_flight = 0

        async def fake_search(*args, **kwargs):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.05)
            in_flight -= 1
            return []

        mock_ums.search = fake_search
        await mock_ums.get_context_for_injection(query="test")
        assert max_in_flight == 2, "Both retrieval lanes must be in flight simultaneously"

    @pytest.mark.asyncio
    async def test_onnx_models_warm_at_init(self, tmp_path):
        """ONNX warm-start task should fire during initialize()."""
        import asyncio

        def factory(name):
            adapter = MagicMock()
            adapter.initialize = AsyncMock()
            adapter.list_all_ids = MagicMock(return_value=[])
            adapter.get_fragment = MagicMock(return_value=None)
            adapter.query_vectors = AsyncMock(return_value=[])
            adapter.upsert_vectors = AsyncMock()
            return adapter

        ums = UnifiedMemorySystem(
            data_dir=str(tmp_path / "data"),
            chromadb_adapter_factory=factory,
        )

        with patch.object(UnifiedMemorySystem, "_warm_onnx_models", new_callable=AsyncMock) as warm:
            await ums.initialize()
            # Give the background task a tick to start
            await asyncio.sleep(0)
            warm.assert_called_once()


class TestV032WebsocketReadyPoll:
    """v0.3.2: agent_chat._run_generation_task ready-poll replaces flat sleep."""

    @pytest.mark.asyncio
    async def test_websocket_ready_poll_exits_early(self):
        """When WS is already populated, polling should make AT MOST a couple
        of short sleep calls (20 ms each) before exiting — never the ~25
        iterations a full 500 ms wait would take. Measure that directly
        instead of wall-clock (robust under CI load).
        """
        import asyncio

        from app.routers import agent_chat as agent_chat_module
        from app.routers.agent_chat import _run_generation_task

        class FakeAppState:
            def __init__(self):
                self.websockets = {}

        app_state = FakeAppState()
        conv_id = "conv-test-ready-poll"
        fake_ws = MagicMock()
        fake_ws.send_json = AsyncMock()
        fake_ws.client_state = "connected"
        app_state.websockets[conv_id] = fake_ws
        request = MagicMock()

        # Count the 20 ms polling sleeps. A flat 500 ms sleep would have
        # made 25 of these; a polling loop on a ready WS makes 0.
        short_sleeps = []
        real_sleep = asyncio.sleep

        async def tracking_sleep(duration):
            if duration == 0.02:
                short_sleeps.append(duration)
            await real_sleep(0)  # yield control but don't actually wait

        with patch.object(agent_chat_module, "agent_service", MagicMock()), \
             patch.object(asyncio, "sleep", side_effect=tracking_sleep):
            try:
                await asyncio.wait_for(
                    _run_generation_task(conv_id, request, "user-1", app_state),
                    timeout=5.0,  # plenty of slack; we're counting, not timing
                )
            except (asyncio.TimeoutError, Exception):
                pass

        # If the ready-poll works, it sees the WS on iteration 0 and breaks
        # before ever hitting the 20 ms sleep. Allow up to 2 to absorb
        # scheduler jitter on slow CI.
        assert len(short_sleeps) <= 2, (
            f"Expected ready-poll to exit early; saw {len(short_sleeps)} "
            f"20ms sleeps (a flat 500ms sleep would be 25)."
        )


class TestV032FactDedup:
    """v0.3.2: Desktop fact dedup — mirror the roampal-labs TagCascade /
    CE-lifecycle pattern. Skip storing a fact whose cosine distance to an
    existing fact is <0.1 (similarity >95%). Facts only; summaries still
    store normally.
    """

    @pytest.fixture
    def mock_ums(self, tmp_path):
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        ums.initialized = True

        # Stub all three tiers with query_vectors we can mock per-test.
        working = MagicMock()
        working.upsert_vectors = AsyncMock()
        working.query_vectors = AsyncMock(return_value=[])
        history = MagicMock()
        history.query_vectors = AsyncMock(return_value=[])
        patterns = MagicMock()
        patterns.query_vectors = AsyncMock(return_value=[])
        ums.collections = {"working": working, "history": history, "patterns": patterns}

        ums._embedding_service = MagicMock()
        ums._embedding_service.embed_text = AsyncMock(return_value=[0.1] * 768)
        return ums

    @pytest.mark.asyncio
    async def test_fact_stores_normally_when_no_duplicate(self, mock_ums):
        """Cold path: empty DB → fact lands in working as usual."""
        doc_id = await mock_ums.store(
            "The Eiffel Tower is in Paris.",
            collection="working",
            metadata={"memory_type": "fact"},
        )
        assert doc_id.startswith("working_")
        mock_ums.collections["working"].upsert_vectors.assert_called_once()

    @pytest.mark.asyncio
    async def test_near_duplicate_fact_is_skipped(self, mock_ums):
        """Duplicate fact (cosine distance <0.1) → return existing id, no write."""
        # Pretend working already has a near-duplicate at distance 0.05.
        mock_ums.collections["working"].query_vectors = AsyncMock(return_value=[
            {"id": "working_pre-existing", "distance": 0.05}
        ])

        doc_id = await mock_ums.store(
            "The Eiffel Tower is in Paris.",
            collection="working",
            metadata={"memory_type": "fact"},
        )
        # Returns the existing doc id, not a fresh one.
        assert doc_id == "working_pre-existing"
        mock_ums.collections["working"].upsert_vectors.assert_not_called()

    @pytest.mark.asyncio
    async def test_distant_fact_is_not_deduped(self, mock_ums):
        """Nearest neighbor at distance 0.4 → store normally (not similar enough)."""
        mock_ums.collections["working"].query_vectors = AsyncMock(return_value=[
            {"id": "working_unrelated", "distance": 0.4}
        ])

        doc_id = await mock_ums.store(
            "The Eiffel Tower is in Paris.",
            collection="working",
            metadata={"memory_type": "fact"},
        )
        assert doc_id.startswith("working_")
        assert doc_id != "working_unrelated"
        mock_ums.collections["working"].upsert_vectors.assert_called_once()

    @pytest.mark.asyncio
    async def test_summaries_are_not_deduped(self, mock_ums):
        """Summary-type memory bypasses the dedup check — always stores."""
        # Arrange a "duplicate" neighbor; the check should NOT fire for summaries.
        mock_ums.collections["working"].query_vectors = AsyncMock(return_value=[
            {"id": "working_existing_summary", "distance": 0.01}
        ])

        doc_id = await mock_ums.store(
            "Paris is a city in France with the Eiffel Tower.",
            collection="working",
            metadata={"memory_type": "summary"},  # not a fact
        )
        assert doc_id.startswith("working_")
        mock_ums.collections["working"].upsert_vectors.assert_called_once()

    @pytest.mark.asyncio
    async def test_dedup_checks_all_tiers(self, mock_ums):
        """Dup can live in history (or patterns), not just working — must still skip."""
        mock_ums.collections["working"].query_vectors = AsyncMock(return_value=[])
        mock_ums.collections["history"].query_vectors = AsyncMock(return_value=[
            {"id": "history_old_fact", "distance": 0.03}
        ])

        doc_id = await mock_ums.store(
            "The Eiffel Tower is in Paris.",
            collection="working",
            metadata={"memory_type": "fact"},
        )
        assert doc_id == "history_old_fact"
        mock_ums.collections["working"].upsert_vectors.assert_not_called()


class TestV032MemoryBankDedup:
    """v0.3.2: Expand fact dedup to cover memory_bank — the AI-facing
    identity/preference/learned-facts tier. Dupes there waste context-injection
    budget at retrieval time. Guards store_memory_bank the same way store()
    guards fact writes, and adds memory_bank to the tier scan so the two
    write surfaces can detect each other's existing rows.
    """

    @pytest.fixture
    def mock_ums(self, tmp_path):
        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        ums.initialized = True

        working = MagicMock()
        working.query_vectors = AsyncMock(return_value=[])
        history = MagicMock()
        history.query_vectors = AsyncMock(return_value=[])
        patterns = MagicMock()
        patterns.query_vectors = AsyncMock(return_value=[])
        memory_bank = MagicMock()
        memory_bank.query_vectors = AsyncMock(return_value=[])
        memory_bank.update_fragment_metadata = MagicMock()
        ums.collections = {
            "working": working,
            "history": history,
            "patterns": patterns,
            "memory_bank": memory_bank,
        }

        ums._embedding_service = MagicMock()
        ums._embedding_service.embed_text = AsyncMock(return_value=[0.1] * 768)

        # Mock the memory_bank_service — store returns a fresh id, update/etc. unused
        ums._memory_bank_service = MagicMock()
        ums._memory_bank_service.store = AsyncMock(return_value="memory_bank_new_id")

        # No tag service — bypass the noun_tags path, keep tests focused on dedup
        ums._tag_service = None
        return ums

    @pytest.mark.asyncio
    async def test_memory_bank_stores_normally_when_no_duplicate(self, mock_ums):
        """Cold path: no dup anywhere → write lands in memory_bank as normal."""
        doc_id = await mock_ums.store_memory_bank(
            text="User lives in Boston.", tags=["identity"]
        )
        assert doc_id == "memory_bank_new_id"
        mock_ums._memory_bank_service.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_bank_near_duplicate_within_tier_is_skipped(self, mock_ums):
        """Dup already in memory_bank → return existing id, don't re-store."""
        mock_ums.collections["memory_bank"].query_vectors = AsyncMock(return_value=[
            {"id": "memory_bank_existing", "distance": 0.05}
        ])
        doc_id = await mock_ums.store_memory_bank(
            text="User lives in Boston.", tags=["identity"]
        )
        assert doc_id == "memory_bank_existing"
        mock_ums._memory_bank_service.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_memory_bank_write_not_blocked_by_working_fact(self, mock_ums):
        """Asymmetric scope: memory_bank must NOT be blocked by an ephemeral
        working-tier copy. If it were, promoting a chat-extracted fact to
        permanent memory would be absorbed and silently lost when working
        ages out at 24h. The working row stays around as dead weight until it
        naturally rolls over; memory_bank's score-1.0 row wins retrieval."""
        # Pretend working has a high-similarity dup that WOULD match if scanned.
        mock_ums.collections["working"].query_vectors = AsyncMock(return_value=[
            {"id": "working_ephemeral_copy", "distance": 0.04}
        ])
        doc_id = await mock_ums.store_memory_bank(
            text="User lives in Boston.", tags=["identity"]
        )
        # Write lands; working-tier dup is ignored by scope.
        assert doc_id == "memory_bank_new_id"
        mock_ums._memory_bank_service.store.assert_called_once()
        # working.query_vectors must NOT have been called — memory_bank is
        # the only tier in the scan set for this surface.
        mock_ums.collections["working"].query_vectors.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_memory_bank_scans_only_memory_bank_tier(self, mock_ums):
        """Regression guard for the asymmetric scope: store_memory_bank
        must call query_vectors on memory_bank only, never on the other
        three tiers. If someone refactors _find_duplicate_fact and drops
        the `tiers` param, this test catches the leak."""
        await mock_ums.store_memory_bank(text="Something", tags=["identity"])
        assert mock_ums.collections["memory_bank"].query_vectors.await_count == 1
        mock_ums.collections["working"].query_vectors.assert_not_awaited()
        mock_ums.collections["history"].query_vectors.assert_not_awaited()
        mock_ums.collections["patterns"].query_vectors.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_working_fact_deduped_against_memory_bank(self, mock_ums):
        """Reverse cross-tier: memory_bank already has it → sidecar fact write skipped."""
        mock_ums.collections["working"].query_vectors = AsyncMock(return_value=[])
        mock_ums.collections["history"].query_vectors = AsyncMock(return_value=[])
        mock_ums.collections["patterns"].query_vectors = AsyncMock(return_value=[])
        mock_ums.collections["memory_bank"].query_vectors = AsyncMock(return_value=[
            {"id": "memory_bank_user_fact", "distance": 0.02}
        ])
        # Stub the working upsert so the happy-path branch has a target even though
        # we expect the dedup gate to fire before we get there.
        mock_ums.collections["working"].upsert_vectors = AsyncMock()

        doc_id = await mock_ums.store(
            "User lives in Boston.",
            collection="working",
            metadata={"memory_type": "fact"},
        )
        assert doc_id == "memory_bank_user_fact"
        mock_ums.collections["working"].upsert_vectors.assert_not_called()

    @pytest.mark.asyncio
    async def test_memory_bank_scan_uses_no_filter(self, mock_ums):
        """memory_bank rows don't carry memory_type — scan must pass filters=None
        or we'd miss every existing entry. Guards against a tempting but wrong
        unification of FACT_DEDUP_FILTERS."""
        await mock_ums.store_memory_bank(text="Something", tags=["identity"])
        memory_bank_call = mock_ums.collections["memory_bank"].query_vectors.await_args
        assert memory_bank_call is not None
        assert memory_bank_call.kwargs.get("filters") is None

    @pytest.mark.asyncio
    async def test_distant_memory_bank_fact_is_not_deduped(self, mock_ums):
        """Nearest neighbor at distance 0.4 → write proceeds normally."""
        mock_ums.collections["memory_bank"].query_vectors = AsyncMock(return_value=[
            {"id": "memory_bank_unrelated", "distance": 0.4}
        ])
        doc_id = await mock_ums.store_memory_bank(
            text="User lives in Boston.", tags=["identity"]
        )
        assert doc_id == "memory_bank_new_id"
        mock_ums._memory_bank_service.store.assert_called_once()


class TestV032Bug5MemoryBankTagAwait:
    """v0.3.2 (Bug 5): store_memory_bank auto-extracts noun_tags by calling
    the TagService. Previously called the sync .extract_tags(); when TagService
    had an async llm_extract_fn wired (the production path), the sync call
    couldn't await and silently returned []. Every AI-created memory_bank
    entry (via create_memory / add_to_memory_bank) landed without noun_tags.
    Fix routes through extract_tags_async() which handles async LLM fns.
    """

    @pytest.fixture
    def mock_ums(self, tmp_path):
        from modules.memory.tag_service import TagService

        ums = UnifiedMemorySystem(data_dir=str(tmp_path / "data"))
        ums.initialized = True

        memory_bank = MagicMock()
        memory_bank.query_vectors = AsyncMock(return_value=[])
        memory_bank.update_fragment_metadata = MagicMock()
        ums.collections = {"memory_bank": memory_bank}

        ums._embedding_service = MagicMock()
        ums._embedding_service.embed_text = AsyncMock(return_value=[0.1] * 768)

        ums._memory_bank_service = MagicMock()
        ums._memory_bank_service.store = AsyncMock(return_value="memory_bank_abc")

        # Real TagService with an ASYNC llm_extract_fn — this is the live
        # shape after main.py wires async_llm_tag_extractor at boot.
        async def async_llm(text: str):
            return ["calvin", "boston"]

        ums._tag_service = TagService(llm_extract_fn=async_llm)
        return ums

    @pytest.mark.asyncio
    async def test_memory_bank_auto_extraction_uses_async_path(self, mock_ums):
        """store_memory_bank with no explicit noun_tags should reach the
        async LLM extractor and write the returned tags onto the row."""
        await mock_ums.store_memory_bank(
            text="Calvin drove to Boston.", tags=["identity"]
        )

        # update_fragment_metadata called with the async-extracted tags
        assert mock_ums.collections["memory_bank"].update_fragment_metadata.called
        call = mock_ums.collections["memory_bank"].update_fragment_metadata.call_args
        meta_patch = call.args[1] if len(call.args) > 1 else call.kwargs.get("metadata", {})
        # meta_patch is a dict {"noun_tags": "<json string>"}
        import json as _json
        stored_tags = _json.loads(meta_patch["noun_tags"])
        assert "calvin" in stored_tags
        assert "boston" in stored_tags

    @pytest.mark.asyncio
    async def test_memory_bank_explicit_noun_tags_shortcircuits_extraction(self, mock_ums):
        """If caller passes noun_tags explicitly, extraction is skipped entirely."""
        await mock_ums.store_memory_bank(
            text="Calvin drove to Boston.",
            tags=["identity"],
            noun_tags=["explicit_override"],
        )
        call = mock_ums.collections["memory_bank"].update_fragment_metadata.call_args
        meta_patch = call.args[1] if len(call.args) > 1 else call.kwargs.get("metadata", {})
        import json as _json
        stored_tags = _json.loads(meta_patch["noun_tags"])
        assert stored_tags == ["explicit_override"]


class TestV032ChromaDBTelemetry:
    """v0.3.2 (0d): ChromaDB telemetry must be disabled at construction."""

    @pytest.mark.asyncio
    async def test_chromadb_persistent_client_has_telemetry_disabled(self, tmp_path):
        """Embedded PersistentClient must be constructed with anonymized_telemetry=False."""
        from modules.memory.chromadb_adapter import ChromaDBAdapter

        with patch("chromadb.PersistentClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_or_create_collection = MagicMock(return_value=MagicMock())
            mock_client_cls.return_value = mock_client

            adapter = ChromaDBAdapter(persistence_directory=str(tmp_path), use_server=False)
            await adapter.initialize(collection_name="roampal_test")

            assert mock_client_cls.called, "PersistentClient must be constructed"
            call_kwargs = mock_client_cls.call_args.kwargs
            settings_arg = call_kwargs.get("settings")
            assert settings_arg is not None, "settings= must be passed"
            assert settings_arg.anonymized_telemetry is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
