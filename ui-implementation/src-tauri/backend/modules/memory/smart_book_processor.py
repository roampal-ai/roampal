"""
Smart Book Processor for Roampal
Processes uploaded documents and stores them in the books collection
"""
import asyncio
import json
import logging
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import aiosqlite
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class SmartBookProcessor:
    """
    Processes books by:
    1. Chunking text into manageable pieces
    2. Generating embeddings
    3. Storing in SQLite (metadata + full-text search)
    4. Storing embeddings in ChromaDB (books collection)
    """

    def __init__(
        self,
        data_dir: str,
        chromadb_adapter=None,
        embedding_service=None
    ):
        """
        Initialize book processor

        Args:
            data_dir: Base directory for book storage (e.g., "data/books")
            chromadb_adapter: ChromaDB adapter for books collection
            embedding_service: Service for generating embeddings
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.data_dir / "books.db"
        self.chromadb_adapter = chromadb_adapter
        self.embedding_service = embedding_service

        # For progress tracking
        self.task_manager = None

        # Token-based encoding for consistent chunk sizes
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Semantic-aware text splitter with token-based sizing
        # Multi-language support: includes CJK, Arabic, and Latin punctuation
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # tokens (~600 words, good for LLM context)
            chunk_overlap=150,  # 15-20% overlap (adaptive to split quality)
            length_function=lambda text: len(self.encoding.encode(text)),
            separators=[
                "\n## ",      # Markdown H2 (sections) - highest priority
                "\n# ",       # Markdown H1 (chapters)
                "\n### ",     # Markdown H3 (subsections)
                "\n\n\n",     # Multiple blank lines (major breaks)
                "\n\n",       # Paragraph breaks
                "```\n",      # Code block boundaries
                "\n",         # Line breaks
                ". ", "! ", "? ",  # Sentence endings (Latin)
                "。", "！", "？",  # Sentence endings (CJK)
                "؟", "۔",     # Arabic/Urdu punctuation
                "; ", ": ",   # Clause separators
                " ",          # Word boundaries
                ""            # Character fallback
            ]
        )

        logger.info(f"SmartBookProcessor initialized with data_dir: {self.data_dir}")

    async def initialize(self):
        """Initialize database schema"""
        await self._init_db()

    def _extract_headings(self, text: str) -> List[Dict]:
        """
        Extract markdown headings with their positions in the text.

        Returns:
            List of {'title': str, 'position': int, 'level': int}
        """
        headings = []
        # Match markdown headings: # Title, ## Subtitle, etc.
        for match in re.finditer(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE):
            headings.append({
                'title': match.group(2).strip(),
                'position': match.start(),
                'level': len(match.group(1))  # Count number of #'s
            })
        return headings

    def _find_source_context(self, chunk_start_pos: int, headings: List[Dict]) -> Optional[str]:
        """
        Find the most relevant heading/section for a chunk based on position.

        Args:
            chunk_start_pos: Character position where chunk starts in full text
            headings: List of heading dicts from _extract_headings()

        Returns:
            Heading title or None
        """
        # Find last heading before this chunk position
        relevant_heading = None
        for heading in headings:
            if heading['position'] < chunk_start_pos:
                relevant_heading = heading
            else:
                break  # Headings are ordered, so we can stop

        return relevant_heading['title'] if relevant_heading else None

    def _detect_code_block(self, chunk: str) -> bool:
        """
        Detect if chunk contains code using heuristics.
        Fast, no LLM needed.
        """
        # Code fence markers
        if '```' in chunk or '~~~' in chunk:
            return True

        # Inline code
        if re.search(r'`[^`]+`', chunk):
            return True

        # Common programming keywords
        code_patterns = [
            r'\bdef\s+\w+\s*\(',           # Python functions
            r'\bclass\s+\w+',              # Class definitions
            r'\bfunction\s+\w+\s*\(',      # JS functions
            r'\bimport\s+[\w.]+',          # Import statements
            r'\bconst\s+\w+\s*=',          # JS const
            r'\bvar\s+\w+\s*=',            # JS var
            r'\blet\s+\w+\s*=',            # JS let
            r'^\s{4,}\w+.*[{};]',          # Indented code with brackets/semicolons
        ]

        for pattern in code_patterns:
            if re.search(pattern, chunk, re.MULTILINE):
                return True

        return False

    async def _init_db(self):
        """Create database tables if they don't exist"""
        async with aiosqlite.connect(str(self.db_path)) as db:
            # Books table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS books (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    author TEXT,
                    source_path TEXT,
                    upload_timestamp TEXT,
                    total_chunks INTEGER DEFAULT 0
                )
            """)

            # Chunks table with FTS5 for full-text search
            await db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                    chunk_id UNINDEXED,
                    book_id UNINDEXED,
                    content,
                    chunk_index UNINDEXED,
                    tokenize='unicode61'
                )
            """)


            await db.commit()

        logger.info("Database initialized successfully")

    async def check_duplicate(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Check if a book with this title already exists

        Returns:
            Book metadata if found, None otherwise
        """
        async with aiosqlite.connect(str(self.db_path)) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT book_id, title, author FROM books WHERE title = ?",
                (title,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return dict(row)
        return None

    async def process_book_with_progress(
        self,
        book_content: str,
        title: str,
        author: str,
        source_path: str,
        task_id: str,
        book_id: str
    ) -> Dict[str, Any]:
        """
        Process a book and send progress updates

        Args:
            book_content: Full text of the book
            title: Book title
            author: Book author
            source_path: Path to original file
            task_id: Task ID for progress tracking
            book_id: Unique book identifier

        Returns:
            Processing result with statistics
        """
        try:
            # Check for potential prompt injection patterns (warn user)
            suspicious_patterns = [
                '<|endoftext|>',
                '<|im_end|>',
                '<|im_start|>',
                'IGNORE PREVIOUS INSTRUCTIONS',
                'IGNORE ALL PREVIOUS',
                'DISREGARD ALL PREVIOUS'
            ]

            if any(pattern in book_content.upper() for pattern in [p.upper() for p in suspicious_patterns]):
                logger.warning(f"Book '{title}' contains potential prompt injection patterns - flagging for user awareness")
                # Note: Still process the book, just log the warning

            # Send initial progress
            await self._send_progress(task_id, "chunking", "Chunking text with semantic awareness...", 10)

            # Extract document structure (headings)
            headings = self._extract_headings(book_content)

            # Chunk the text with semantic awareness
            raw_chunks = self.text_splitter.split_text(book_content)
            total_chunks = len(raw_chunks)

            # Enrich chunks with source context and metadata
            chunks_with_metadata = []
            for i, chunk in enumerate(raw_chunks):
                # Find position in original text to locate source context
                chunk_pos = book_content.find(chunk[:100])  # Use first 100 chars to locate
                source_context = self._find_source_context(chunk_pos, headings)

                chunks_with_metadata.append({
                    'text': chunk,
                    'source_context': source_context or title,  # Fallback to book title
                    'position': round(i / max(total_chunks - 1, 1), 3),
                    'has_code': self._detect_code_block(chunk),
                    'token_count': len(self.encoding.encode(chunk))
                })

            logger.info(f"Book '{title}' split into {total_chunks} semantic chunks")

            # Send progress
            await self._send_progress(task_id, "storing", f"Storing {total_chunks} chunks...", 40)

            # Store in SQLite
            await self._store_in_sqlite(
                book_id=book_id,
                title=title,
                author=author,
                source_path=source_path,
                chunks_with_metadata=chunks_with_metadata
            )

            # Send progress
            await self._send_progress(task_id, "embedding", "Generating embeddings...", 70)

            # Store embeddings in ChromaDB with enhanced metadata
            if self.chromadb_adapter and self.embedding_service:
                await self._store_embeddings_with_metadata(
                    book_id=book_id,
                    chunks_with_metadata=chunks_with_metadata,
                    task_id=task_id,
                    total_chunks=total_chunks,
                    title=title,
                    author=author
                )
            else:
                logger.warning("ChromaDB adapter or embedding service not available - skipping embeddings")

            # Send final progress
            await self._send_progress(task_id, "complete", "Processing complete!", 100,
                                     total_chunks=total_chunks)

            return {
                "success": True,
                "total_chunks": total_chunks
            }

        except Exception as e:
            logger.error(f"Error processing book '{title}': {e}", exc_info=True)
            await self._send_progress(task_id, "error", f"Error: {str(e)}", 0)
            raise

    async def _store_in_sqlite(
        self,
        book_id: str,
        title: str,
        author: str,
        source_path: str,
        chunks_with_metadata: List[Dict]
    ):
        """Store book and chunks in SQLite database"""
        async with aiosqlite.connect(str(self.db_path)) as db:
            # Insert book
            await db.execute("""
                INSERT INTO books (book_id, title, author, source_path, upload_timestamp, total_chunks)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                book_id,
                title,
                author,
                source_path,
                datetime.utcnow().isoformat(),
                len(chunks_with_metadata)
            ))

            # Get the book's internal ID
            async with db.execute("SELECT id FROM books WHERE book_id = ?", (book_id,)) as cursor:
                row = await cursor.fetchone()
                internal_book_id = row[0] if row else None

            if not internal_book_id:
                raise Exception("Failed to get internal book ID")

            # Insert chunks (extract text from metadata dicts)
            for idx, chunk_data in enumerate(chunks_with_metadata):
                chunk_id = f"{book_id}_chunk_{idx:04d}"
                await db.execute("""
                    INSERT INTO chunks (chunk_id, book_id, content, chunk_index)
                    VALUES (?, ?, ?, ?)
                """, (chunk_id, internal_book_id, chunk_data['text'], idx))

            await db.commit()

        logger.info(f"Stored book '{title}' with {len(chunks_with_metadata)} chunks in SQLite")

    async def _store_embeddings_with_metadata(
        self,
        book_id: str,
        chunks_with_metadata: List[Dict],
        task_id: str,
        total_chunks: int,
        title: str = "Untitled Document",
        author: str = "Unknown"
    ):
        """Generate and store embeddings in ChromaDB with enhanced metadata including title and author"""
        try:
            # Get upload timestamp from SQLite for this book
            upload_timestamp = None
            async with aiosqlite.connect(str(self.db_path)) as db:
                async with db.execute("SELECT upload_timestamp FROM books WHERE book_id = ?", (book_id,)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        upload_timestamp = row[0]

            batch_size = 10  # Process 10 chunks at a time

            for batch_start in range(0, len(chunks_with_metadata), batch_size):
                # Check if task was cancelled
                if self.task_manager and task_id in self.task_manager.cancelled_tasks:
                    logger.info(f"Task {task_id} was cancelled, stopping embedding generation")
                    return

                batch_end = min(batch_start + batch_size, len(chunks_with_metadata))
                batch_chunk_data = chunks_with_metadata[batch_start:batch_end]

                # Extract text for embedding with contextual prefix (v0.2.6)
                # Prepending "Book: {title}, Section: {section}" improves retrieval ~49% (Anthropic research)
                batch_texts = [
                    f"Book: {title}, Section: {chunk_data.get('source_context', title)}. {chunk_data['text']}"
                    for chunk_data in batch_chunk_data
                ]
                logger.debug(f"[BOOK_PROCESSOR] Embedding {len(batch_texts)} chunks with contextual prefix")

                # Generate embeddings in parallel for the batch
                embedding_tasks = [
                    self.embedding_service.embed_text(chunk_text)
                    for chunk_text in batch_texts
                ]
                embeddings = await asyncio.gather(*embedding_tasks)

                # Prepare batch data with improved metadata
                chunk_ids = [f"{book_id}_chunk_{idx:04d}" for idx in range(batch_start, batch_end)]
                metadatas = []

                for i in range(len(batch_chunk_data)):
                    idx = batch_start + i
                    chunk_data = batch_chunk_data[i]

                    # Enhanced metadata schema with title/author for better LLM citations
                    metadata = {
                        # Base metadata
                        "book_id": book_id,
                        "title": title or "Untitled Document",
                        "author": author or "Unknown",
                        "chunk_index": idx,
                        "type": "book_chunk",
                        "content": chunk_data['text'],
                        "text": chunk_data['text'],
                        "upload_timestamp": upload_timestamp,

                        # Improved metadata (from semantic chunking)
                        "source_context": chunk_data.get("source_context", ""),
                        "doc_position": chunk_data.get("position", 0.0),
                        "has_code": chunk_data.get("has_code", False),
                        "token_count": chunk_data.get("token_count", 0)
                    }

                    metadatas.append(metadata)

                # Store batch in ChromaDB
                await self.chromadb_adapter.upsert_vectors(
                    ids=chunk_ids,
                    vectors=embeddings,
                    metadatas=metadatas
                )

                # Send progress update
                progress = 70 + int((batch_end / total_chunks) * 25)
                await self._send_progress(
                    task_id,
                    "embedding",
                    f"Embedding chunk {batch_end}/{total_chunks}...",
                    progress,
                    current_chunk=batch_end
                )

            logger.info(f"Stored {len(chunks_with_metadata)} embeddings with improved metadata in ChromaDB")

        except Exception as e:
            logger.error(f"Error storing embeddings: {e}", exc_info=True)
            # Don't fail the whole process if embeddings fail

    async def backfill_book_timestamps(self):
        """Backfill upload_timestamp to existing book chunks in ChromaDB"""
        try:
            if not self.chromadb_adapter:
                logger.warning("ChromaDB adapter not available, skipping timestamp backfill")
                return

            # Get all books from SQLite
            async with aiosqlite.connect(str(self.db_path)) as db:
                async with db.execute("SELECT book_id, upload_timestamp FROM books") as cursor:
                    books = await cursor.fetchall()

            if not books:
                logger.info("No books found for timestamp backfill")
                return

            logger.info(f"Backfilling timestamps for {len(books)} books...")

            for book_id, upload_timestamp in books:
                # Get all chunk IDs for this book from ChromaDB
                try:
                    collection = self.chromadb_adapter.collection
                    # Get chunks matching this book_id
                    results = collection.get(
                        where={"book_id": book_id},
                        include=["metadatas"]
                    )

                    if results and results['ids']:
                        chunk_ids = results['ids']
                        metadatas = results['metadatas']

                        # Update each metadata to include upload_timestamp
                        updated_metadatas = []
                        for metadata in metadatas:
                            if metadata and 'upload_timestamp' not in metadata:
                                metadata['upload_timestamp'] = upload_timestamp
                            updated_metadatas.append(metadata)

                        # Update ChromaDB with new metadata
                        collection.update(
                            ids=chunk_ids,
                            metadatas=updated_metadatas
                        )

                        logger.info(f"Updated {len(chunk_ids)} chunks for book {book_id}")

                except Exception as e:
                    logger.error(f"Error backfilling timestamps for book {book_id}: {e}")
                    continue

            logger.info("Timestamp backfill complete")

        except Exception as e:
            logger.error(f"Error during timestamp backfill: {e}", exc_info=True)

    async def _send_progress(self, task_id: str, status: str, message: str, progress: int, **kwargs):
        """Send progress update via WebSocket"""
        if not self.task_manager:
            return

        try:
            from backend.api.websocket_progress import send_progress_update
            await send_progress_update(
                task_id,
                status=status,
                message=message,
                progress=progress,
                **kwargs
            )
        except Exception as e:
            logger.warning(f"Failed to send progress update: {e}")

    async def search_full_text(
        self,
        query: str,
        book_title: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Full-text search across all books

        Args:
            query: Search query
            book_title: Optional filter by book title
            limit: Max results
            offset: Pagination offset

        Returns:
            List of matching chunks with metadata
        """
        results = []

        async with aiosqlite.connect(str(self.db_path)) as db:
            db.row_factory = aiosqlite.Row

            if book_title:
                # Search within specific book
                sql = """
                    SELECT c.chunk_id, c.content, c.chunk_index, b.title, b.author
                    FROM chunks c
                    JOIN books b ON c.book_id = b.id
                    WHERE c.content MATCH ? AND b.title = ?
                    LIMIT ? OFFSET ?
                """
                params = (query, book_title, limit, offset)
            else:
                # Search all books
                sql = """
                    SELECT c.chunk_id, c.content, c.chunk_index, b.title, b.author
                    FROM chunks c
                    JOIN books b ON c.book_id = b.id
                    WHERE c.content MATCH ?
                    LIMIT ? OFFSET ?
                """
                params = (query, limit, offset)

            async with db.execute(sql, params) as cursor:
                async for row in cursor:
                    results.append(dict(row))

        return results

    async def get_surrounding_chunks(
        self,
        chunk_id: str,
        radius: int = 2
    ) -> Dict[str, Any]:
        """
        Get chunks surrounding a specific chunk for expanded context.

        Args:
            chunk_id: Chunk identifier (e.g., "book_abc123_chunk_0025")
            radius: Number of chunks before/after to retrieve (default: 2)

        Returns:
            Dictionary with:
                - chunks: List of chunk texts in sequential order
                - book_title: Title of the book
                - book_author: Author of the book
                - chunk_range: String describing the range (e.g., "23-27")
                - center_index: Index of the original chunk

        Raises:
            ValueError: If chunk_id format is invalid
        """
        # Parse chunk_id to extract book_id and chunk_index
        # Expected format: "book_<uuid>_chunk_<index>"
        try:
            parts = chunk_id.split('_chunk_')
            if len(parts) != 2:
                raise ValueError(f"Invalid chunk_id format: {chunk_id}")

            book_id = parts[0]
            chunk_index = int(parts[1])
        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse chunk_id '{chunk_id}': {e}")

        # Calculate range
        start_idx = max(0, chunk_index - radius)
        end_idx = chunk_index + radius

        async with aiosqlite.connect(str(self.db_path)) as db:
            db.row_factory = aiosqlite.Row

            # Get book_id from books table (chunk_id uses external book_id format)
            sql = """
                SELECT c.content, c.chunk_index, b.title, b.author
                FROM chunks c
                JOIN books b ON c.book_id = b.id
                WHERE b.book_id = ?
                AND c.chunk_index BETWEEN ? AND ?
                ORDER BY c.chunk_index
            """

            results = []
            book_title = None
            book_author = None

            async with db.execute(sql, (book_id, start_idx, end_idx)) as cursor:
                async for row in cursor:
                    results.append({
                        "content": row["content"],
                        "chunk_index": row["chunk_index"]
                    })
                    if book_title is None:
                        book_title = row["title"]
                        book_author = row["author"]

        if not results:
            return {
                "chunks": [],
                "book_title": None,
                "book_author": None,
                "chunk_range": f"{start_idx}-{end_idx}",
                "center_index": chunk_index
            }

        return {
            "chunks": [r["content"] for r in results],
            "book_title": book_title,
            "book_author": book_author,
            "chunk_range": f"{start_idx}-{end_idx}",
            "center_index": chunk_index
        }