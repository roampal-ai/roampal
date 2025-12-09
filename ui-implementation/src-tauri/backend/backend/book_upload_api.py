"""
Book Upload API without shard architecture
Simplified version for LoopSmith's single-user architecture
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks, Request, Depends, WebSocket
from typing import Optional, Dict, List, Any
from pathlib import Path
from datetime import datetime
import json
import uuid
import shutil
import asyncio
from pydantic import BaseModel
import logging
from backend.api.websocket_progress import initialize_task, update_progress, manager, websocket_endpoint

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/book-upload", tags=["book-upload"])

class BookUploadResponse(BaseModel):
    success: bool
    book_id: str
    filename: str
    message: str
    file_size: int
    upload_timestamp: str
    processing_status: str
    task_id: Optional[str] = None
    error: Optional[str] = None

@router.post("/upload", response_model=BookUploadResponse)
async def upload_book(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None, max_length=200),
    description: Optional[str] = Form(None, max_length=1000),
    check_duplicate: bool = Form(True)
):
    """Upload a book file for ingestion"""
    logger.info(f"Starting upload for file '{file.filename}'")
    try:
        # Validate file type
        allowed_extensions = {'.txt', '.md'}
        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )

        # Validate file size (10MB limit)
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning

        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is 10MB (file is {file_size / (1024 * 1024):.1f}MB)"
            )

        # Generate unique book ID
        book_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        logger.debug(f"Generated book_id '{book_id}' for file '{file.filename}'")

        # Check for duplicate if enabled
        if check_duplicate:
            # Use the global book processor from app.state
            if not hasattr(request.app.state, 'book_processor'):
                raise HTTPException(status_code=500, detail="Book processor not initialized")

            processor = request.app.state.book_processor

            # Use the title or filename as the book title
            book_title = title or Path(file.filename).stem
            existing = await processor.check_duplicate(book_title)

            if existing:
                logger.info(f"Duplicate book detected: '{book_title}'")
                return BookUploadResponse(
                    success=False,
                    book_id=existing.get('book_id', ''),
                    filename=file.filename,
                    message=f"Book '{book_title}' already exists in the database",
                    file_size=0,
                    upload_timestamp=datetime.now().isoformat(),
                    processing_status="duplicate",
                    error=f"Duplicate book: '{book_title}' already exists"
                )

        # Create book directory structure
        from config.settings import settings
        books_dir = settings.paths.get_book_folder_path()
        uploads_dir = books_dir / "uploads"
        metadata_dir = books_dir / "metadata"
        db_dir = books_dir / "db"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        db_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory structure under '{books_dir}'")

        # Save uploaded file
        file_path = uploads_dir / f"{book_id}_{file.filename}"
        logger.debug(f"Saving file to '{file_path}'")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.debug(f"File saved successfully, size: {file_path.stat().st_size} bytes")

        # Validate file is valid UTF-8 text
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1024)  # Read first 1KB to verify encoding
        except UnicodeDecodeError:
            # Clean up invalid file
            file_path.unlink()
            raise HTTPException(
                status_code=400,
                detail="File is not valid UTF-8 text. Please ensure the file is a text document."
            )

        # Create book metadata
        book_metadata = {
            "book_id": book_id,
            "original_filename": file.filename,
            "stored_filename": file_path.name,
            "file_size": file_path.stat().st_size,
            "file_type": file_extension,
            "upload_timestamp": timestamp,
            "title": title or Path(file.filename).stem,
            "description": description,
            "processing_status": "pending",
            "file_path": str(file_path)
        }

        # Save metadata
        metadata_file = metadata_dir / f"{book_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(book_metadata, f, indent=2)
        logger.debug(f"Metadata saved to '{metadata_file}'")

        # Initialize task for tracking
        task_id = initialize_task(book_id)
        logger.debug(f"Task initialized with ID '{task_id}'")

        # Start background processing
        background_tasks.add_task(
            process_book_task,
            book_id=book_id,
            file_path=str(file_path),
            metadata=book_metadata,
            task_id=task_id,
            request=request
        )

        logger.info(f"Background task started for book_id '{book_id}' with task_id '{task_id}'")

        return BookUploadResponse(
            success=True,
            book_id=book_id,
            filename=file.filename,
            message="Book uploaded successfully",
            file_size=file_path.stat().st_size,
            upload_timestamp=timestamp,
            processing_status="pending",
            task_id=task_id
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found during upload: {e}")
        return BookUploadResponse(
            success=False,
            book_id="",
            filename=file.filename if file else "",
            message=f"File not found: {str(e)}",
            file_size=0,
            upload_timestamp=datetime.now().isoformat(),
            processing_status="failed",
            error=str(e)
        )
    except PermissionError as e:
        logger.error(f"Permission denied during upload: {e}")
        return BookUploadResponse(
            success=False,
            book_id="",
            filename=file.filename if file else "",
            message=f"Permission denied: {str(e)}",
            file_size=0,
            upload_timestamp=datetime.now().isoformat(),
            processing_status="failed",
            error=str(e)
        )
    except Exception as e:
        # Only include traceback in debug mode to avoid verbose logs in production
        from config.settings import settings
        logger.error(f"Unexpected error during upload: {e}", exc_info=settings.app.log_level == "DEBUG")
        return BookUploadResponse(
            success=False,
            book_id="",
            filename=file.filename if file else "",
            message=f"Upload failed: {str(e)}",
            file_size=0,
            upload_timestamp=datetime.now().isoformat(),
            processing_status="failed",
            error=str(e)
        )

@router.post("/process/{book_id}")
async def process_book(
    book_id: str,
    request: Request,
    background_tasks: BackgroundTasks
):
    """Process an already uploaded book"""
    try:
        from config.settings import settings
        metadata_file = settings.paths.get_book_folder_path() / f"{book_id}.metadata.json"

        if not metadata_file.exists():
            raise HTTPException(status_code=404, detail="Book not found")

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        task_id = str(uuid.uuid4())
        background_tasks.add_task(
            process_book_task,
            book_id=book_id,
            file_path=metadata['file_path'],
            metadata=metadata,
            task_id=task_id,
            request=request
        )

        return {
            "success": True,
            "book_id": book_id,
            "task_id": task_id,
            "message": "Processing started"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

async def process_book_task(
    book_id: str,
    file_path: str,
    metadata: Dict,
    task_id: str,
    request: Request
):
    """Background task to process uploaded book"""
    try:
        logger.info(f"[BOOK_PROCESSOR] Starting processing for book {book_id}")

        # Update progress: Starting
        update_progress(task_id, {
            "task_id": task_id,
            "book_id": book_id,
            "status": "processing",
            "stage": "initializing",
            "progress": 5,
            "message": "Starting book processing..."
        })

        # Check for cancellation before starting
        if task_id in manager.cancelled_tasks:
            logger.info(f"Task {task_id} was cancelled before starting")
            manager.complete_task(task_id, success=False, error="Cancelled by user")
            return

        # Read book content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                book_content = f.read()
        except Exception as e:
            logger.error(f"Failed to read book file {file_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to read book file: {e}")

        # Use the global book processor from app.state
        if not hasattr(request.app.state, 'book_processor'):
            raise HTTPException(status_code=500, detail="Book processor not initialized")

        processor = request.app.state.book_processor

        # Enable cancellation support
        processor.task_manager = manager

        # Store with title from metadata
        title = metadata.get('title', Path(file_path).stem)

        # Process the book with progress updates
        result = await processor.process_book_with_progress(
            book_content=book_content,
            title=title,
            author=metadata.get('author', 'Unknown'),
            source_path=file_path,
            task_id=task_id,
            book_id=book_id
        )

        # Update metadata to mark as processed
        metadata['processing_status'] = 'completed'
        metadata['processed_timestamp'] = datetime.now().isoformat()

        # Add processing stats
        if result:
            metadata['processing_stats'] = {
                'total_chunks': result.get('total_chunks', 0),
                'chunks_processed': result.get('total_chunks', 0)
            }

        from config.settings import settings
        metadata_file = settings.paths.get_book_folder_path() / "metadata" / f"{book_id}.json"
        if metadata_file.exists():
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                logger.info(f"Updated metadata file for processed book {book_id}")

        # Archive the original file after successful processing (instead of deleting)
        from config.settings import settings
        books_dir = settings.paths.get_book_folder_path()
        book_file = Path(file_path)
        if book_file.exists():
            try:
                archive_dir = books_dir / "archive"
                archive_dir.mkdir(exist_ok=True)
                shutil.move(str(book_file), str(archive_dir / book_file.name))
                logger.info(f"Archived uploaded book file to {archive_dir / book_file.name}")
            except Exception as e:
                logger.warning(f"Could not archive book file {book_file.name}: {e}")

        # Send final progress update
        from backend.api.websocket_progress import send_progress_update
        logger.info(f"[BOOK_PROCESSOR] Sending completion update for task {task_id}")
        await send_progress_update(
            task_id,
            status="completed",
            message="Processing completed successfully",
            progress=100,
            total_chunks=result.get('total_chunks', 0)
        )

        # Give WebSocket time to deliver the message
        await asyncio.sleep(0.5)

        # Mark task as complete
        manager.complete_task(task_id, success=True)
        logger.info(f"[BOOK_PROCESSOR] Task {task_id} marked as complete")

        logger.info(f"[BOOK_PROCESSOR] Successfully processed book {book_id}")

    except asyncio.CancelledError:
        # Task was cancelled
        logger.info(f"Task {task_id} was cancelled")
        manager.complete_task(task_id, success=False, error="Cancelled by user")
    except Exception as e:
        # Mark task as failed with detailed error
        error_msg = str(e)
        logger.error(f"Failed to process book {book_id}: {error_msg}")

        # Store error in metadata
        metadata['processing_status'] = 'failed'
        metadata['error'] = error_msg
        metadata['failed_timestamp'] = datetime.now().isoformat()

        from config.settings import settings
        metadata_file = settings.paths.get_book_folder_path() / "metadata" / f"{book_id}.json"
        if metadata_file.exists():
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        manager.complete_task(task_id, success=False, error=error_msg)
        raise

@router.get("/progress/{book_id}")
async def get_processing_progress(book_id: str):
    """Get the progress of book processing"""
    # Find the task ID for this book
    for task_id, progress_data in manager.tasks.items():
        if progress_data.get("book_id") == book_id:
            return progress_data

    # If no active task, check if book is already processed
    from backend.config.settings import settings
    metadata_file = settings.paths.get_book_folder_path() / f"{book_id}.metadata.json"

    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        if metadata.get('processing_status') == 'completed':
            return {
                "book_id": book_id,
                "status": "completed",
                "progress": 100,
                "message": "Book processing complete"
            }

    raise HTTPException(status_code=404, detail="No processing task found for this book")

@router.get("/status/{book_id}")
async def get_book_status(book_id: str):
    """Get the status of a book"""
    try:
        from config.settings import settings
        metadata_file = settings.paths.get_book_folder_path() / f"{book_id}.metadata.json"

        if not metadata_file.exists():
            raise HTTPException(status_code=404, detail="Book not found")

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        return metadata

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.get("/active-tasks")
async def get_active_tasks():
    """Get all active processing tasks"""
    return {
        "tasks": list(manager.tasks.values())
    }

@router.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a processing task"""
    try:
        if task_id in manager.tasks:
            manager.cancelled_tasks.add(task_id)
            manager.update_progress(task_id, {
                "status": "cancelled",
                "message": "Task cancelled by user"
            })
            return {
                "success": True,
                "message": f"Task {task_id} cancelled"
            }
        else:
            return {
                "success": False,
                "message": f"Task {task_id} not found"
            }
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@router.get("/search")
async def search_books(
    query: str,
    book_title: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    request: Request = None
):
    """Full-text search across all books"""
    try:
        # Use the global book processor from app.state
        if not hasattr(request.app.state, 'book_processor'):
            raise HTTPException(status_code=500, detail="Book processor not initialized")

        processor = request.app.state.book_processor

        # Perform search
        results = await processor.search_full_text(
            query=query,
            book_title=book_title,
            limit=limit,
            offset=offset
        )

        return {
            "success": True,
            "query": query,
            "book_title": book_title,
            "results": results,
            "count": len(results),
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(f"Error searching books: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/books")
async def list_books():
    """List all books in the system"""
    try:
        from config.settings import settings
        books = []

        # Get books directory
        books_dir = settings.paths.get_book_folder_path()

        # Read metadata files
        metadata_dir = books_dir / "metadata"
        if metadata_dir.exists():
            for metadata_file in metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        books.append(metadata)
                except Exception as e:
                    logger.error(f"Error reading metadata file {metadata_file}: {e}")
                    continue

        # Get database info if available
        db_path = settings.paths.get_book_content_dir() / "books.db"
        if db_path.exists():
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Get book info from database
            cursor.execute("""
                SELECT b.id, b.title, b.author, b.total_chunks
                FROM books b
            """)

            db_books = {}
            for row in cursor.fetchall():
                db_books[row[0]] = {
                    "db_id": row[0],
                    "title": row[1],
                    "author": row[2],
                    "chunk_count": row[3]
                }

            conn.close()

            # Merge database info with metadata
            for book in books:
                # Try to find matching database entry by title
                for db_id, db_info in db_books.items():
                    if db_info['title'] in book.get('title', '') or book.get('title', '') in db_info['title']:
                        book['processing_stats'] = {
                            "total_chunks": db_info['chunk_count'],
                            "chunks_processed": db_info['chunk_count']
                        }
                        break

        return {
            "success": True,
            "books": books,
            "total": len(books)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list books: {str(e)}")

@router.delete("/books/{book_id}")
async def delete_book(book_id: str, request: Request):
    """Delete a book - removes from database, ChromaDB, and filesystem"""
    try:
        import sqlite3
        from config.settings import settings

        # Validate book_id is a valid UUID to prevent path traversal
        try:
            uuid.UUID(book_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid book_id format")

        # Delete from SQLite database
        db_path = settings.paths.get_book_folder_path() / "books.db"
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Get the book's internal ID directly from the books table using book_id (UUID)
            cursor.execute("""
                SELECT id, title
                FROM books
                WHERE book_id = ?
                LIMIT 1
            """, (book_id,))
            book_record = cursor.fetchone()

            if book_record:
                internal_id, title = book_record

                # Get chunk IDs BEFORE deleting (for ChromaDB cleanup)
                cursor.execute("SELECT chunk_id FROM chunks WHERE book_id = ?", (internal_id,))
                chunk_ids = [row[0] for row in cursor.fetchall()]

                # Delete chunks
                cursor.execute("DELETE FROM chunks WHERE book_id = ?", (internal_id,))
                chunks_deleted = cursor.rowcount

                # Delete book
                cursor.execute("DELETE FROM books WHERE id = ?", (internal_id,))

                conn.commit()
                conn.close()

                logger.info(f"Deleted book '{title}' (ID: {internal_id}) and {chunks_deleted} chunks from database")

                # Delete from ChromaDB if available
                if hasattr(request.app.state, 'memory_collections'):
                    try:
                        # Access the books collection from UnifiedMemorySystem
                        books_adapter = None
                        if hasattr(request.app.state.memory_collections, 'collections'):
                            books_adapter = request.app.state.memory_collections.collections.get('books')
                        elif hasattr(request.app.state.memory_collections, 'books_collection'):
                            books_adapter = request.app.state.memory_collections.books_collection

                        if books_adapter:
                            # Delete chunks by book_id pattern (for new chunk ID format)
                            # Also try to delete by chunk_ids from DB (for old format)
                            try:
                                # First try to delete by book_id pattern (new format)
                                # Get all items and filter by book_id prefix
                                all_items = books_adapter.collection.get()
                                book_chunk_ids = [
                                    id for id in all_items['ids']
                                    if id.startswith(f"{book_id}_chunk_")
                                ]

                                deleted_count = 0
                                if book_chunk_ids:
                                    books_adapter.collection.delete(ids=book_chunk_ids)
                                    deleted_count = len(book_chunk_ids)
                                    logger.info(f"Deleted {deleted_count} embeddings from ChromaDB for book '{title}' (by book_id pattern)")

                                # Also try old format chunk_ids (defensive - ensures cleanup)
                                if chunk_ids:
                                    try:
                                        books_adapter.collection.delete(ids=chunk_ids)
                                        deleted_count += len(chunk_ids)
                                        logger.info(f"Deleted {len(chunk_ids)} additional embeddings from ChromaDB (by database chunk_ids)")
                                    except Exception as e:
                                        # OK if these don't exist (already deleted by pattern match)
                                        logger.debug(f"Chunk IDs already deleted: {e}")

                                # Verify deletion
                                if deleted_count != chunks_deleted:
                                    logger.warning(f"ChromaDB deletion mismatch: deleted {deleted_count} embeddings but expected {chunks_deleted}")
                            except Exception as e:
                                logger.warning(f"Error deleting chunks: {e}")
                    except Exception as chroma_err:
                        logger.warning(f"Failed to delete ChromaDB embeddings for book '{title}': {chroma_err}")
                        # Don't fail the whole deletion if ChromaDB cleanup fails

                # Clean Action KG examples referencing deleted book chunks (v0.2.6)
                # Use chunk_ids from database (always available at this point)
                if hasattr(request.app.state, 'memory') and chunk_ids:
                    try:
                        memory = request.app.state.memory
                        cleaned = await memory.cleanup_action_kg_for_doc_ids(chunk_ids)
                        if cleaned > 0:
                            logger.info(f"Cleaned {cleaned} Action KG examples for deleted book '{title}'")
                    except Exception as kg_err:
                        logger.warning(f"Failed to clean Action KG for book '{title}': {kg_err}")

        # Delete files from filesystem
        books_dir = settings.paths.get_book_folder_path()

        deleted_files = []

        # Delete from metadata directory
        metadata_dir = books_dir / "metadata"
        if metadata_dir.exists():
            for file_path in metadata_dir.glob(f"{book_id}*.json"):
                try:
                    file_path.unlink()
                    deleted_files.append(f"metadata/{file_path.name}")
                    logger.info(f"Deleted metadata file: {file_path.name}")
                except Exception as e:
                    logger.error(f"Error deleting metadata file {file_path}: {e}")

        # Delete from uploads directory if exists
        uploads_dir = books_dir / "uploads"
        if uploads_dir.exists():
            for file_path in uploads_dir.glob(f"{book_id}*"):
                try:
                    file_path.unlink()
                    deleted_files.append(f"uploads/{file_path.name}")
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {e}")

        return {
            "success": True,
            "book_id": book_id,
            "message": f"Deleted book from database and {len(deleted_files)} files",
            "deleted_files": deleted_files
        }

    except HTTPException:
        raise  # Re-raise HTTP exceptions (like the UUID validation error)
    except Exception as e:
        logger.error(f"Failed to delete book: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete book: {str(e)}")

# WebSocket endpoint for progress tracking
@router.websocket("/ws/progress/{task_id}")
async def progress_websocket(websocket: WebSocket, task_id: str):
    await websocket_endpoint(websocket, task_id)