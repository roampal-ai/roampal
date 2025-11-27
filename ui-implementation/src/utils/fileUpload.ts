/**
 * File upload utilities following CLAUDE.md principles
 * - KISS: Single purpose functions
 * - Fail Fast: Validate early
 * - <20 lines per function
 */

import logger from './logger';
import { ROAMPAL_CONFIG } from '../config/roampal';

// Types for file upload tracking
export interface FileUploadState {
  file: File;
  status: 'pending' | 'uploading' | 'uploaded' | 'error';
  progress: number;
  bookId?: string;
  error?: string;
}

// Allowed file types per backend (book_upload_api.py)
const ALLOWED_EXTENSIONS = ['.txt', '.md'];
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB limit

/**
 * Validate file before upload (Fail Fast principle)
 */
export function validateFile(file: File): { valid: boolean; error?: string } {
  // Check file extension
  const ext = '.' + file.name.split('.').pop()?.toLowerCase();
  if (!ALLOWED_EXTENSIONS.includes(ext)) {
    return {
      valid: false,
      error: `Only ${ALLOWED_EXTENSIONS.join(', ')} files are supported`
    };
  }

  // Check file size
  if (file.size > MAX_FILE_SIZE) {
    return {
      valid: false,
      error: `File too large. Maximum size is ${MAX_FILE_SIZE / (1024 * 1024)}MB`
    };
  }

  return { valid: true };
}

/**
 * Upload single file to book-upload endpoint
 * Returns book_id on success
 */
export async function uploadFile(
  file: File,
  onProgress?: (progress: number) => void
): Promise<string> {
  // Validate first (Fail Fast)
  const validation = validateFile(file);
  if (!validation.valid) {
    throw new Error(validation.error);
  }

  const formData = new FormData();
  formData.append('file', file);
  formData.append('title', file.name.replace(/\.[^/.]+$/, '')); // Remove extension
  formData.append('check_duplicate', 'true');

  // XMLHttpRequest for progress tracking
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    // Progress handler
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable && onProgress) {
        const progress = Math.round((e.loaded / e.total) * 100);
        onProgress(progress);
      }
    });

    // Success handler
    xhr.addEventListener('load', () => {
      if (xhr.status === 200) {
        try {
          const response = JSON.parse(xhr.responseText);
          if (response.success && response.book_id) {
            logger.log('[FileUpload] Upload successful:', response.book_id);
            resolve(response.book_id);
          } else {
            reject(new Error(response.error || 'Upload failed'));
          }
        } catch (e) {
          reject(new Error('Invalid server response'));
        }
      } else {
        reject(new Error(`Upload failed: ${xhr.statusText}`));
      }
    });

    // Error handler
    xhr.addEventListener('error', () => {
      reject(new Error('Network error during upload'));
    });

    // Send request
    xhr.open('POST', `${ROAMPAL_CONFIG.apiUrl}/api/book-upload/upload`);
    xhr.send(formData);
  });
}

/**
 * Upload multiple files sequentially (not parallel to avoid server overload)
 */
export async function uploadFiles(
  files: File[],
  onFileProgress?: (fileIndex: number, progress: number) => void
): Promise<string[]> {
  const bookIds: string[] = [];

  for (let i = 0; i < files.length; i++) {
    try {
      const bookId = await uploadFile(files[i], (progress) => {
        if (onFileProgress) {
          onFileProgress(i, progress);
        }
      });
      bookIds.push(bookId);
    } catch (error) {
      logger.error('[FileUpload] Failed to upload file:', files[i].name, error);
      throw error; // Fail Fast - stop on first error
    }
  }

  return bookIds;
}

/**
 * Format file size for display
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

/**
 * Get file icon class based on file type
 */
export function getFileIconType(file: File): 'text' | 'image' | 'document' {
  const type = file.type.toLowerCase();
  if (type.includes('image')) return 'image';
  if (type.includes('text') || file.name.endsWith('.md')) return 'text';
  return 'document';
}