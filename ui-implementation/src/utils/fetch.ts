/**
 * Fetch wrapper that works in both dev and Tauri production builds
 * Uses Tauri's HTTP client in production, native fetch in dev
 */

import { fetch as tauriFetch, ResponseType, Body } from '@tauri-apps/api/http';

const isTauri = () => typeof window !== 'undefined' && window.__TAURI__ !== undefined;

/**
 * Unified fetch function that works in both dev and production
 */
export async function apiFetch(url: string, options?: RequestInit): Promise<Response> {
  // Use native fetch for:
  // 1. Dev mode (non-Tauri)
  // 2. Localhost URLs (CSP allows it, and native fetch supports streaming)
  if (!isTauri() || url.includes('localhost') || url.includes('127.0.0.1')) {
    return fetch(url, options);
  }

  // In Tauri production for external URLs, use Tauri's HTTP client
  try {
    const method = (options?.method || 'GET') as any;
    const headers = options?.headers as Record<string, string> || {};

    let body: Body | undefined;
    if (options?.body) {
      if (typeof options.body === 'string') {
        body = Body.text(options.body);
      } else if (options.body instanceof FormData || options.body instanceof URLSearchParams) {
        // FormData and URLSearchParams need to be converted to string
        body = Body.text(options.body.toString());
      } else if (options.body instanceof Blob || options.body instanceof ArrayBuffer) {
        // For binary data, convert to string (base64 encoding might be needed for actual use)
        body = Body.text('[Binary data]');
      } else if (typeof options.body === 'object') {
        // JSON objects
        body = Body.text(JSON.stringify(options.body));
      } else {
        // Fallback for any other type
        body = Body.text(String(options.body));
      }
    }

    const response = await tauriFetch(url, {
      method,
      headers,
      body,
      responseType: ResponseType.Text,
    });

    // Convert Tauri response to fetch Response interface
    const responseInit: ResponseInit = {
      status: response.status,
      statusText: response.ok ? 'OK' : 'Error',
      headers: new Headers(response.headers),
    };

    return new Response(response.data as string, responseInit);
  } catch (error: any) {
    throw new Error(`Tauri fetch failed: ${error.message || error}`);
  }
}
