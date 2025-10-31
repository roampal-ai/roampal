import { useState, useCallback, useEffect, useRef } from 'react';

interface UseSplitPaneOptions {
  initialSize?: number;
  minSize?: number;
  maxSize?: number;
  direction?: 'horizontal' | 'vertical';
  storageKey?: string;
  onResize?: (size: number) => void;
  inverted?: boolean; // For right sidebar - drag left to grow
}

export const useSplitPane = ({
  initialSize = 300,
  minSize = 200,
  maxSize = 600,
  direction = 'horizontal',
  storageKey,
  onResize,
  inverted = false,
}: UseSplitPaneOptions = {}) => {
  const [size, setSize] = useState(() => {
    if (storageKey) {
      const stored = localStorage.getItem(storageKey);
      if (stored) {
        const parsed = parseInt(stored, 10);
        if (!isNaN(parsed)) {
          // If stored size is collapsed (<= 50), use initialSize instead to start expanded
          if (parsed <= 50) {
            return initialSize;
          }
          return Math.max(minSize, Math.min(maxSize, parsed));
        }
      }
    }
    return initialSize;
  });

  const [isDragging, setIsDragging] = useState(false);
  const dragStartPos = useRef<number>(0);
  const dragStartSize = useRef<number>(0);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
    dragStartPos.current = direction === 'horizontal' ? e.clientX : e.clientY;
    dragStartSize.current = size;
    document.body.style.cursor = direction === 'horizontal' ? 'col-resize' : 'row-resize';
    document.body.style.userSelect = 'none';
  }, [direction, size]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging) return;

    const currentPos = direction === 'horizontal' ? e.clientX : e.clientY;
    let delta = currentPos - dragStartPos.current;

    // Invert delta for right sidebar (drag left to grow)
    if (inverted) {
      delta = -delta;
    }

    const newSize = Math.max(minSize, Math.min(maxSize, dragStartSize.current + delta));

    setSize(newSize);
    onResize?.(newSize);
  }, [isDragging, direction, minSize, maxSize, onResize, inverted]);

  const handleMouseUp = useCallback(() => {
    if (!isDragging) return;

    setIsDragging(false);
    document.body.style.cursor = '';
    document.body.style.userSelect = '';

    if (storageKey) {
      localStorage.setItem(storageKey, size.toString());
    }
  }, [isDragging, size, storageKey]);

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);

      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp]);

  // Removed redundant localStorage save - already handled in handleMouseUp

  const reset = useCallback(() => {
    setSize(initialSize);
    if (storageKey) {
      localStorage.setItem(storageKey, initialSize.toString());
    }
  }, [initialSize, storageKey]);

  return {
    size,
    isDragging,
    handleMouseDown,
    reset,
    setSize,
  };
};