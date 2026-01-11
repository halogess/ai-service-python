/**
 * Canvas Renderer Module
 * Provides canvas drawing utilities for bounding boxes
 * Used by both extraction and alignment pages
 */

const CanvasRenderer = (function () {
    'use strict';

    // Default colors for different item types
    const DEFAULT_COLORS = {
        group: '#8e44ad',
        table: '#e74c3c',
        hline_table: '#c0392b',
        shape: '#3498db',
        image: '#27ae60',
        cell: '#f39c12',
        alignment: '#4caf50'
    };

    /**
     * Load page image and draw on canvas
     * @param {HTMLCanvasElement} canvas - Canvas element
     * @param {string} imageUrl - URL of page image
     * @returns {Promise<{image: HTMLImageElement, scale: number}>} - Loaded image and scale factor
     */
    async function loadPageImage(canvas, imageUrl) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = function () {
                const ctx = canvas.getContext('2d');

                // Calculate scale to fit canvas container
                const container = canvas.parentElement;
                const containerWidth = container ? container.clientWidth : 800;
                const scale = containerWidth / img.width;

                canvas.width = containerWidth;
                canvas.height = img.height * scale;

                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

                resolve({ image: img, scale: scale });
            };
            img.onerror = reject;
            img.src = imageUrl;
        });
    }

    /**
     * Draw a single bounding box
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Array} bbox - [x0, y0, x1, y1] coordinates
     * @param {number} scale - Scale factor
     * @param {string} color - Box color
     * @param {string} label - Optional label to display
     * @param {Object} options - Additional options
     */
    function drawBbox(ctx, bbox, scale, color, label = null, options = {}) {
        if (!bbox || bbox.length < 4) return;

        const {
            lineWidth = 2,
            alpha = 0.3,
            labelFontSize = 10,
            fillBox = true
        } = options;

        const x = bbox[0] * scale;
        const y = bbox[1] * scale;
        const width = (bbox[2] - bbox[0]) * scale;
        const height = (bbox[3] - bbox[1]) * scale;

        // Draw filled rectangle
        if (fillBox) {
            ctx.fillStyle = hexToRgba(color, alpha);
            ctx.fillRect(x, y, width, height);
        }

        // Draw border
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.strokeRect(x, y, width, height);

        // Draw label if provided
        if (label) {
            ctx.font = `bold ${labelFontSize}px Arial`;
            ctx.fillStyle = color;

            // Background for label
            const textMetrics = ctx.measureText(label);
            const textWidth = textMetrics.width;
            const textHeight = labelFontSize;
            ctx.fillStyle = 'white';
            ctx.fillRect(x, y - textHeight - 2, textWidth + 4, textHeight + 2);

            // Label text
            ctx.fillStyle = color;
            ctx.fillText(label, x + 2, y - 4);
        }
    }

    /**
     * Draw extraction items on canvas
     * @param {HTMLCanvasElement} canvas - Canvas element
     * @param {HTMLImageElement} pageImage - Page image
     * @param {number} scale - Scale factor
     * @param {Array} items - Extraction items
     * @param {Set} visibleItems - Set of visible item indices
     * @param {Set} visibleCells - Set of visible cell keys (e.g., "table_0_cell_1_2")
     * @param {Object} options - Additional options
     */
    function drawExtractionItems(canvas, pageImage, scale, items, visibleItems, visibleCells = new Set(), options = {}) {
        const ctx = canvas.getContext('2d');

        // Clear and redraw page image
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (pageImage) {
            ctx.drawImage(pageImage, 0, 0, canvas.width, canvas.height);
        }

        // Draw visible items
        items.forEach((item, idx) => {
            if (!visibleItems.has(idx)) return;

            const color = DEFAULT_COLORS[item.type] || '#888';
            const label = `[${idx}] ${item.type.toUpperCase()}`;

            // Draw item bbox
            if (item.bbox) {
                drawBbox(ctx, item.bbox, scale, color, label);
            }

            // For tables, also draw visible cells
            if ((item.type === 'table' || item.type === 'hline_table') && item.data.rows) {
                item.data.rows.forEach((row, rIdx) => {
                    if (row.cells) {
                        row.cells.forEach((cell, cIdx) => {
                            const cellKey = `table_${idx}_cell_${rIdx}_${cIdx}`;
                            if (visibleCells.has(cellKey) && cell.bbox) {
                                drawBbox(ctx, cell.bbox, scale, DEFAULT_COLORS.cell, `R${rIdx}C${cIdx}`, { alpha: 0.2 });

                                // Also draw content bbox if exists
                                if (cell.content) {
                                    cell.content.forEach(content => {
                                        if (content.bbox) {
                                            const contentColor = content.type === 'text' ? '#9b59b6' : '#27ae60';
                                            drawBbox(ctx, content.bbox, scale, contentColor, null, { lineWidth: 1, fillBox: false });
                                        }
                                    });
                                }
                            }
                        });
                    }
                });
            }
        });
    }

    /**
     * Draw alignment bboxes on canvas
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} scale - Scale factor
     * @param {Array} alignments - Alignment data
     * @param {Set} visibleAlignBbox - Visible alignment element IDs
     */
    function drawAlignmentBboxes(ctx, scale, alignments, visibleAlignBbox) {
        alignments.forEach(alignment => {
            if (!visibleAlignBbox.has(alignment.element_id)) return;

            // Draw merged_bbox for the whole alignment
            if (alignment.merged_bbox) {
                drawBbox(ctx, alignment.merged_bbox, scale, DEFAULT_COLORS.alignment,
                    `#${alignment.element_sequence}`, { alpha: 0.15, lineWidth: 3 });
            }

            // Draw individual matched PDF unit bboxes
            if (alignment.matched_pdf_units) {
                alignment.matched_pdf_units.forEach(unit => {
                    if (unit.bbox) {
                        // Use different color for absorbed units
                        const color = unit.absorbed ? '#ff9800' : DEFAULT_COLORS.alignment;
                        const alpha = unit.absorbed ? 0.3 : 0.25;
                        drawBbox(ctx, unit.bbox, scale, color, null, { alpha: alpha, lineWidth: 1 });
                    }
                });
            }

            // For tables, draw cell bboxes
            if (alignment.is_table && alignment.cells) {
                alignment.cells.forEach(cell => {
                    const cellId = `${alignment.element_id}_r${cell.row}_c${cell.col}`;
                    if (visibleAlignBbox.has(cellId)) {
                        // Draw cell merged_bbox
                        if (cell.merged_bbox) {
                            drawBbox(ctx, cell.merged_bbox, scale, '#27ae60',
                                `R${cell.row}C${cell.col}`, { alpha: 0.15, lineWidth: 2 });
                        }
                        // Draw individual matched PDF units in cell
                        if (cell.matched_pdf_units) {
                            cell.matched_pdf_units.forEach(unit => {
                                if (unit.bbox) {
                                    const color = unit.absorbed ? '#ff9800' : '#4caf50';
                                    const alpha = unit.absorbed ? 0.3 : 0.2;
                                    drawBbox(ctx, unit.bbox, scale, color, null, { alpha: alpha, lineWidth: 1 });
                                }
                            });
                        }
                    }
                });
            }
        });
    }

    /**
     * Convert hex color to rgba
     * @param {string} hex - Hex color code
     * @param {number} alpha - Alpha value (0-1)
     * @returns {string} - RGBA color string
     */
    function hexToRgba(hex, alpha) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        if (result) {
            const r = parseInt(result[1], 16);
            const g = parseInt(result[2], 16);
            const b = parseInt(result[3], 16);
            return `rgba(${r}, ${g}, ${b}, ${alpha})`;
        }
        return hex;
    }

    /**
     * Get color for item type
     * @param {string} type - Item type
     * @returns {string} - Color code
     */
    function getColor(type) {
        return DEFAULT_COLORS[type] || '#888';
    }

    // Public API
    return {
        loadPageImage,
        drawBbox,
        drawExtractionItems,
        drawAlignmentBboxes,
        hexToRgba,
        getColor,
        DEFAULT_COLORS
    };
})();

// Export for module environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CanvasRenderer;
}
