/**
 * PDF Extraction Module
 * Provides core functionality for extracting and processing PDF data via merging API
 * Can be used by both pymupdf_extract.html and dokumen_elemen_viewer.html
 */

const PdfExtraction = (function () {
    'use strict';

    /**
     * Fetch and process merging data from the backend API
     * @param {number} docId - Document ID
     * @param {number} pageNum - Page number (1-indexed)
     * @returns {Promise<Object>} - Processed extraction result
     */
    async function extractMergingData(docId, pageNum) {
        const response = await fetch(`/pymupdf-api/merging/${docId}/${pageNum}`);
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Extraction failed');
        }

        return processMergingResponse(data);
    }

    /**
     * Process the raw API response into a unified structure
     * @param {Object} data - Raw API response
     * @returns {Object} - Processed data with items array
     */
    function processMergingResponse(data) {
        const items = [];

        // Get data from response
        const unclaimedGroups = data.char_groups || [];
        const tablesData = data.basic_tables || [];
        const hlineTablesData = data.hline_tables || [];
        const shapesData = data.shapes || [];
        const pageImagesData = data.page_images || [];
        const debugGroupsLog = data.debug_groups_log || [];

        // Add unclaimed groups
        unclaimedGroups.forEach(group => {
            items.push({
                type: 'group',
                data: {
                    text: group.text,
                    mergedBbox: group.merged_bbox,
                    isSingle: group.is_single,
                    blockIdx: group.block_idx
                },
                bbox: group.merged_bbox
            });
        });

        // Add basic tables
        tablesData.forEach(table => {
            items.push({
                type: 'table',
                data: table,
                bbox: table.bbox
            });
        });

        // Add hline tables
        hlineTablesData.forEach(hTable => {
            items.push({
                type: 'hline_table',
                data: hTable,
                bbox: hTable.bbox
            });
        });

        // Add shapes
        shapesData.forEach(shape => {
            items.push({
                type: 'shape',
                data: shape,
                bbox: shape.bbox
            });
        });

        // Add page images (already filtered by backend)
        pageImagesData.forEach(img => {
            items.push({
                type: 'image',
                data: img,
                bbox: img.bbox
            });
        });

        // Debug: log items before sorting
        console.log('[PdfExtraction] Before sort:', items.map((item, i) => ({
            idx: i,
            type: item.type,
            topY: item.bbox ? item.bbox[1] : null,
            text: item.type === 'group' ? item.data.text?.substring(0, 30) : `(${item.type})`
        })));

        // Sort by reading order
        sortByReadingOrder(items);

        // Debug: log items after sorting
        console.log('[PdfExtraction] After sort:', items.map((item, i) => ({
            idx: i,
            type: item.type,
            topY: item.bbox ? item.bbox[1] : null,
            text: item.type === 'group' ? item.data.text?.substring(0, 30) : `(${item.type})`
        })));

        return {
            items: items,
            width: data.width,
            height: data.height,
            rawdata: data,  // Keep original for debugging
            stats: {
                groups: items.filter(i => i.type === 'group').length,
                tables: items.filter(i => i.type === 'table').length,
                hlineTables: items.filter(i => i.type === 'hline_table').length,
                shapes: items.filter(i => i.type === 'shape').length,
                images: items.filter(i => i.type === 'image').length,
                total: items.length
            },
            debugGroupsLog: debugGroupsLog
        };
    }

    /**
     * Sort items by reading order: Y first, then X if on same line
     * Items are considered "same line" if 30% of the smaller element's 
     * height overlaps vertically with the larger element
     * @param {Array} items - Array of items with bbox property
     */
    function sortByReadingOrder(items) {
        items.sort((a, b) => {
            if (!a.bbox || !b.bbox) return 0;

            const yA0 = a.bbox[1];  // top Y of A
            const yA1 = a.bbox[3];  // bottom Y of A
            const yB0 = b.bbox[1];  // top Y of B
            const yB1 = b.bbox[3];  // bottom Y of B

            const heightA = yA1 - yA0;
            const heightB = yB1 - yB0;

            // Calculate Y overlap amount
            const overlapStart = Math.max(yA0, yB0);
            const overlapEnd = Math.min(yA1, yB1);
            const overlapAmount = Math.max(0, overlapEnd - overlapStart);

            // Find the smaller element's height
            const smallerHeight = Math.min(heightA, heightB);

            // Calculate overlap ratio based on smaller element
            const overlapRatio = smallerHeight > 0 ? overlapAmount / smallerHeight : 0;

            // Same line if 30% of smaller element overlaps
            const OVERLAP_THRESHOLD = 0.30;
            const isSameLine = overlapRatio >= OVERLAP_THRESHOLD;

            if (isSameLine) {
                // Same line - sort by X (left to right)
                return a.bbox[0] - b.bbox[0];
            } else {
                // Different lines - sort by top Y (top to bottom)
                return yA0 - yB0;
            }
        });
    }

    /**
     * Get statistics string for display
     * @param {Object} stats - Statistics object from processMergingResponse
     * @returns {string} - Formatted stats string
     */
    function getStatsString(stats) {
        return `${stats.total} items (${stats.groups} groups, ${stats.tables} tables, ${stats.hlineTables} h-tables, ${stats.shapes} shapes, ${stats.images} images)`;
    }

    /**
     * Get text content from an item (flattened)
     * @param {Object} item - Item from items array
     * @returns {string} - Text content
     */
    function getItemText(item) {
        if (item.type === 'group') {
            return item.data.text || '';
        } else if (item.type === 'table' || item.type === 'hline_table') {
            // Collect text from all cells
            const texts = [];
            const table = item.data;
            if (table.rows) {
                table.rows.forEach(row => {
                    if (row.cells) {
                        row.cells.forEach(cell => {
                            if (cell.content) {
                                cell.content.forEach(c => {
                                    if (c.type === 'text' && c.text) {
                                        texts.push(c.text);
                                    }
                                });
                            }
                        });
                    }
                });
            }
            return texts.join(' ');
        } else if (item.type === 'shape') {
            return item.data.text || '';
        }
        return '';
    }

    /**
     * Get color for item type
     * @param {string} type - Item type
     * @returns {string} - CSS color
     */
    function getTypeColor(type) {
        const colors = {
            'group': '#8e44ad',
            'table': '#e74c3c',
            'hline_table': '#c0392b',
            'shape': '#3498db',
            'image': '#27ae60'
        };
        return colors[type] || '#888';
    }

    /**
     * Get type label for display
     * @param {string} type - Item type
     * @returns {string} - Display label
     */
    function getTypeLabel(type) {
        const labels = {
            'group': 'GROUP',
            'table': 'TABLE',
            'hline_table': 'H-TABLE',
            'shape': 'SHAPE',
            'image': 'IMAGE'
        };
        return labels[type] || type.toUpperCase();
    }

    // Public API
    return {
        extractMergingData,
        processMergingResponse,
        sortByReadingOrder,
        getStatsString,
        getItemText,
        getTypeColor,
        getTypeLabel
    };
})();

// Export for module environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PdfExtraction;
}
