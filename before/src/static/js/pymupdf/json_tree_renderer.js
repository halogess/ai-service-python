/**
 * JSON Tree Renderer Module
 * Renders extracted PDF data as an interactive JSON tree
 * Used for displaying merging() results in both extraction and alignment pages
 */

const JsonTreeRenderer = (function () {
    'use strict';

    // Utility to escape HTML
    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Generate unique node ID
     * @returns {string} - Unique ID
     */
    function generateNodeId() {
        return 'node_' + Math.random().toString(36).substr(2, 9);
    }

    /**
     * Toggle node expansion
     * @param {string} nodeId - ID of node children container
     * @param {HTMLElement} rowElement - The row element clicked
     */
    function toggleNode(nodeId, rowElement) {
        const children = document.getElementById(nodeId);
        const toggle = rowElement.querySelector('.json-toggle');
        if (children) {
            children.classList.toggle('show');
            if (toggle) {
                toggle.classList.toggle('expanded');
            }
        }
    }

    /**
     * Build tree for extraction items (groups, tables, shapes, images)
     * @param {Array} items - Array of extraction items
     * @param {Object} options - Rendering options
     * @returns {string} - HTML string
     */
    function buildExtractionTree(items, options = {}) {
        const {
            onItemToggle = null,  // Callback name for item bbox toggle
            onCellToggle = null,  // Callback name for cell bbox toggle
            showBboxButtons = true
        } = options;

        if (!items || items.length === 0) {
            return '<div style="padding: 15px; text-align: center; color: #888;">No items extracted</div>';
        }

        let html = '';
        items.forEach((item, idx) => {
            const nodeId = generateNodeId();
            const bbox = item.bbox ? `[${item.bbox.map(v => v.toFixed(1)).join(', ')}]` : 'null';

            if (item.type === 'group') {
                html += buildGroupNode(item, idx, nodeId, bbox, onItemToggle, showBboxButtons);
            } else if (item.type === 'table' || item.type === 'hline_table') {
                html += buildTableNode(item, idx, nodeId, bbox, onItemToggle, onCellToggle, showBboxButtons);
            } else if (item.type === 'shape') {
                html += buildShapeNode(item, idx, nodeId, bbox, onItemToggle, showBboxButtons);
            } else if (item.type === 'image') {
                html += buildImageNode(item, idx, nodeId, bbox, onItemToggle, showBboxButtons);
            }
        });

        return html;
    }

    /**
     * Build group node
     */
    function buildGroupNode(item, idx, nodeId, bbox, onToggle, showBboxButtons) {
        const group = item.data;
        const textPreview = group.text ? (group.text.length > 50 ? group.text.substring(0, 50) + '...' : group.text) : '';
        const toggleBtn = showBboxButtons && onToggle
            ? `<button class="bbox-toggle merging-item-toggle" data-idx="${idx}" style="border-color: #8e44ad; color: #8e44ad;" onclick="${onToggle}(${idx}, event)">Show</button>`
            : '';

        return `<div class="json-node">
            <div class="json-row expandable" onclick="JsonTreeRenderer.toggleNode('${nodeId}', this)">
                <span class="json-toggle"><i class="fas fa-chevron-right"></i></span>
                <span class="json-key">[${idx}]</span>
                <span class="json-type-hint" style="color: #8e44ad;">GROUP: "${escapeHtml(textPreview)}"</span>
                ${toggleBtn}
            </div>
            <div class="json-children" id="${nodeId}">
                <div class="json-node"><div class="json-row"><span class="json-toggle empty"></span><span class="json-key">"type"</span><span class="json-colon">:</span><span class="json-value string">"group"</span></div></div>
                <div class="json-node"><div class="json-row"><span class="json-toggle empty"></span><span class="json-key">"text"</span><span class="json-colon">:</span><span class="json-value string">"${escapeHtml(group.text || '')}"</span></div></div>
                <div class="json-node"><div class="json-row"><span class="json-toggle empty"></span><span class="json-key">"bbox"</span><span class="json-colon">:</span><span class="json-value">${bbox}</span></div></div>
            </div>
        </div>`;
    }

    /**
     * Build table node (for both table and hline_table)
     */
    function buildTableNode(item, idx, nodeId, bbox, onItemToggle, onCellToggle, showBboxButtons) {
        const table = item.data;
        const isHline = item.type === 'hline_table';
        const color = isHline ? '#c0392b' : '#e74c3c';
        const typeLabel = isHline ? 'H-TABLE' : 'TABLE';
        const info = `${table.row_count || 0} rows × ${table.col_count || 0} cols`;

        const toggleBtn = showBboxButtons && onItemToggle
            ? `<button class="bbox-toggle merging-item-toggle" data-idx="${idx}" style="border-color: ${color}; color: ${color};" onclick="${onItemToggle}(${idx}, event)">Show</button>`
            : '';

        let html = `<div class="json-node">
            <div class="json-row expandable" onclick="JsonTreeRenderer.toggleNode('${nodeId}', this)">
                <span class="json-toggle"><i class="fas fa-chevron-right"></i></span>
                <span class="json-key">[${idx}]</span>
                <span class="json-type-hint" style="color: ${color};">${typeLabel}: ${info}</span>
                ${toggleBtn}
            </div>
            <div class="json-children" id="${nodeId}">
                <div class="json-node"><div class="json-row"><span class="json-toggle empty"></span><span class="json-key">"type"</span><span class="json-colon">:</span><span class="json-value string">"${item.type}"</span></div></div>
                <div class="json-node"><div class="json-row"><span class="json-toggle empty"></span><span class="json-key">"bbox"</span><span class="json-colon">:</span><span class="json-value">${bbox}</span></div></div>
                <div class="json-node"><div class="json-row"><span class="json-toggle empty"></span><span class="json-key">"row_count"</span><span class="json-colon">:</span><span class="json-value number">${table.row_count || 0}</span></div></div>
                <div class="json-node"><div class="json-row"><span class="json-toggle empty"></span><span class="json-key">"col_count"</span><span class="json-colon">:</span><span class="json-value number">${table.col_count || 0}</span></div></div>`;

        // Add cells
        if (table.rows && table.rows.length > 0) {
            const cellsNodeId = generateNodeId();
            let totalCells = 0;
            table.rows.forEach(r => { if (r.cells) totalCells += r.cells.length; });

            html += `<div class="json-node">
                <div class="json-row expandable" onclick="JsonTreeRenderer.toggleNode('${cellsNodeId}', this)">
                    <span class="json-toggle"><i class="fas fa-chevron-right"></i></span>
                    <span class="json-key">"cells"</span>
                    <span class="json-type-hint">${totalCells} cells</span>
                </div>
                <div class="json-children" id="${cellsNodeId}">`;

            table.rows.forEach((row, rIdx) => {
                if (row.cells) {
                    row.cells.forEach((cell, cIdx) => {
                        if (cell && cell.bbox) {
                            html += buildCellNode(cell, idx, rIdx, cIdx, onCellToggle, showBboxButtons);
                        }
                    });
                }
            });

            html += `</div></div>`;
        }

        html += `</div></div>`;
        return html;
    }

    /**
     * Build cell node for table
     */
    function buildCellNode(cell, tableIdx, rIdx, cIdx, onCellToggle, showBboxButtons) {
        const cellBbox = `[${cell.bbox.map(v => v.toFixed(1)).join(', ')}]`;
        const cellNodeId = generateNodeId();
        const contentItems = cell.content || [];
        let contentPreview = '';

        const textCount = contentItems.filter(c => c.type === 'text').length;
        const imageCount = contentItems.filter(c => c.type === 'image').length;

        if (textCount > 0 || imageCount > 0) {
            let parts = [];
            if (textCount > 0) parts.push(`${textCount} text`);
            if (imageCount > 0) parts.push(`${imageCount} img`);
            contentPreview = parts.join(', ');
        } else {
            contentPreview = 'empty';
        }

        const toggleBtn = showBboxButtons && onCellToggle
            ? `<button class="bbox-toggle merging-cell-toggle" style="border-color: #c0392b; color: #c0392b; font-size: 0.6rem;" onclick="${onCellToggle}(${tableIdx}, ${rIdx}, ${cIdx}, event)">Show</button>`
            : '';

        let html = `<div class="json-node">
            <div class="json-row expandable" onclick="JsonTreeRenderer.toggleNode('${cellNodeId}', this)">
                <span class="json-toggle"><i class="fas fa-chevron-right"></i></span>
                <span class="json-key">R${cell.row}C${cell.col}</span>
                <span class="json-type-hint">(${contentPreview})</span>
                <span class="json-value" style="margin-left: 5px; color: #999; font-size: 0.7rem;">${cellBbox}</span>
                ${toggleBtn}
            </div>
            <div class="json-children" id="${cellNodeId}">`;

        // Show content items
        contentItems.forEach((content, contentIdx) => {
            if (content.type === 'text') {
                const textPreview = content.text ? (content.text.length > 40 ? content.text.substring(0, 40) + '...' : content.text) : '';
                const textBbox = content.bbox ? `[${content.bbox.map(v => v.toFixed(1)).join(', ')}]` : 'null';
                html += `<div class="json-node">
                    <div class="json-row">
                        <span class="json-toggle empty"></span>
                        <span style="color: #8e44ad;">[TEXT]</span>
                        <span class="json-value string" style="margin-left: 5px;">"${escapeHtml(textPreview)}"</span>
                        <span style="margin-left: 5px; color: #1abc9c; font-size: 0.7rem;">bbox: ${textBbox}</span>
                        <span style="margin-left: 5px; color: #888; font-size: 0.65rem;">(${content.groups_count || 1} groups)</span>
                    </div>
                </div>`;
            } else if (content.type === 'image') {
                const imgBbox = content.bbox ? `[${content.bbox.map(v => v.toFixed(1)).join(', ')}]` : 'null';
                html += `<div class="json-node">
                    <div class="json-row">
                        <span class="json-toggle empty"></span>
                        <span style="color: #27ae60;">[IMAGE]</span>
                        <span style="margin-left: 5px; color: #888;">${content.width}×${content.height}</span>
                        <span style="margin-left: 5px; color: #27ae60; font-size: 0.7rem;">bbox: ${imgBbox}</span>
                    </div>
                </div>`;
            }
        });

        html += `</div></div>`;
        return html;
    }

    /**
     * Build shape node
     */
    function buildShapeNode(item, idx, nodeId, bbox, onToggle, showBboxButtons) {
        const shape = item.data;
        const textPreview = shape.text ? (shape.text.length > 40 ? shape.text.substring(0, 40) + '...' : shape.text) : '';
        const toggleBtn = showBboxButtons && onToggle
            ? `<button class="bbox-toggle merging-item-toggle" data-idx="${idx}" style="border-color: #3498db; color: #3498db;" onclick="${onToggle}(${idx}, event)">Show</button>`
            : '';

        return `<div class="json-node">
            <div class="json-row expandable" onclick="JsonTreeRenderer.toggleNode('${nodeId}', this)">
                <span class="json-toggle"><i class="fas fa-chevron-right"></i></span>
                <span class="json-key">[${idx}]</span>
                <span class="json-type-hint" style="color: #3498db;">SHAPE: "${escapeHtml(textPreview)}"</span>
                ${toggleBtn}
            </div>
            <div class="json-children" id="${nodeId}">
                <div class="json-node"><div class="json-row"><span class="json-toggle empty"></span><span class="json-key">"type"</span><span class="json-colon">:</span><span class="json-value string">"shape"</span></div></div>
                <div class="json-node"><div class="json-row"><span class="json-toggle empty"></span><span class="json-key">"text"</span><span class="json-colon">:</span><span class="json-value string">"${escapeHtml(shape.text || '')}"</span></div></div>
                <div class="json-node"><div class="json-row"><span class="json-toggle empty"></span><span class="json-key">"bbox"</span><span class="json-colon">:</span><span class="json-value">${bbox}</span></div></div>
                <div class="json-node"><div class="json-row"><span class="json-toggle empty"></span><span class="json-key">"groups_count"</span><span class="json-colon">:</span><span class="json-value number">${shape.groups_count || 0}</span></div></div>
            </div>
        </div>`;
    }

    /**
     * Build image node
     */
    function buildImageNode(item, idx, nodeId, bbox, onToggle, showBboxButtons) {
        const img = item.data;
        const toggleBtn = showBboxButtons && onToggle
            ? `<button class="bbox-toggle merging-item-toggle" data-idx="${idx}" style="border-color: #27ae60; color: #27ae60;" onclick="${onToggle}(${idx}, event)">Show</button>`
            : '';

        return `<div class="json-node">
            <div class="json-row expandable" onclick="JsonTreeRenderer.toggleNode('${nodeId}', this)">
                <span class="json-toggle"><i class="fas fa-chevron-right"></i></span>
                <span class="json-key">[${idx}]</span>
                <span class="json-type-hint" style="color: #27ae60;">IMAGE: ${img.width || '?'}×${img.height || '?'}</span>
                ${toggleBtn}
            </div>
            <div class="json-children" id="${nodeId}">
                <div class="json-node"><div class="json-row"><span class="json-toggle empty"></span><span class="json-key">"type"</span><span class="json-colon">:</span><span class="json-value string">"image"</span></div></div>
                <div class="json-node"><div class="json-row"><span class="json-toggle empty"></span><span class="json-key">"bbox"</span><span class="json-colon">:</span><span class="json-value">${bbox}</span></div></div>
                <div class="json-node"><div class="json-row"><span class="json-toggle empty"></span><span class="json-key">"width"</span><span class="json-colon">:</span><span class="json-value number">${img.width || 0}</span></div></div>
                <div class="json-node"><div class="json-row"><span class="json-toggle empty"></span><span class="json-key">"height"</span><span class="json-colon">:</span><span class="json-value number">${img.height || 0}</span></div></div>
            </div>
        </div>`;
    }

    /**
     * Get CSS styles for the JSON tree (to be included in page)
     * @returns {string} - CSS styles
     */
    function getStyles() {
        return `
        .json-tree { font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace; font-size: 0.8rem; }
        .json-node { border-bottom: 1px solid #f0f0f0; }
        .json-row { display: flex; align-items: center; padding: 4px 8px; cursor: default; gap: 6px; }
        .json-row:hover { background: #f8f9fa; }
        .json-row.expandable { cursor: pointer; }
        .json-toggle { width: 14px; height: 14px; display: flex; align-items: center; justify-content: center; color: #6c757d; font-size: 0.6rem; flex-shrink: 0; transition: transform 0.15s; }
        .json-toggle.expanded { transform: rotate(90deg); }
        .json-toggle.empty { visibility: hidden; }
        .json-key { color: #0550ae; font-weight: 500; flex-shrink: 0; }
        .json-colon { color: #6c757d; margin: 0 4px; }
        .json-value { word-break: break-all; flex: 1; min-width: 0; }
        .json-value.string { color: #0a3069; }
        .json-value.number { color: #0550ae; }
        .json-type-hint { color: #8b949e; font-size: 0.75rem; }
        .json-children { display: none; padding-left: 16px; border-left: 1px dashed #dee2e6; margin-left: 6px; }
        .json-children.show { display: block; }
        .bbox-toggle { padding: 2px 6px; border-radius: 3px; font-size: 0.65rem; font-weight: 500; border: 1px solid; cursor: pointer; margin-left: auto; background: white; }
        .bbox-toggle:hover { opacity: 0.8; }
        .bbox-toggle.active { color: white !important; }
        `;
    }

    // Public API
    return {
        buildExtractionTree,
        toggleNode,
        escapeHtml,
        getStyles
    };
})();

// Export for module environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = JsonTreeRenderer;
}
