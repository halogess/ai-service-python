/**
 * PyMuPDF Rawdict Character Grouping Module
 * Detects overlapping characters based on X-coordinate and groups them together
 */

// Constants
const X_TOLERANCE = 3;  // X tolerance in DPI (PDF points at 72 DPI)
/**
 * Collect all characters from rawdict data into a flat array
 * @param {Object} data - The rawdict extraction data with blocks
 * @returns {Array} Flat array of char objects with position indices
 */
function collectAllChars(data) {
    const chars = [];
    const blocks = data?.blocks || [];

    blocks.forEach((block, blockIdx) => {
        const lines = block.lines || [];
        lines.forEach((line, lineIdx) => {
            const spans = line.spans || [];
            spans.forEach((span, spanIdx) => {
                const spanChars = span.chars || [];
                spanChars.forEach((char, charIdx) => {
                    const charText = char.c || '';

                    // Skip empty or whitespace-only characters
                    if (!charText || charText.trim() === '') {
                        return;
                    }

                    if (char.bbox && char.bbox.length >= 4) {
                        chars.push({
                            c: charText,
                            bbox: char.bbox,
                            blockIdx,
                            lineIdx,
                            spanIdx,
                            charIdx,
                            originalChar: char
                        });
                    }
                });
            });
        });
    });

    return chars;
}

/**
 * Check if bbox2's x0 is within bbox1's x range (x0 to x1) with tolerance
 * @param {Array} bbox1 - [x0, y0, x1, y1] - the reference bbox
 * @param {Array} bbox2 - [x0, y0, x1, y1] - the bbox to check
 * @param {number} tolerance - tolerance in PDF points (72 DPI, 1 pt = 1/72 inch ≈ 0.35mm)
 * @returns {boolean}
 */
function xOverlap(bbox1, bbox2, tolerance = 3) {
    // bbox2.x0 must be within [bbox1.x0 - tolerance, bbox1.x1 + tolerance]
    return bbox2[0] >= (bbox1[0] - tolerance) && bbox2[0] <= (bbox1[2] + tolerance);
}

const Y_OVERLAP_MIN_RATIO = 0.3;  // Minimum 30% overlap of min height required

/**
 * Check if bbox2 has at least 50% Y overlap with bbox1 (based on minimum height)
 * @param {Array} bbox1 - [x0, y0, x1, y1] - the reference bbox
 * @param {Array} bbox2 - [x0, y0, x1, y1] - the bbox to check
 * @returns {boolean}
 */
function yOverlap(bbox1, bbox2) {
    // Calculate heights
    const height1 = bbox1[3] - bbox1[1];
    const height2 = bbox2[3] - bbox2[1];
    const minHeight = Math.min(height1, height2);

    if (minHeight <= 0) return false;

    // Calculate Y overlap amount
    const overlapStart = Math.max(bbox1[1], bbox2[1]);
    const overlapEnd = Math.min(bbox1[3], bbox2[3]);
    const overlapAmount = Math.max(0, overlapEnd - overlapStart);

    // Check if overlap is at least 50% of min height
    const overlapRatio = overlapAmount / minHeight;
    return overlapRatio >= Y_OVERLAP_MIN_RATIO;
}

/**
 * Check if two bboxes overlap based on user's criteria:
 * - X: bbox2.x0 must be within bbox1's x range (with tolerance)
 * - Y: bbox2.y0 OR bbox2.y1 must be within bbox1's y range
 * @param {Array} groupBbox - merged bbox of the group [x0, y0, x1, y1]
 * @param {Array} charBbox - bbox of character to check [x0, y0, x1, y1]
 * @param {number} xTol - X tolerance in PDF points
 * @returns {boolean}
 */
function isOverlapping(groupBbox, charBbox, xTol = 3) {
    return xOverlap(groupBbox, charBbox, xTol) && yOverlap(groupBbox, charBbox);
}

/**
 * Find all overlapping character groups using DFS (Depth-First Search)
 * Each character can find multiple overlapping neighbors
 * @param {Array} chars - Flat array of char objects from collectAllChars
 * @returns {Array} Array of groups, each with chars array and mergedBbox, sorted by Y
 */
function findOverlappingGroups(chars) {
    if (!chars || chars.length === 0) return [];

    // Sort by X first (left to right), then by Y (top to bottom)
    const sortedChars = [...chars].sort((a, b) => {
        const xDiff = a.bbox[0] - b.bbox[0];
        if (Math.abs(xDiff) > 2) return xDiff;
        return a.bbox[1] - b.bbox[1];
    });

    const groups = [];
    const processed = new Set();

    for (let i = 0; i < sortedChars.length; i++) {
        if (processed.has(i)) continue;

        // DFS with stack (LIFO instead of FIFO)
        const stack = [i];
        const group = {
            chars: [],
            mergedBbox: null,
            buildLog: []  // Track how group was built
        };

        while (stack.length > 0) {
            const currentIdx = stack.pop();  // DFS: pop from end (stack)
            if (processed.has(currentIdx)) continue;

            const currentChar = sortedChars[currentIdx];
            processed.add(currentIdx);
            group.chars.push(currentChar);

            // Track the blockIdx of the group (first char sets it)
            if (group.blockIdx === undefined) {
                group.blockIdx = currentChar.blockIdx;
            }

            // Log this char being added
            const logEntry = {
                char: currentChar.c,
                bbox: currentChar.bbox.map(v => v.toFixed(1)),
                foundNeighbors: []
            };

            // Update merged bbox
            if (!group.mergedBbox) {
                group.mergedBbox = [...currentChar.bbox];
            } else {
                group.mergedBbox[0] = Math.min(group.mergedBbox[0], currentChar.bbox[0]);
                group.mergedBbox[1] = Math.min(group.mergedBbox[1], currentChar.bbox[1]);
                group.mergedBbox[2] = Math.max(group.mergedBbox[2], currentChar.bbox[2]);
                group.mergedBbox[3] = Math.max(group.mergedBbox[3], currentChar.bbox[3]);
            }

            // Find all neighbors that overlap with merged bbox (expanded area)
            // IMPORTANT: Only allow merging characters from the SAME block
            for (let j = 0; j < sortedChars.length; j++) {
                if (processed.has(j) || stack.includes(j)) continue;

                const otherChar = sortedChars[j];

                // Skip if different blockIdx - prevents cross-block merging
                if (otherChar.blockIdx !== group.blockIdx) continue;

                // Check overlap with merged bbox (using X_TOLERANCE)
                if (isOverlapping(group.mergedBbox, otherChar.bbox, X_TOLERANCE)) {
                    stack.push(j);  // DFS: push to stack
                    logEntry.foundNeighbors.push({
                        char: otherChar.c,
                        bbox: otherChar.bbox.map(v => v.toFixed(1))
                    });
                }
            }

            group.buildLog.push(logEntry);
        }

        // Add groups with more than 1 character (actual overlaps)
        if (group.chars.length > 1) {
            // Sort chars by X position (left to right) before generating text
            group.chars.sort((a, b) => a.bbox[0] - b.bbox[0]);
            group.id = `chargroup_${groups.length}`;
            group.text = group.chars.map(c => c.c).join('');
            group.isSingle = false;
            groups.push(group);
        } else if (group.chars.length === 1) {
            // Singles (ungrouped chars) - will be added at the end
            group.id = `charsingle_${groups.length}`;
            group.text = group.chars[0].c;
            group.isSingle = true;
            groups.push(group);
        }
    }

    // Separate groups and singles
    const overlappingGroups = groups.filter(g => !g.isSingle);
    const singles = groups.filter(g => g.isSingle);

    // Sort overlapping groups by Y position (top to bottom)
    overlappingGroups.sort((a, b) => a.mergedBbox[1] - b.mergedBbox[1]);

    // Sort singles by Y position (top to bottom)
    singles.sort((a, b) => a.mergedBbox[1] - b.mergedBbox[1]);

    // Combine: groups first, then singles at the end
    const result = [...overlappingGroups, ...singles];

    // Re-assign IDs after sorting
    result.forEach((group, idx) => {
        group.id = group.isSingle ? `charsingle_${idx}` : `chargroup_${idx}`;
    });

    return result;
}

/**
 * Get the combined text of a character group
 * @param {Object} group - A group object with chars array
 * @returns {string}
 */
function getGroupText(group) {
    return group.chars.map(c => c.c).join('');
}

// Export for use in main script
window.CharGrouping = {
    collectAllChars,
    findOverlappingGroups,
    xOverlap,
    yOverlap,
    isOverlapping,
    getGroupText
};
