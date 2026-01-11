"""Phase 4: Global container cleanup - Force strict bbox from children"""


def cleanup_container_bboxes(final_aligned, log_file=None):
    """Refine container bboxes to strictly fit children on same page"""
    
    # Gap threshold for Y-clustering (prevents including non-table content)
    GAP_THRESHOLD = 150.0  # ~5 rows worth of gap
    
    # Map table_id -> page -> [child_bboxes]
    table_children_bboxes = {}
    for item in final_aligned:
        pid = item.get('parent_element_id')
        page = item.get('page')
        if pid and item.get('bbox') and page is not None:
            pid_str = str(pid)
            if pid_str not in table_children_bboxes:
                table_children_bboxes[pid_str] = {}
            
            if page not in table_children_bboxes[pid_str]:
                table_children_bboxes[pid_str][page] = []
                
            table_children_bboxes[pid_str][page].append(item['bbox'])
    
    # Update Containers
    for item in final_aligned:
        if item.get('is_table_container') or (item.get('text', '').startswith('Table ') and not item.get('parent_element_id')):
            eid_str = str(item.get('element_id'))
            page = item.get('page')
            
            if page is None:
                continue
            
            if eid_str in table_children_bboxes and page in table_children_bboxes[eid_str]:
                children_bboxes = table_children_bboxes[eid_str][page]
                
                if children_bboxes:
                    # NEW: Cluster by Y position to avoid including non-table content
                    if len(children_bboxes) > 1:
                        sorted_bboxes = sorted(children_bboxes, key=lambda b: b['y0'])
                        clusters = []
                        current_cluster = [sorted_bboxes[0]]
                        
                        for i in range(1, len(sorted_bboxes)):
                            prev_y1 = sorted_bboxes[i-1]['y1']
                            curr_y0 = sorted_bboxes[i]['y0']
                            gap = curr_y0 - prev_y1
                            
                            if gap > GAP_THRESHOLD:
                                clusters.append(current_cluster)
                                current_cluster = []
                            current_cluster.append(sorted_bboxes[i])
                        
                        clusters.append(current_cluster)
                        
                        # Use largest cluster for container bbox
                        if len(clusters) > 1:
                            main_cluster = max(clusters, key=len)
                            if log_file:
                                log_file.write(f"Container {item['element_id']}: split into {len(clusters)} clusters, using main with {len(main_cluster)} items\n")
                            children_bboxes = main_cluster
                    
                    new_x0 = min(b['x0'] for b in children_bboxes)
                    new_y0 = min(b['y0'] for b in children_bboxes)
                    new_x1 = max(b['x1'] for b in children_bboxes)
                    new_y1 = max(b['y1'] for b in children_bboxes)
                    
                    if log_file:
                        old_h = item['bbox']['y1'] - item['bbox']['y0']
                        new_h = new_y1 - new_y0
                        if abs(old_h - new_h) > 10:
                            log_file.write(f"Refining Container {item['element_id']} on page {page}: H {old_h:.1f} -> {new_h:.1f}\n")
                    
                    item['bbox'] = {"x0": new_x0, "y0": new_y0, "x1": new_x1, "y1": new_y1}

