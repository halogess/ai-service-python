"""
Alignment Visualizer
Draws alignment bounding boxes on PDF page images
"""

import os
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger(__name__)


class AlignmentVisualizer:
    """Visualize alignment results by drawing bounding boxes on images"""
    
    def __init__(self):
        self.colors = {
            'aligned': '#00FF00',      # Green for aligned items
            'unaligned': '#FF0000',    # Red for unaligned items
            'header_footer': '#FFA500' # Orange for header/footer
        }
    
    def draw_alignments_on_page(self, image_path: str, alignment_result: dict, output_path: str, scale_factor: float = 300/72):
        """
        Draw alignment bounding boxes on a page image.
        
        Args:
            image_path: Path to the source image
            alignment_result: Alignment result dict for this page
            output_path: Path to save the annotated image
            scale_factor: Scale factor for coordinates (default 300/72 for 300 DPI)
        """
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"DRAWING ALIGNMENTS ON PAGE")
            logger.info(f"{'='*80}")
            logger.info(f"Image: {image_path}")
            logger.info(f"Output: {output_path}")
            logger.info(f"Scale factor: {scale_factor}")
            
            # Open image
            img = Image.open(image_path)
            logger.info(f"Image size: {img.size}, mode: {img.mode}")
            
            # Debug alignment result structure
            logger.info(f"\nAlignment result keys: {list(alignment_result.keys())}")
            
            alignments = alignment_result.get('alignments', [])
            unaligned = alignment_result.get('unaligned_pdf_units', [])
            header_footer = alignment_result.get('header_footer_units', [])
            
            logger.info(f"\nData counts:")
            logger.info(f"  - Alignments: {len(alignments)}")
            logger.info(f"  - Unaligned: {len(unaligned)}")
            logger.info(f"  - Header/Footer: {len(header_footer)}")
            
            if alignments:
                logger.info(f"\nFirst alignment sample:")
                first = alignments[0]
                logger.info(f"  Keys: {list(first.keys())}")
                logger.info(f"  merged_bbox: {first.get('merged_bbox')}")
                logger.info(f"  matched_pdf_units count: {len(first.get('matched_pdf_units', []))}")
                if first.get('matched_pdf_units'):
                    logger.info(f"  First unit bbox: {first['matched_pdf_units'][0].get('bbox')}")
            
            if unaligned:
                logger.info(f"\nUnaligned units sample:")
                logger.info(f"  Type: {type(unaligned[0])}")
                if isinstance(unaligned[0], dict):
                    logger.info(f"  Keys: {list(unaligned[0].keys())}")
                    logger.info(f"  bbox: {unaligned[0].get('bbox')}")
            
            draw = ImageDraw.Draw(img)
            
            boxes_drawn = 0
            boxes_attempted = 0
            
            # Draw aligned items (green boxes)
            logger.info(f"\nDrawing {len(alignments)} alignments...")
            for idx, alignment in enumerate(alignments):
                boxes_attempted += 1
                bbox = alignment.get('merged_bbox')
                if bbox:
                    logger.debug(f"  Alignment {idx}: merged_bbox={bbox}")
                    scaled_bbox = [coord * scale_factor for coord in bbox]
                    logger.debug(f"    Scaled: {scaled_bbox}")
                    self._draw_bbox(draw, scaled_bbox, self.colors['aligned'], 3)
                    boxes_drawn += 1
                    
                    # Draw individual matched units (thinner lines)
                    for unit in alignment.get('matched_pdf_units', []):
                        boxes_attempted += 1
                        unit_bbox = unit.get('bbox')
                        if unit_bbox:
                            scaled_unit_bbox = [coord * scale_factor for coord in unit_bbox]
                            self._draw_bbox(draw, scaled_unit_bbox, self.colors['aligned'], 1)
                            boxes_drawn += 1
                else:
                    logger.warning(f"  Alignment {idx}: NO merged_bbox!")
            
            # Draw unaligned items (red boxes)
            logger.info(f"\nDrawing {len(unaligned)} unaligned units...")
            if unaligned and isinstance(unaligned, list) and len(unaligned) > 0:
                # Check if it's a list of indices or list of dicts
                if isinstance(unaligned[0], dict):
                    for idx, unit in enumerate(unaligned):
                        boxes_attempted += 1
                        bbox = unit.get('bbox')
                        if bbox:
                            logger.debug(f"  Unaligned {idx}: bbox={bbox}")
                            scaled_bbox = [coord * scale_factor for coord in bbox]
                            self._draw_bbox(draw, scaled_bbox, self.colors['unaligned'], 3)
                            boxes_drawn += 1
                        else:
                            logger.warning(f"  Unaligned {idx}: NO bbox!")
            
            # Draw header/footer items (orange boxes)
            logger.info(f"\nDrawing {len(header_footer)} header/footer units...")
            if header_footer and isinstance(header_footer, list) and len(header_footer) > 0:
                if isinstance(header_footer[0], dict):
                    for idx, unit in enumerate(header_footer):
                        boxes_attempted += 1
                        bbox = unit.get('bbox')
                        if bbox:
                            logger.debug(f"  H/F {idx}: bbox={bbox}")
                            scaled_bbox = [coord * scale_factor for coord in bbox]
                            self._draw_bbox(draw, scaled_bbox, self.colors['header_footer'], 3)
                            boxes_drawn += 1
                        else:
                            logger.warning(f"  H/F {idx}: NO bbox!")
            
            logger.info(f"\nSUMMARY: Drew {boxes_drawn} / {boxes_attempted} bounding boxes")
            
            # Save annotated image
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only create directory if path has a directory component
                os.makedirs(output_dir, exist_ok=True)
            img.save(output_path, 'PNG')
            logger.info(f"Saved alignment visualization to: {output_path}")
            logger.info(f"{'='*80}\n")
            
        except Exception as e:
            logger.error(f"Error drawing alignments on {image_path}: {e}", exc_info=True)
            raise
    
    def _draw_bbox(self, draw: ImageDraw, bbox: list, color: str, width: int):
        """Draw a bounding box on the image"""
        if not bbox or len(bbox) < 4:
            logger.warning(f"Invalid bbox (empty or too short): {bbox}")
            return
        
        # Check for None values
        if any(coord is None for coord in bbox):
            logger.warning(f"Invalid bbox (contains None): {bbox}")
            return
        
        x0, y0, x1, y1 = bbox
        
        # Clamp coordinates to image bounds
        img_width, img_height = draw.im.size
        x0 = max(0, min(x0, img_width))
        y0 = max(0, min(y0, img_height))
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        
        # Skip if invalid
        if x1 <= x0 or y1 <= y0:
            logger.warning(f"Invalid bbox after clamp (x1<=x0 or y1<=y0): [{x0}, {y0}, {x1}, {y1}]")
            return
        
        logger.debug(f"Drawing bbox: [{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}] color={color}")
        try:
            draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        except Exception as e:
            logger.error(f"Error drawing rectangle: {e}, bbox=[{x0}, {y0}, {x1}, {y1}]")
