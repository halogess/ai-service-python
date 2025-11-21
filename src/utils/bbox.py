from dataclasses import dataclass

@dataclass
class BBox:
    left: float
    top: float
    right: float
    bottom: float
    
    @property
    def width(self):
        return self.right - self.left
    
    @property
    def height(self):
        return self.bottom - self.top
    
    def overlaps(self, other: 'BBox') -> bool:
        return not (self.right < other.left or other.right < self.left or 
                   self.bottom < other.top or other.bottom < self.top)
    
    def distance_to(self, other: 'BBox') -> float:
        dx = max(0, max(self.left - other.right, other.left - self.right))
        dy = max(0, max(self.top - other.bottom, other.top - self.bottom))
        return (dx**2 + dy**2)**0.5