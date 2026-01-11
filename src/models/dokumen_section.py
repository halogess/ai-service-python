from sqlalchemy import Column, Integer, String, Boolean, Enum
from database import Base

class DokumenSection(Base):
    __tablename__ = "dokumen_section"
    
    dsec_id = Column(Integer, primary_key=True, autoincrement=True)
    dokumen_id = Column(Integer, nullable=True)
    dsec_index = Column(Integer, nullable=True)
    dsec_type = Column(String(50), nullable=True)
    dsec_has_title_page = Column(Boolean, nullable=False, default=False)
    dsec_different_odd_even = Column(Boolean, nullable=False, default=False)
    dsec_page_num_format = Column(String(32), nullable=True)
    dsec_page_num_start = Column(Integer, nullable=True)
    dsec_page_width_twips = Column(Integer, nullable=True)
    dsec_page_height_twips = Column(Integer, nullable=True)
    dsec_orientation = Column(Enum('portrait', 'landscape'), nullable=True)
    dsec_margin_top_twips = Column(Integer, nullable=True)
    dsec_margin_bottom_twips = Column(Integer, nullable=True)
    dsec_margin_left_twips = Column(Integer, nullable=True)
    dsec_margin_right_twips = Column(Integer, nullable=True)
    dsec_header_margin_twips = Column(Integer, nullable=True)
    dsec_footer_margin_twips = Column(Integer, nullable=True)
    dsec_gutter_twips = Column(Integer, nullable=True)
    dsec_gutter_position = Column(Enum('top', 'left'), nullable=True)
    dsec_column_count = Column(Integer, nullable=True)
