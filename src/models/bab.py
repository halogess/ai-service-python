from sqlalchemy import Column, Integer, String
from database import Base

class Bab(Base):
    __tablename__ = "bab"
    
    bab_id = Column(Integer, primary_key=True, autoincrement=True)
    buku_id = Column(Integer, nullable=False)
    bab_order = Column(Integer, nullable=True)
    bab_filename = Column(String(255), nullable=False)
    bab_docx_path = Column(String(255), nullable=True)
    bab_pdf_path = Column(String(255), nullable=True)
    bab_images_path = Column(String(255), nullable=True)
