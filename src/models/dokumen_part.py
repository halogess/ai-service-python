from sqlalchemy import Column, Integer, String
from database import Base

class DokumenPart(Base):
    __tablename__ = "dokumen_part"
    
    dpart_id = Column(Integer, primary_key=True, autoincrement=True)
    dsec_id = Column(Integer, nullable=False)
    dpart_type = Column(String(20), nullable=False)
    dpart_position = Column(String(10), nullable=True)
