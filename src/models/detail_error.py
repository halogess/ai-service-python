from sqlalchemy import Column, Integer, String, Enum, Float, Text, ForeignKey
from db import Base

class DetailErrorVisual(Base):
    __tablename__ = "detail_error_visual"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    hasil_id = Column(Integer, ForeignKey('hasil_validasi_visual.id'))
    kode = Column(String(100))
    pesan_singkat = Column(String(255))
    penjelasan = Column(Text)
    severity = Column(Enum('minor', 'major', 'critical'))
    page = Column(Integer)
    bbox_left = Column(Float)
    bbox_top = Column(Float)
    bbox_right = Column(Float)
    bbox_bottom = Column(Float)
    meta = Column(Text, nullable=True)