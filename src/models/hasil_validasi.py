from sqlalchemy import Column, Integer, String, Enum, Float, DateTime, ForeignKey, func
from db import Base

class HasilValidasiVisual(Base):
    __tablename__ = "hasil_validasi_visual"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    antrian_id = Column(Integer, ForeignKey('antrian.antrian_id'))
    tipe_target = Column(Enum('dokumen', 'bab', 'buku'))
    target_id = Column(Integer)
    total_error = Column(Integer)
    skor_visual = Column(Float)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())