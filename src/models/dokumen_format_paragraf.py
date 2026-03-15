from sqlalchemy import Column, Integer, String
from database import Base


class DokumenFormatParagraf(Base):
    __tablename__ = "dokumen_format_paragraf"

    dfp_id = Column(Integer, primary_key=True, autoincrement=True)
    dfp_p_style_id = Column(String(128), nullable=True)
    dfp_jc = Column(String(10), nullable=True)
