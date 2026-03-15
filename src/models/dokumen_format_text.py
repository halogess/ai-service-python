from sqlalchemy import Column, Integer, String
from database import Base


class DokumenFormatText(Base):
    __tablename__ = "dokumen_format_text"

    dftx_id = Column(Integer, primary_key=True, autoincrement=True)
    dftx_font_ascii = Column(String(128), nullable=True)
    dftx_bold = Column(Integer, nullable=True)
    dftx_italic = Column(Integer, nullable=True)
    dftx_underline = Column(String(10), nullable=True)
