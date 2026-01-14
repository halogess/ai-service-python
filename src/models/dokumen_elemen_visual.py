from sqlalchemy import Column, Integer, String, BigInteger, Float, Text
from database import Base

class DokumenElemenVisual(Base):
    __tablename__ = "dokumen_elemen_visual"

    dev_id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    dokumen_id = Column(Integer)
    dev_page = Column(Integer)
    dokumen_elemen_id = Column(BigInteger)
    dev_bbox_x0 = Column(Float)
    dev_bbox_y0 = Column(Float)
    dev_bbox_x1 = Column(Float)
    dev_bbox_y1 = Column(Float)
    dev_label = Column(String(50))
    dev_text = Column(Text)

    def __repr__(self):
        return f"<DokumenElemenVisual(id={self.dev_id}, doc={self.dokumen_id}, page={self.dev_page})>"
