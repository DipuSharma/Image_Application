from db.db_config import Base, engine
from sqlalchemy import Column, Float, Integer, String, Boolean, ForeignKey, Table
from sqlalchemy.orm import relationship


class UploadImage(Base):
    __tablename__ = 'upload_image'
    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, nullable=True)

class VerifiedImage(Base):
    __tablename__ = 'verified_image'
    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, nullable=True)


Base.metadata.create_all(engine)