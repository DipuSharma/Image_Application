from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from typing import Generator

Database_url = "sqlite:///./image_db.db"

engine = create_engine(Database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=True, bind=engine)

Base = declarative_base()

def get_db() -> Generator:
    global db
    try:
        db=SessionLocal()
        yield db
    finally:
        db.close()

        