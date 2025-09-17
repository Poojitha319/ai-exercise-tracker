from sqlalchemy import Column, Integer, String, Boolean
from app.database import Base

class Exercise(Base):
    __tablename__ = "exercises"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    source_url = Column(String, nullable=True)
    is_predefined = Column(Boolean, default=False)
