import numpy as np
from app import db
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy import LargeBinary

class Chunk(db.Model):
    """Chunk model for storing text segments and embeddings."""
    __tablename__ = 'chunks'
    
    id = db.Column(db.Integer, primary_key=True)
    chapter_id = db.Column(db.Integer, db.ForeignKey('chapters.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    page_number = db.Column(db.Integer, nullable=True)
    sequence = db.Column(db.Integer, nullable=False)  # Order in the chapter
    
    # Embedding as a vector (using ARRAY for PostgreSQL with pgvector)
    # When using SQLite during development, this will fall back to LargeBinary
    try:
        embedding = db.Column(ARRAY(db.Float), nullable=True)
    except Exception:
        # Fallback for SQLite or when pgvector is not available
        embedding = db.Column(LargeBinary, nullable=True)
    
    def __repr__(self):
        return f'<Chunk {self.id} (Chapter: {self.chapter_id})>'
    
    def set_embedding(self, embedding_np):
        """Set embedding from numpy array."""
        if embedding_np is not None:
            if isinstance(self.embedding, list) or self.embedding is None:
                # PostgreSQL ARRAY column
                self.embedding = embedding_np.tolist()
            else:
                # Binary storage for SQLite
                self.embedding = embedding_np.tobytes()
    
    def get_embedding(self):
        """Get embedding as numpy array."""
        if self.embedding is None:
            return None
            
        if isinstance(self.embedding, list):
            # PostgreSQL ARRAY column
            return np.array(self.embedding, dtype=np.float32)
        else:
            # Binary storage for SQLite
            return np.frombuffer(self.embedding, dtype=np.float32)