from datetime import datetime
from app import db

class Document(db.Model):
    """Document model for storing uploaded files metadata."""
    __tablename__ = 'documents'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    filename = db.Column(db.String(255), nullable=False, unique=True)
    file_path = db.Column(db.String(512), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)  # Size in bytes
    page_count = db.Column(db.Integer, nullable=True)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    processed = db.Column(db.Boolean, default=False)
    processing_error = db.Column(db.Text, nullable=True)
    
    # Relationships
    chapters = db.relationship('Chapter', backref='document', lazy='dynamic', 
                               cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Document {self.title}>'
    
    @property
    def processing_status(self):
        """Return document processing status."""
        if self.processing_error:
            return 'error'
        elif self.processed:
            return 'processed'
        else:
            return 'pending'