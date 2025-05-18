from app import db

class Chapter(db.Model):
    """Chapter model for storing document chapter data."""
    __tablename__ = 'chapters'
    
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('documents.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    start_page = db.Column(db.Integer, nullable=False)
    end_page = db.Column(db.Integer, nullable=False)
    sequence = db.Column(db.Integer, nullable=False)  # Order in the document
    
    # Relationships
    chunks = db.relationship('Chunk', backref='chapter', lazy='dynamic',
                            cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Chapter {self.title} (Doc: {self.document_id})>'