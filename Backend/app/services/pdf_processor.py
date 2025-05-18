import fitz  # PyMuPDF
import os
import re
import logging
from datetime import datetime
from app import db
from app.models.document import Document
from app.models.chapter import Chapter

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Service for processing PDF documents."""
    
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
    
    def save_uploaded_file(self, file, title=None):
        """
        Save an uploaded file to the upload folder and create a Document record.
        
        Args:
            file: The uploaded file object
            title: Optional title, defaults to filename if not provided
            
        Returns:
            Document: The created Document instance
        """
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(self.upload_folder, filename)
        
        # Save the file
        file.save(file_path)
        
        # Create document record
        document_title = title or os.path.splitext(file.filename)[0]
        document = Document(
            title=document_title,
            filename=filename,
            file_path=file_path,
            file_size=os.path.getsize(file_path),
            processed=False
        )
        
        try:
            # Get page count
            doc = fitz.open(file_path)
            document.page_count = len(doc)
            doc.close()
            
            # Save to database
            db.session.add(document)
            db.session.commit()
            
            logger.info(f"Document saved: {document.title}")
            return document
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving document: {str(e)}")
            # Clean up the file if there was an error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise
    
    def extract_chapters(self, document_id):
        """
        Extract chapters from a document.
        
        Args:
            document_id: The ID of the document to process
            
        Returns:
            list: The extracted chapters
        """
        document = Document.query.get(document_id)
        if not document:
            raise ValueError(f"Document with ID {document_id} not found")
        
        try:
            # Open the PDF
            doc = fitz.open(document.file_path)
            
            # Find chapters
            chapters = []
            seen_chapters = {}
            chapter_pattern = re.compile(r"^(Chapter\s+\d+(:\s+.+)?)$", re.IGNORECASE)
            
            last_chapter = None
            last_page_num = None
            
            # Iterate through pages to find chapter headings
            for page_num in range(len(doc)):
                text = doc[page_num].get_text("text")
                lines = text.split("\n")
                
                for line in lines:
                    line = line.strip()
                    if chapter_pattern.match(line):
                        logger.debug(f"Found chapter match on page {page_num+1}: '{line}'")
                        
                        # Handle special cases for chapter titles
                        if ": " in line and last_chapter and last_page_num == page_num - 1:
                            short_title = last_chapter
                            if short_title in seen_chapters and not seen_chapters[short_title][1] > seen_chapters[short_title][0]:
                                del seen_chapters[short_title]
                                last_chapter = None
                                
                        if last_chapter and last_chapter in seen_chapters:
                            seen_chapters[last_chapter][1] = page_num
                            
                        if line not in seen_chapters:
                            seen_chapters[line] = [page_num + 1, page_num + 1]
                            
                        last_chapter = line
                        last_page_num = page_num
            
            # Handle the last chapter
            if last_chapter is not None:
                seen_chapters[last_chapter][1] = len(doc)
            
            # Sort chapters by page number
            chapter_list = sorted(
                [{"title": title, "pages": [pages[0], pages[1]]} for title, pages in seen_chapters.items()],
                key=lambda x: x["pages"][0]
            )
            
            # Add chapters to database
            for idx, ch_data in enumerate(chapter_list):
                chapter = Chapter(
                    document_id=document.id,
                    title=ch_data["title"],
                    start_page=ch_data["pages"][0],
                    end_page=ch_data["pages"][1],
                    sequence=idx + 1
                )
                db.session.add(chapter)
                chapters.append(chapter)
            
            # Update document status
            document.processed = True
            db.session.commit()
            
            logger.info(f"Extracted {len(chapters)} chapters from document {document.id}")
            doc.close()
            return chapters
            
        except Exception as e:
            db.session.rollback()
            document.processing_error = str(e)
            db.session.commit()
            logger.error(f"Error extracting chapters: {str(e)}")
            raise