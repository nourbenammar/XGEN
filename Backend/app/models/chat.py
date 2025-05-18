# File: /Users/zrx/Desktop/XGen/app/models/chat.py
# VERSION WITH dqn2_training_context REMOVED and PRINT DEBUGGING

from datetime import datetime
from app import db
import numpy as np
import json # Added json import
import logging # Added logging import

# Get a logger instance for non-debug info
logger = logging.getLogger(__name__)

class ChatSession(db.Model):
    """Chat session model for tracking student interactions."""
    __tablename__ = 'chat_sessions'

    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('documents.id'), nullable=False)
    user_identifier = db.Column(db.String(255), nullable=False)
    session_name = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    document = db.relationship('Document', backref=db.backref('chat_sessions', cascade='all, delete-orphan'))
    messages = db.relationship('ChatMessage', backref='chat_session',
                             lazy='dynamic', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<ChatSession {self.id}: {self.session_name}>'


class ChatMessage(db.Model):
    """Chat message model for storing messages in a session."""
    __tablename__ = 'chat_messages'

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id'), nullable=False)
    is_user = db.Column(db.Boolean, default=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    error = db.Column(db.Boolean, default=False)
    embedding = db.Column(db.LargeBinary, nullable=True)
    ppo_memory_data = db.Column(db.Text, nullable=True) # Store serialized JSON PPO data
    # --- REMOVED dqn2_training_context ---
    # dqn2_training_context = db.Column(db.Text, nullable=True) # Removed this line
    # --- END REMOVAL ---
    relevance_feedback = db.Column(db.Boolean, nullable=True)
    clarity_feedback = db.Column(db.Boolean, nullable=True)
    length_feedback = db.Column(db.String(10), nullable=True)

    source_mappings = db.relationship('SourceMapping', backref='message',
                                    lazy='dynamic', cascade='all, delete-orphan')
    irrelevant_chunks = db.relationship('IrrelevantChunk', backref='message',
                                      lazy='dynamic', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<ChatMessage {self.id}: {"User" if self.is_user else "System"}>'

    def set_embedding(self, embedding_np):
        """Set embedding from numpy array."""
        if embedding_np is not None:
            if embedding_np.dtype != np.float32:
                 embedding_np = embedding_np.astype(np.float32)
            self.embedding = embedding_np.tobytes()

    def get_embedding(self):
        """Get embedding as numpy array."""
        if self.embedding is None:
            return None
        try:
            return np.frombuffer(self.embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error converting embedding blob to numpy array for message {self.id}: {e}", exc_info=True)
            return None

    def set_ppo_data(self, state, action_tanh, log_prob, value):
        """Serialize and store PPO data needed for feedback."""
        if state is None or action_tanh is None or log_prob is None or value is None:
             print(f">>> CHAT_MODEL (set_ppo_data): Attempted to set PPO data with None values for message {self.id}. Skipping.") # DEBUG PRINT
             self.ppo_memory_data = None
             return

        try:
            print(f">>> CHAT_MODEL (set_ppo_data): Serializing PPO data for message {self.id}: state_type={type(state)}, action_type={type(action_tanh)}, log_prob={log_prob}, value={value}") # DEBUG PRINT
            if isinstance(state, np.ndarray): print(f"  State shape: {state.shape}") # DEBUG PRINT
            if isinstance(action_tanh, np.ndarray): print(f"  Action shape: {action_tanh.shape}") # DEBUG PRINT

            state_list = state.tolist() if isinstance(state, np.ndarray) else list(state)
            action_list = action_tanh.tolist() if isinstance(action_tanh, np.ndarray) else list(action_tanh)

            data_to_store = {
                "state": state_list,
                "action_tanh": action_list,
                "log_prob": float(log_prob),
                "value": float(value)
            }
            self.ppo_memory_data = json.dumps(data_to_store)
            print(f">>> CHAT_MODEL (set_ppo_data): Successfully serialized PPO data for message {self.id}. JSON preview: {self.ppo_memory_data[:200]}...") # DEBUG PRINT
        except Exception as e:
            logger.error(f"Error serializing PPO data for message {self.id}: {e}", exc_info=True)
            print(f">>> CHAT_MODEL (set_ppo_data): ERROR serializing PPO data for message {self.id}: {e}") # DEBUG PRINT
            self.ppo_memory_data = None

    def get_ppo_data(self):
        """Deserialize PPO data."""
        print(f">>> CHAT_MODEL (get_ppo_data): Getting PPO data for message {self.id}. Raw DB value: {self.ppo_memory_data[:200] if self.ppo_memory_data else 'None'}") # DEBUG PRINT
        if not self.ppo_memory_data:
            return None
        try:
            data = json.loads(self.ppo_memory_data)
            print(f">>> CHAT_MODEL (get_ppo_data): JSON deserialization successful for message {self.id}. Keys: {list(data.keys())}") # DEBUG PRINT
            if not all(k in data for k in ["state", "action_tanh", "log_prob", "value"]):
                 print(f">>> CHAT_MODEL (get_ppo_data): WARNING - Deserialized PPO data for message {self.id} is missing required keys.") # DEBUG PRINT
                 return None
            return data
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error deserializing PPO data for message {self.id}: {e}", exc_info=True)
            print(f">>> CHAT_MODEL (get_ppo_data): JSON Decode Error for message {self.id}: {e}") # DEBUG PRINT
            return None
        except Exception as e:
            logger.error(f"Unexpected error deserializing PPO data for message {self.id}: {e}", exc_info=True)
            print(f">>> CHAT_MODEL (get_ppo_data): Unexpected Error for message {self.id}: {e}") # DEBUG PRINT
            return None


class SourceMapping(db.Model):
    """Source mapping model for linking answer sentences to source chunks."""
    __tablename__ = 'source_mappings'

    id = db.Column(db.Integer, primary_key=True)
    message_id = db.Column(db.Integer, db.ForeignKey('chat_messages.id'), nullable=False)
    sentence = db.Column(db.Text, nullable=False)
    chunk_id = db.Column(db.Integer, db.ForeignKey('chunks.id'), nullable=False)

    chunk = db.relationship('Chunk')

    def __repr__(self):
        return f'<SourceMapping {self.id}: Message {self.message_id} to Chunk {self.chunk_id}>'


class IrrelevantChunk(db.Model):
    """Model for tracking chunks marked as irrelevant during feedback."""
    __tablename__ = 'irrelevant_chunks'

    id = db.Column(db.Integer, primary_key=True)
    message_id = db.Column(db.Integer, db.ForeignKey('chat_messages.id'), nullable=False)
    chunk_id = db.Column(db.Integer, db.ForeignKey('chunks.id'), nullable=False)

    chunk = db.relationship('Chunk')

    def __repr__(self):
        return f'<IrrelevantChunk {self.id}: Message {self.message_id}, Chunk {self.chunk_id}>'