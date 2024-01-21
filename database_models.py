from sqlmodel import Field, SQLModel
from datetime import datetime


# Define the database model
class ConversationHistory(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default=datetime.now())
    user_id: str
    user_input: str
    agent_output: str
