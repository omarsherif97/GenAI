from pydantic import BaseModel, Field, field_validator
from bson.objectid import ObjectId
from typing import Optional 



class Project(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id") 
    project_id:str = Field(..., min_length=1)
    
    
    @field_validator("project_id")
    def validate_project_id(cls, v):
        if not v.isalnum():
            raise ValueError("Invalid project ID")
        return v
    
    class Config:
        arbitrary_types_allowed = True
    
