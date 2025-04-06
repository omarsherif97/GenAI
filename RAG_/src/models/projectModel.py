from .BaseDataModel import BaseDataModel
from .enums.DB_Enums import DB_Collections
from .db_schemas.project import Project
from helpers.config import get_settings, Settings


class ProjectModel(BaseDataModel):
    app_settings:Settings = get_settings()
    def __init__(self,db_client:object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client[self.app_settings.MONGODB_DATABASE][DB_Collections.COLLECTION_PROJECT_NAME.value]
        
        
    async def create_project(self,project:Project)->Project:
        try:
            result = await self.collection.insert_one(project.model_dump(by_alias=True, exclude_unset=True))
            project._id = result.inserted_id
            return project
        except Exception as e:
            raise e
        
    async def get_project_by_id(self,project_id:str)->Project:
        try:
            project = await self.collection.find_one({"project_id":project_id}) #filter by project_id
            if not project:
                #create a new project
                project = Project(project_id=project_id)
                return await self.create_project(project)
            #convert the bson object to a pydantic object
            return Project(**project)
        except Exception as e:
            raise e

            
        
        
        
    
        
        
        
