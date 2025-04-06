from .BaseDataModel import BaseDataModel
from .enums.DB_Enums import DB_Collections
from .db_schemas.chunkData import DataChunk 
from helpers.config import get_settings, Settings
from bson.objectid import ObjectId
from pymongo import InsertOne

class ChunkModel(BaseDataModel):
    app_settings:Settings = get_settings()
    def __init__(self,db_client:object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client[self.app_settings.MONGODB_DATABASE][DB_Collections.COLLECTION_CHUNK_NAME.value]
        
    async def create_chunk(self,chunk:DataChunk)->DataChunk:
        try:
            result = await self.collection.insert_one(chunk.model_dump(by_alias=True, exclude_unset=True))
            chunk._id = result.inserted_id
            return chunk
        except Exception as e:
            raise e
    async def get_chunk_by_id(self,chunk_id:str)->DataChunk:
        try:
            chunk = await self.collection.find_one({"_id":ObjectId(chunk_id)})
            return DataChunk(**chunk)
        except Exception as e:
            raise e

    async def insert_many_chunks(self, chunks: list, batch_size: int=100):

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]

            operations = [
                InsertOne(chunk.dict(by_alias=True, exclude_unset=True))
                for chunk in batch
            ]

            await self.collection.bulk_write(operations)
        
        return len(chunks)

    async def delete_chunks_by_project_id(self, project_id: ObjectId):
        result = await self.collection.delete_many({
            "chunk_project_id": project_id
        })

        return result.deleted_count