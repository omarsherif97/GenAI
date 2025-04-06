from fastapi import APIRouter, Depends, UploadFile, HTTPException, status, File, Request
from helpers.config import get_settings, Settings
from controllers import DataController,ProjectController,ProcessingController
from fastapi.responses import JSONResponse
import os
import aiofiles
from models.enums import ResponseEnums
import logging
from .schema import ProcessRequest
from models.projectModel import ProjectModel
from models.db_schemas.project import Project
from models.ChunkModel import ChunkModel
from models.db_schemas.chunkData import DataChunk

logger = logging.getLogger("uvicorn.error")





data_router = APIRouter(
    prefix="/api/v1/data",
)


@data_router.post("/upload/{project_id}")
async def upload_data(
    request: Request,
    project_id: str,
    file: UploadFile = File(...),
    app_settings: Settings = Depends(get_settings)
    ):
    
    project_model = ProjectModel(db_client=request.app.state.db_client)
    project = Project(project_id=project_id)
    project = await project_model.get_project_by_id(project_id=project_id)
    
    
    # validate file type and size
    data_controller = DataController()
    is_valid,message = await data_controller.validate_file(file=file)

    
    # Process the file here
    # You can add code to save the file or process its contents
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
            "message":message
        }
    )
    else:
        
        file_path,file_name = data_controller.generate_unique_filename(
            org_filename=file.filename,
            project_id=project_id
            )
        
        try:
            async with aiofiles.open(file_path, "wb") as f:
                while chunk := await file.read(app_settings.FILE_MAX_CHUNK_SIZE):
                    await f.write(chunk)
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "message":ResponseEnums.FILE_UPLOAD_FAILED.value,
                }
            )
            
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message":ResponseEnums.FILE_UPLOADED_SUCCESSFULLY.value,
                "file_path":file_path,
                "file_name":file_name,
                "project_id":project_id,
                "_id":str(project._id)
            }
        )

@data_router.post("/process/{project_id}")
async def process_data(
    request: Request, 
    project_id: str,
    process_request: ProcessRequest,
    app_settings: Settings = Depends(get_settings)
):
    file_id = process_request.file_id
    chunk_size = process_request.chunk_size
    overlap_size = process_request.chunk_overlap
    do_reset = process_request.do_reset

    project_model = ProjectModel(
        db_client=request.app.state.db_client
    )

    project = await project_model.get_project_by_id(
        project_id=project_id
    )

    process_controller = ProcessingController(project_id=project_id)

    file_content = process_controller.get_file_content(file_id=file_id)

    file_chunks = process_controller.process_file(
        file_content=file_content,
        file_id=file_id,
        chunk_size=chunk_size,
        chunk_overlap=overlap_size
    )

    if file_chunks is None or len(file_chunks) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseEnums.PROCESSING_FAILED.value
            }
        )

    file_chunks_records = [
        DataChunk(
            chunk_text=chunk.page_content,
            chunk_metadata=chunk.metadata,
            chunk_order=i+1,
            chunk_project_id=project.id,
        )
        for i, chunk in enumerate(file_chunks)
    ]

    chunk_model = ChunkModel(
        db_client=request.app.state.db_client
    )

    if do_reset == 1:
        _ = await chunk_model.delete_chunks_by_project_id(
            project_id=project.id
        )

    no_records = await chunk_model.insert_many_chunks(chunks=file_chunks_records)

    return JSONResponse(
        content={
            "signal": ResponseEnums.PROCESSING_SUCCESS.value,
            "inserted_chunks": no_records
        }
    )
