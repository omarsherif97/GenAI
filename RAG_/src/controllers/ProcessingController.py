from .BaseController import BaseController
from .ProjectController import ProjectController
import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import TextLoader
from models.enums import ProcessingEnums
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class ProcessingController(BaseController):
    
    def __init__(self,project_id:str):
        super().__init__()
        self.project_id = project_id
        
        self.project_dir = ProjectController().get_project_dir(project_id=self.project_id)
        
        
    def get_file_extension(self,file_id:str):
        return os.path.splitext(file_id)[-1]        
    
    def get_file_loader(self,file_id:str):
        file_extension = self.get_file_extension(file_id=file_id)
        file_path = os.path.join(self.project_dir,file_id)
        
        if file_extension == ProcessingEnums.PDF.value:
            return PyMuPDFLoader(file_path)
        
        elif file_extension == ProcessingEnums.TEXT.value:
            return TextLoader(file_path,encoding="utf-8")
        
        else:
            return None
    
    def get_file_content(self,file_id:str):
        file_loader = self.get_file_loader(file_id=file_id)
        if file_loader is None:
            return None
        
        return file_loader.load()
    
    def process_file(
        self,file_id:str,
        file_content:list[Document],
        chunk_size:int=1000,
        chunk_overlap:int=200
        ):
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
            )
        
        
        file_content_text = [
            rec.page_content for rec in file_content
        ]
        
        file_metadata = [
            rec.metadata for rec in file_content
        ]
        
        chunks = text_splitter.create_documents(
            file_content_text,
            file_metadata
            )
        
        return chunks

