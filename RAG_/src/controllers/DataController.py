from .BaseController import BaseController
from .ProjectController import ProjectController
from fastapi import UploadFile,File
from models.enums import ResponseEnums
import os
import re

# Define DataController class that inherits from BaseController
class DataController(BaseController):
    # Initialize the DataController class
    def __init__(self):
        # Call the parent class constructor
        super().__init__()
        # Define size converter constant (1MB in bytes)
        self.size_converter = 1048576

    
    # Asynchronous method to validate uploaded files
    async def validate_file(self,file:UploadFile = File(...)):
        
        # Check if the file type is supported
        if file.content_type not in self.app_settings.FILE_TYPE:
            # Return false and error message if file type not supported
            return False,ResponseEnums.FILE_TYPE_NOT_SUPPORTED.value
        
        # Check if the file size exceeds the maximum allowed size
        if file.size > (self.app_settings.FILE_SIZE * self.size_converter):
            # Return false and error message if file size is too large
            return False,ResponseEnums.FILE_SIZE_TOO_LARGE.value
        
        # Return true and success message if file passes all validations
        return True,ResponseEnums.FILE_UPLOADED_SUCCESSFULLY.value
    
    
    
    # Method to generate a unique filename for uploaded files
    def generate_unique_filename(self,org_filename:str,project_id:str):
        
        # Generate a random string to make filename unique
        random_key = self.generate_random_string()
        
        # Create an instance of ProjectController
        project_controller = ProjectController()
        # Get the project directory path
        project_dir = project_controller.get_project_dir(project_id=project_id)
        
        # Clean the original filename
        cleaned_file_name = self.get_clean_file_name(org_filename)
        
        # Create a new filename by joining project directory, random key and cleaned filename
        new_filename = os.path.join(
            project_dir,
            random_key + "_" + cleaned_file_name
        )
        
        # Check if the filename already exists and generate a new one if it does
        while os.path.exists(new_filename):
            # Generate a new random key
            random_key = self.generate_random_string()
            # Create a new filename with the new random key
            new_filename = os.path.join(
                project_dir,
                random_key + "_" + cleaned_file_name
            )
        
        # Return the unique filename
        return new_filename, random_key + "_" + cleaned_file_name
    
    
    
    # Method to clean a filename by removing special characters
    def get_clean_file_name(self, orig_file_name: str):

        # Remove any special characters, except underscore and period
        cleaned_file_name = re.sub(r'[^\w.]', '', orig_file_name.strip())

        # Replace spaces with underscore
        cleaned_file_name = cleaned_file_name.replace(" ", "_")

        # Return the cleaned filename
        return cleaned_file_name
