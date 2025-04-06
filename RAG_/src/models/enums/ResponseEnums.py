from enum import Enum

class ResponseEnums(Enum):
    FILE_TYPE_NOT_SUPPORTED = "File type not supported"
    FILE_SIZE_TOO_LARGE = "File size too large"
    FILE_UPLOADED_SUCCESSFULLY = "File uploaded successfully"
    FILE_UPLOADED_FAILED = "File uploaded failed"
    PROCESSING_FAILED = "Processing failed"
    PROCESSING_SUCCESS = "Processing success"

