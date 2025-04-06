from helpers.config import get_settings,Settings
from fastapi import Depends
import os
import string
import random
class BaseController:
    def __init__(self):
        self.app_settings = get_settings()

        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.project_dir = os.path.join(self.base_dir, "assets/files")
        
    def generate_random_string(self,length:int=4):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

