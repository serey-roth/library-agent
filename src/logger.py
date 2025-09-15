import logging
import os
import sys
from typing import Optional


class AppLogger:
    def __init__(self, name: str = "library_agent", debug: Optional[bool] = None):
        self.logger = logging.getLogger(name)
        
        if debug is None:
            debug = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
        
        level = logging.DEBUG if debug else logging.INFO
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(console_handler)
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        self.logger.critical(message, extra=kwargs)

logger = AppLogger()
