import os


class DirectContext:

    def __init__(self, direct_name):
        self.direct_name = direct_name
        self.origin_path = os.getcwd()

    def __enter__(self):
        os.chdir( self.origin_path + os.altsep + self.direct_name )

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir( self.origin_path )

        if exc_val:
            raise
