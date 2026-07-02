import os

class ResourceLocator:
    @staticmethod
    def find_resource(name):
        if os.path.exists(name):
            return os.path.abspath(name)
        return None
