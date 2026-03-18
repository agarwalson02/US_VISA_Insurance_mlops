from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trianed_file_path:str
    test_file_path:str
