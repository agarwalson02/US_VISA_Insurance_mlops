import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.usvisa_data import USvisaData

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):

        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise MyException(e,sys)
    
    def export_data_into_feature_store(self)->DataFrame:

        try:
            logging.info(f"Exporting data from mongodb")
            usvisa_data=USvisaData()
            dataframe=usvisa_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            features_store_file_path=self.data_ingestion_config.feature_store_file_path
            dir_path=os.path.dirname(features_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"Saving data into feature_Store: {features_store_file_path}")
            dataframe.to_csv(features_store_file_path,index=False,header=True)
            return dataframe
        except Exception as e:
            raise MyException(e,sys)

    def split_data(self,dataframe:DataFrame)->None:

        try:
            train_Set,test_Set=train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed test train split on data")
            dir_path=os.path.dirname(self.data_ingestion_config.testing_file_path)
            os.makedirs(dir_path,exist_ok=True)

            logging.info(f"Exporting train and test split")
            train_Set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_Set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)
            logging.info(f"Exported train and test data to their file paths")
        except Exception as e:
            raise MyException(e,sys) from e

    def initiate_data_ingestion(self)->DataIngestionArtifact:

        logging.info(f"Entered intiate_Data_ingestion")
        try:
            dataframe=self.export_data_into_feature_store()
            logging.info("Recieved data from mongodb")
            self.split_data(dataframe)
            logging.info("Data split done into test adn training data")
            data_ingestion_artifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path)
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e


