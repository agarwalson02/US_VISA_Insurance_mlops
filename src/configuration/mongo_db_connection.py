import sys

from src.exception import MyException
from src.logger import logging

import os
from src.constants import DATABASE_NAME, MONGODB_URL_KEY
import pymongo
import certifi
from dotenv import load_dotenv
load_dotenv()

ca = certifi.where() #This is to avoid timeout error

class MongoDBClient:
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)

                if not mongo_db_url:
                    raise ValueError(f"{MONGODB_URL_KEY} not found in environment")

                MongoDBClient.client = pymongo.MongoClient(
                    mongo_db_url,
                    tlsCAFile=certifi.where()
                )

                logging.info("MongoDB client created")

            self.client = MongoDBClient.client
            self.database = self.client[database_name]

            logging.info(f"Connected to database: {database_name}")

        except Exception as e:
            raise MyException(e, sys)