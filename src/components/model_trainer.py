import sys

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from neuro_mf import ModelFactory   
from typing import Tuple

from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact,ClassificationMetricArtifact
from src.utils.main_utils import read_yaml_file,load_object,save_object,load_numpy_array_data
from src.entity.estimator import USvisaModel
import dagshub
import mlflow
import mlflow.sklearn



# Below code block is for local use
#-------------------------------------------------------------------------------------



class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        try:
            logging.info("Using neuro_mf to get best model object and report")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

            best_model_detail = model_factory.get_best_model(
                X=x_train,y=y_train,base_accuracy=self.model_trainer_config.expected_accuracy
            )
            model_obj = best_model_detail.best_model

            y_pred = model_obj.predict(x_test)
            
            accuracy = accuracy_score(y_test, y_pred) 
            f1 = f1_score(y_test, y_pred)  
            precision = precision_score(y_test, y_pred)  
            recall = recall_score(y_test, y_pred)
            metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)
            
            return best_model_detail, metric_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e
        

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        
        try:
            train_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )

            import mlflow
            import mlflow.sklearn

            with mlflow.start_run(run_name="model_evaluation"):

                run_id = mlflow.active_run().info.run_id

                # 🔥 TRAIN + GET METRICS
                best_model_detail, metric_artifact = self.get_model_object_and_report(
                    train=train_arr, test=test_arr
                )

                # ✅ LOG METRICS
                mlflow.log_metric("f1_score", metric_artifact.f1_score)
                mlflow.log_metric("precision", metric_artifact.precision_score)
                mlflow.log_metric("recall", metric_artifact.recall_score)

                # ✅ LOG PARAM
                mlflow.log_param(
                    "model_name",
                    type(best_model_detail.best_model).__name__
                )

                preprocessing_obj = load_object(
                    file_path=self.data_transformation_artifact.transformed_object_file_path
                )

                usvisa_model = USvisaModel(
                    preprocessing_object=preprocessing_obj,
                    trained_model_object=best_model_detail.best_model
                )

                # ✅ LOG FULL PIPELINE (BEST PRACTICE)
                mlflow.sklearn.log_model(
                    sk_model=best_model_detail.best_model,
                    artifact_path="model"
                )
                
            # ⬇️ AFTER MLFLOW RUN ENDS

            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                raise Exception("No best model found with score more than base score")

            save_object(
                self.model_trainer_config.trained_model_file_path,
                usvisa_model
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
                run_id=run_id
            )

            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e