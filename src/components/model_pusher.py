import sys
import mlflow
from mlflow.tracking import MlflowClient
from src.exception import MyException
from src.logger import logging
from src.cloud_storage.aws_storage import SimpleStorageService
from src.entity.artifact_entity import ModelEvaluationArtifact , ModelPusherArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import USvisaEstimator
import mlflow.sklearn
import dagshub



class ModelPusher:

    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.usvisa_estimator = USvisaEstimator(bucket_name=model_pusher_config.bucket_name,
                                model_path=model_pusher_config.s3_model_key_path)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")

        try:
            logging.info("Uploading artifacts folder to s3 bucket")

            self.usvisa_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)

            #Register model in MLflow
            run_id = self.model_evaluation_artifact.run_id  


            model_uri = f"runs:/{run_id}/model"

            logging.info(f"Registering model from URI: {model_uri}")

            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name="USVisaModel"
            )

            #Promote to Production (if accepted)
            client = MlflowClient()

            if self.model_evaluation_artifact.is_model_accepted:
                logging.info("Promoting model to Production")

                client.transition_model_version_stage(
                    name="USVisaModel",
                    version=registered_model.version,
                    stage="Production"
                )

            model_pusher_artifact = ModelPusherArtifact(bucket_name=self.model_pusher_config.bucket_name,
                                                        s3_model_path=self.model_pusher_config.s3_model_key_path)

            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelTrainer class")
            
            return model_pusher_artifact
        except Exception as e:
            raise MyException(e, sys) from e