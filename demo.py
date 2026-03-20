from src.pipline.training_pipeline import TrainPipeline

import mlflow
import mlflow.sklearn
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/agarwalson02/US_VISA_Insurance_mlops.mlflow/')
dagshub.init(repo_owner='agarwalson02', repo_name='MLOPS-imdb-pipeline', mlflow=True)
mlflow.set_experiment("US_VISA_PIPELINE")

pipeline=TrainPipeline()
pipeline.run_pipeline()