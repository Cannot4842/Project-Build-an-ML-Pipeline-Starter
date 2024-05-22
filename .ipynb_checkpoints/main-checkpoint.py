import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        # Assuming 'active_steps' is a list of steps to be executed.
        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description": "Data with outliers and null values removed",
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']
                },
            )

            
           

        if "data_check" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                "main",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["data_check"]["min_price"],
                    "max_price": config["data_check"]["max_price"]
                },
            )

            
            

        if "data_split" in active_steps:
            try:
                _ = mlflow.run(
                    f"{config['main']['components_repository']}/train_val_test_split",
                    'main',
                    parameters={
                        "input": "clean_sample.csv:latest",
                        "test_size": config["modeling"]["test_size"],
                        "random_seed": config["modeling"]["random_seed"],
                        "stratify_by": config["modeling"]["stratify_by"]
                    }
                )
            except Exception as e:
                print(f"Error running data split: {e}")
                
        if "train_random_forest" in active_steps:
    # Define the path for the random forest configuration JSON
            rf_config_path = os.path.abspath("rf_config.json")
    
    # Serialize the random forest configuration into JSON
            try:
                with open(rf_config_path, "w+") as fp:
                    json.dump(dict(config["modeling"]["random_forest"].items()), fp)
            except Exception as e:
                print(e)
                raise

    # Execute the MLflow run command with parameters
            try:
                _ = mlflow.run(
                    f"{config['main']['components_repository']}/train_random_forest",
                    'main',
                    parameters={
                        "trainval_artifact": "trainval_data.csv:latest",
                        "val_size": str(config["modeling"]["val_size"]),
                        "random_seed": str(config["modeling"]["random_seed"]),
                        "stratify_by": config["modeling"]["stratify_by"],
                        "rf_config": rf_config_path,
                        "max_tfidf_features": str(config["modeling"]["max_tfidf_features"]),
                        "output_artifact": "random_forest_export"
                    }
                )
            except Exception as e:
                print(e)

                
#         if "train_random_forest" in active_steps:
                
#     # NOTE: we need to serialize the random forest configuration into JSON
#             rf_config = os.path.abspath("rf_config.json")
#             with open(rf_config, "w+") as fp:
#                 json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH
#     # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest

#             try:
#                 _ = mlflow.run(
#                     f"{config['main']['components_repository']}/train_random_forest",
#                     'main',
#                     parameters={
#                         "trainval_artifact": "trainval_data.csv:latest",
#                         "val_size": str(config["modeling"]["val_size"]),
#                         "random_seed": str(config["modeling"]["random_seed"]),
#                         "stratify_by": config["modeling"]["stratify_by"],
#                         "rf_config": rf_config,
#                         "max_tfidf_features": str(config["modeling"]["max_tfidf_features"]),
#                         "output_artifact": "random_forest_export"
#                     }
                
#                 )

#             except Exception as e:
#                 logger.error(f"Error running train_random_forest: {e}")

#         if "train_random_forest" in active_steps:
#             # NOTE: we need to serialize the random forest configuration into JSON
#                 rf_config = os.path.abspath("rf_config.json")
#                 with open(rf_config, "w+") as fp:
#                     json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH
#                 # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
#             try:
                
#                 # step

#                 _ = mlflow.run(
#                 f"{config['main']['components_repository']}/train_random_forest",
#                 'main',
#                 parameters={
#                     "trainval_artifact": "trainval_data.csv:latest",
#                     "val_size": str(config["modeling"]["val_size"]),
#                     "random_seed": str(config["modeling"]["random_seed"]),
#                     "stratify_by": config["modeling"]["stratify_by"],
#                     "rf_config": rf_config_path,
#                     "max_tfidf_features": str(config["modeling"]["max_tfidf_features"]),
#                     "output_artifact": "random_forest_export"
                    
#                     }
#                 )

  
#             except Exception as e:
#                 logger.error(f"Error running train_random_forest: {e}")

        if "test_regression_model" in active_steps:

            ##################
            # Implement here #
            ##################

            pass


if __name__ == "__main__":
    go()
