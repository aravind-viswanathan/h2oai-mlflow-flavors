import os
import warnings
import yaml

import mlflow
import shutil
import uuid
import mlflow.utils.environment
import mlflow.utils.model_utils

from mlflow.exceptions import MlflowException
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.utils.environment import (
    _validate_env_arguments,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

from mlflow.utils.model_utils import (
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.file_utils import (
    write_to,
)

from h2o_mlflow_flavors.utils import unzip_specific_file
from h2o_mlflow_flavors.utils import match_file_from_name_pattern
from h2o_mlflow_flavors.utils import unzip_specific_folder
from h2o_mlflow_flavors.utils import zip_folder

import h2o_mlflow_flavors

FLAVOR_NAME = "h2o_driverless_ai"
MOJO_FILE = "mojo-pipeline/pipeline.mojo"
PY_SCORING_WHL_FILE_PATTERN = "scoring-pipeline/scoring_h2oai_experiment.*\.whl"
PY_SCORING_SUMMARY_FILE_PATTERN = "scoring-pipeline/h2oai_experiment_summary.*\.zip"
PY_SCORING_FILE_NAME = "scorer"
PY_SCORING_CUSTOM_RECIPES_FOLDER = "tmp"

def save_model(
        h2o_dai_artifact_location,
        h2o_dai_model_file,
        path,
        model_type,
        conda_env=None,
        mlflow_model=None,
        settings=None,
        signature: ModelSignature = None,
        input_example: ModelInputExample = None,
        pip_requirements=None,
        extra_pip_requirements=None,
):
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    model_data_subpath = "model.h2o.dai"
    model_data_path = os.path.join(path, model_data_subpath)
    os.makedirs(model_data_path)


    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    if settings is None:
        settings = {}
    settings["dai_source_file"] = h2o_dai_artifact_location
    settings["dai_model"] = _get_file_name(h2o_dai_model_file)
    with open(os.path.join(model_data_path, "h2o_dai.yaml"), "w") as settings_file:
        yaml.safe_dump(settings, stream=settings_file)

    shutil.copy(h2o_dai_model_file, model_data_path+"/"+_get_file_name(h2o_dai_model_file))

    mlflow_model.add_flavor(
        FLAVOR_NAME,  type=model_type
    )

    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

def _get_file_name(path):
    return os.path.basename(path)

def _get_dir_path(path):
    return os.path.dirname(path)

def log_model(h2o_dai_artifact_location,
            artifact_path,
            model_type="dai/mojo_pipeline",
            h2o_dai_model_download_location="/tmp/"+str(uuid.uuid1()),
            conda_env=None,
            registered_model_name=None,
            signature: ModelSignature = None,
            input_example: ModelInputExample = None,
            pip_requirements=None,
            extra_pip_requirements=None,
            **kwargs,
    ):

        valid_dai_model_types = ['dai/mojo_pipeline', 'dai/scoring_pipeline']
        if model_type not in valid_dai_model_types:
            raise MlflowException.invalid_parameter_value("Invalid value for model_type. Valid values are pipeline/mojo or scoring-pipeline")


        h2o_dai_model_file = determine_model_file(model_type, h2o_dai_artifact_location, h2o_dai_model_download_location)

        return Model.log(
            artifact_path=artifact_path,
            flavor=h2o_mlflow_flavors.driverless,
            registered_model_name=registered_model_name,
            h2o_dai_artifact_location=h2o_dai_artifact_location,
            conda_env=conda_env,
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            model_type = model_type,
            h2o_dai_model_file=h2o_dai_model_file,
            **kwargs,
        )

def determine_model_file(model_type, h2o_dai_model, h2o_dai_model_download_location):
    if model_type == 'dai/mojo_pipeline':
        return _minimise_mojo_scoring_model(h2o_dai_model)
    elif model_type == 'dai/scoring_pipeline':
        return _minimise_python_scoring_pipeline_model(h2o_dai_model, h2o_dai_model_download_location)

def _minimise_mojo_scoring_model(h2o_dai_model):
    if match_file_from_name_pattern(h2o_dai_model, MOJO_FILE) is None:
        raise MlflowException.invalid_parameter_value("Not a valid DAI MOJO Pipeline - pipeline.mojo not present in the provided model.")
    directory = "/tmp"
    unzip_specific_file(h2o_dai_model, MOJO_FILE, directory=directory)
    return directory + "/" + MOJO_FILE

def _minimise_python_scoring_pipeline_model(h2o_dai_model, h2o_dai_model_download_location):
    location = h2o_dai_model_download_location + "/"
    minimal_model_file_location = location + "model"
    wheel_file = match_file_from_name_pattern(h2o_dai_model, PY_SCORING_WHL_FILE_PATTERN)
    summary_file = match_file_from_name_pattern(h2o_dai_model, PY_SCORING_SUMMARY_FILE_PATTERN)

    if wheel_file is None or summary_file is None:
        raise MlflowException.invalid_parameter_value("Not a valid DAI Scoring Pipeline - Experiment wheel or Summary file not present in the provided model")

    unzip_specific_file(h2o_dai_model, wheel_file, summary_file, directory=minimal_model_file_location)
    unzip_specific_folder(h2o_dai_model, PY_SCORING_CUSTOM_RECIPES_FOLDER, directory=minimal_model_file_location)
    return zip_folder(minimal_model_file_location, location + PY_SCORING_FILE_NAME)
