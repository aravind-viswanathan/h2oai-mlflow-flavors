import os
import warnings
import yaml

import mlflow
import shutil
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
    _mlflow_conda_env,
    _validate_env_arguments,
    _process_pip_requirements,
    _process_conda_env,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _PythonEnv,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

from mlflow.utils.model_utils import (
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _add_code_from_conf_to_system_path,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.file_utils import (
    write_to,
)

import h2o_mlflow_flavors

FLAVOR_NAME = "h2o_driverless_ai"


def save_model(
        h2o_dai_model,
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
    if h2o_dai_model is not None:
        print(h2o_dai_model)

    if settings is None:
        settings = {}
    settings["full_file"] = h2o_dai_model
    settings["model_file"] = _get_file_name(h2o_dai_model)
    settings["model_dir"] = _get_dir_path(h2o_dai_model)
    with open(os.path.join(model_data_path, "h2o_dai.yaml"), "w") as settings_file:
        yaml.safe_dump(settings, stream=settings_file)

    shutil.copy(h2o_dai_model, model_data_path+"/"+_get_file_name(h2o_dai_model))

    mlflow_model.add_flavor(
        FLAVOR_NAME,  type=model_type
    )

    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

def _get_file_name(path):
    return os.path.basename(path)

def _get_dir_path(path):
    return os.path.dirname(path)

def log_model(h2o_dai_model,
            artifact_path,
            model_type="dai/mojo_pipeline",
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

        return Model.log(
            artifact_path=artifact_path,
            flavor=h2o_mlflow_flavors.driverless,
            registered_model_name=registered_model_name,
            h2o_dai_model=h2o_dai_model,
            conda_env=conda_env,
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            model_type = model_type,
            **kwargs,
        )