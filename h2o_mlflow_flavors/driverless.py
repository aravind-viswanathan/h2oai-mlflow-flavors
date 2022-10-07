import os
import warnings
import yaml

import mlflow
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
        conda_env=None,
        code_paths=None,
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
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    if settings is None:
        settings = {}
    settings["full_file"] = "h2o_save_location"
    settings["model_file"] = "pipeline.mojo"
    settings["model_dir"] = "pipeline"
    with open(os.path.join(model_data_path, "h2o_dai.yaml"), "w") as settings_file:
        yaml.safe_dump(settings, stream=settings_file)

    mlflow_model.add_flavor(
        FLAVOR_NAME, data=model_data_subpath,  type="pipeline/mojo"
    )

    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def log_model(h2o_dai_model,
            artifact_path,
            conda_env=None,
            code_paths=None,
            registered_model_name=None,
            signature: ModelSignature = None,
            input_example: ModelInputExample = None,
            pip_requirements=None,
            extra_pip_requirements=None,
            **kwargs,
    ):

        return Model.log(
            artifact_path=artifact_path,
            flavor=h2o_mlflow_flavors.driverless,
            registered_model_name=registered_model_name,
            h2o_dai_model=h2o_dai_model,
            conda_env=conda_env,
            code_paths=code_paths,
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            **kwargs,
        )