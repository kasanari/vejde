import contextlib
import logging
from dataclasses import asdict
from pathlib import Path

import mlflow

from .config import Args
from .train_agent import train

logger = logging.getLogger(__name__)


def train_with_mlflow(
    args: Args, mlflow_tracking_uri: str, run_name: str, batch_id: str | None = None
):
    mlflow.enable_system_metrics_logging()
    mlflow.set_tracking_uri(uri=mlflow_tracking_uri)

    with contextlib.suppress(mlflow.MlflowException):
        mlflow.create_experiment(run_name)

    mlflow.set_experiment(run_name)

    with mlflow.start_run():
        logger.info(f"Connected to mlflow at {mlflow_tracking_uri}")
        mlflow.log_param("using_edge_attr", True)
        mlflow.log_param("using_scaling", True)

        mlflow.log_artifact(__file__)
        if Path("uv.lock").exists():
            mlflow.log_artifact("uv.lock")
        if Path("pyproject.toml").exists():
            mlflow.log_artifact("pyproject.toml")
        if batch_id:
            mlflow.log_param("batch_id", batch_id)

        try:
            stats, agent = train(args)
        except Exception as e:
            logger.exception("Exception during training:")
            raise e

        logged_config = vars(args) | asdict(agent.config)
        mlflow.log_params(logged_config)

        agent.save_agent(stats["run_folder"] / f"{run_name}.zip")
        mlflow.log_artifact(str(stats["run_folder"] / f"{run_name}.zip"))

        for k, v in stats.items():
            if k != "returns":
                mlflow.log_metric(f"train_eval/{k}", v)

        return stats, agent
