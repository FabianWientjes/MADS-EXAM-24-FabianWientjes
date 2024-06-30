from pathlib import Path
from typing import Dict

import ray
import torch
from filelock import FileLock
from loguru import logger
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics
import numpy as np
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB  # Corrected import statement
from src import models, datasets
import sys
sys.path.append('../')

SAMPLE_INT = tune.sample_from
SAMPLE_FLOAT = tune.sample_from

class Recall:
    def __repr__(self) -> str:
        return "Recall"

    def __call__(self, y, yhat):
        true_positives = np.sum((np.argmax(yhat, axis=1) == y) & (y == 1))
        false_negatives = np.sum((np.argmax(yhat, axis=1) != y) & (y == 1))
        return true_positives / (true_positives + false_negatives + 1e-10)  # Prevent division by zero

def train(config: Dict):
    """
    The train function should receive a config file, which is a Dict
    ray will modify the values inside the config before it is passed to the train
    function.
    """

    from src.datasets import HeartDataset1D  # Assuming 'HeartDataset1D' is defined in src.datasets
    from mads_datasets.base import BaseDatastreamer
    from mltrainer.preprocessors import BasePreprocessor

    trainfile = Path("../data/heart_big_train.parq").resolve()
    testfile = Path('../data/heart_big_test.parq').resolve()

    traindataset = HeartDataset1D(trainfile, target="target")
    testdataset = HeartDataset1D(testfile, target="target")

    thisfile = Path(__file__)
    logger.info(f"thisfile {thisfile}")
        
    trainfile = (thisfile / "../data/heart_big_train.parq").resolve()
    testfile = (thisfile / "../data/heart_big_test.parq").resolve()
    traindataset = datasets.HeartDataset1D(trainfile, "target")
    testdataset = datasets.HeartDataset1D(testfile, target="target")

    trainstreamer = BaseDatastreamer(traindataset, preprocessor=BasePreprocessor(), batchsize=32)
    teststreamer = BaseDatastreamer(testdataset, preprocessor=BasePreprocessor(), batchsize=32)

    # Locking the data directory to avoid parallel instances trying to
    # access it simultaneously
    with FileLock(config["data_dir"] / ".lock"):
        streamers = {
            "train": trainstreamer,
            "valid": teststreamer  # Assuming test dataset is used for validation
        }

    # Setting up the metric
    recall = Recall()
    model = models.GRUModel(config)

    trainersettings = TrainerSettings(
        epochs=50,
        metrics=[recall],
        logdir=Path("."),
        train_steps=len(trainstreamer),
        valid_steps=len(teststreamer),
        reporttypes=[ReportTypes.RAY],
        scheduler_kwargs={"factor": 0.5, "patience": 5},
        earlystop_kwargs=None,
    )

    # Setting up Trainer instance
    trainer = Trainer(
        model=model,
        settings=trainersettings,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        traindataloader=trainstreamer.stream(),
        validdataloader=teststreamer.stream(),
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

    trainer.loop()


if __name__ == "__main__":
    ray.init()

    data_dir = Path("data/heart_bi").resolve()
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        logger.info(f"Created {data_dir}")
    
    tune_dir = Path("models/ray").resolve()

    config = {
        "hidden": tune.randint(64, 513),
        "dropout": tune.uniform(0.0, 0.3),
        "output": 2,
        "num_heads": 4,
        "num_blocks": tune.randint(1, 6),
        "data_dir": data_dir,
        "tune_dir": tune_dir
    }

    reporter = CLIReporter()
    reporter.add_metric_column("Recall")

    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=50,
        reduction_factor=3,
        stop_last_trials=False,
    )

    bohb_search = TuneBOHB()

    analysis = tune.run(
        train,
        config=config,
        metric="test_loss",
        mode="min",
        progress_reporter=reporter,
        num_samples=50,
        search_alg=bohb_search,
        scheduler=bohb_hyperband,
        verbose=1,
    )

    ray.shutdown()