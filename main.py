import argparse
import traceback
from typing import Any, Dict, List

import yaml
from base import BaseStep
from error import UnrecoverableError
from step import Aligner, BuildAlignerManifest


def init_pipeline(pipeline: List[BaseStep], pipeline_config: Dict[str, Any]):
    for k, v in pipeline_config["pipeline"]["global"].items():
        print(k, v)
        pipeline[0].set_state(k, v)

    for item in pipeline:
        params = pipeline_config["pipeline"]["steps"][item.get_id()].get("params", {})
        infra = pipeline_config["pipeline"]["steps"][item.get_id()]["infra"]
        item.initialise(infra, **params)


def run_pipeline(pipeline: List[BaseStep]):
    for item in pipeline:
        print(f"Running {item.get_id()}")
        item.run()
        print(f"Finished running {item.get_id()}")


def main(args: argparse.Namespace):
    with open(args.config) as stream:
        pipeline_config = yaml.safe_load(stream)

    pipeline: List[BaseStep] = [BuildAlignerManifest(), Aligner()]

    print("Initialising pipeline")
    init_pipeline(pipeline, pipeline_config)
    run_pipeline(pipeline)


parser = argparse.ArgumentParser(
    prog="Spoken Translation Pipeline",
    description="",
    epilog="",
)

parser.add_argument("-c", "--config", type=str, required=True)
args = parser.parse_args()
main(args)
