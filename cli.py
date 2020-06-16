from argparse import ArgumentParser
import json
import sys
import tempfile
from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple

import _jsonnet
from cliff.app import App
from cliff.command import Command
from cliff.commandmanager import CommandManager

import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import CategoricalDistribution


_logger = optuna.logging.get_logger(__name__)


class Search(Command):
    """Optimize hyperparameter."""

    def get_parser(self, prog_name: str) -> ArgumentParser:

        parser = super(Search, self).get_parser(prog_name)
        parser.add_argument(
            "--experiment-name", help="Name of the experiment. Alias for study name of Optuna.",
        )
        parser.add_argument("--study-name", help="A study name.")
        parser.add_argument("--storage", help="DB URL.")
        parser.add_argument(
            "--search-space", required=True, help="Path to search space configuration."
        )
        parser.add_argument("--num-samples", type=int, help="Alias for the number of trials of Optuna.")
        parser.add_argument("--n-trials", type=int, help="The number of trials.")
        parser.add_argument("--timeout", type=int, help="Seconds to timeout.")
        parser.add_argument(
            "--base-config", required=True, help="Path to configuration file of AllenNLP."
        )
        return parser

    def _get_metric_and_direction(
        self, base_config: str, params: Dict[str, Any]
    ) -> Tuple[str, str]:
        json_str = _jsonnet.evaluate_file(base_config, ext_vars=params)
        config = json.loads(json_str)
        # The default validation metric is "-loss" according to
        # https://github.com/allenai/allennlp/blob/master/allennlp/training/trainer.py#L221-L225.
        validation_metric = config["trainer"].get("validation_metric", "-loss")
        optuna_metric = "best_validation_{}".format(validation_metric[1:])
        direction = "maximize" if validation_metric.startswith("+") else "minimize"
        return optuna_metric, direction

    def _parse_optuna_search_space(self, search_space_path):
        with open(search_space_path) as fin:
            search_space = json.load(fin)

        optuna_search_space: Dict[str, BaseDistribution] = {}
        low_values: Dict[str, Any] = {}
        for name, value in search_space.items():
            if not isinstance(value, dict):
                d = CategoricalDistribution((value,))
                low_values[name] = str(value)
                optuna_search_space[name] = d
                continue

            sampling_strategy = value["sampling strategy"]
            if sampling_strategy == "choice":
                d = CategoricalDistribution(tuple(value["choices"]))
                optuna_search_space[name] = d
                low_values[name] = str(value["choices"][0])
                continue

            if sampling_strategy == "integer":
                d = IntUniformDistribution(value["bounds"][0], value["bounds"][1])
                optuna_search_space[name] = d
            elif sampling_strategy == "uniform":
                d = UniformDistribution(value["bounds"][0], value["bounds"][1])
                optuna_search_space[name] = d
            elif sampling_strategy == "loguniform":
                d = LogUniformDistribution(value["bounds"][0], value["bounds"][1])
                optuna_search_space[name] = d
            else:
                raise ValueError(f"Unknown sampling strategy: {sampling_strategy}.")
            low_values[name] = str(value["bounds"][0])
        return optuna_search_space, low_values

    def _create_objective_and_direction(
        self, base_config_path: str, search_space_path: str
    ) -> Tuple[Callable[[optuna.Trial], float], str]:

        optuna_search_space, low_values = self._parse_optuna_search_space(search_space_path)
        optuna_metric, direction = self._get_metric_and_direction(base_config_path, low_values)

        def objective(trial: optuna.Trial) -> float:
            for name, distribution in optuna_search_space.items():
                trial._suggest(name, distribution)

            print(trial.params)
            with tempfile.TemporaryDirectory() as out_dir:
                executor = optuna.integration.AllenNLPExecutor(
                    trial, base_config_path, out_dir, optuna_metric,
                )
                obj_value = executor.run()
            return obj_value

        return objective, direction

    def take_action(self, parsed_args: Dict[str, Any]):

        objective, direction = self._create_objective_and_direction(
            parsed_args.base_config, parsed_args.search_space
        )
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=parsed_args.n_trials, timeout=parsed_args.timeout)

        best_trial = study.best_trial
        print(f"Best score: {best_trial.value}")
        print(json.dumps(best_trial.params, indent=2))

class AllenNLPOptunaApp(App):
    def __init__(self) -> None:
        super(AllenNLPOptunaApp, self).__init__(
            description="", version="0.0.1", command_manager=CommandManager("allenopt.command"),
        )


def main() -> int:
    argv = sys.argv[1:] if len(sys.argv) > 1 else ["help"]
    return AllenNLPOptunaApp().run(argv)


if __name__ == "__main__":
    sys.exit(main())
