import os
import json
from argparse import ArgumentParser
from google.protobuf.json_format import MessageToDict, ParseDict
from pb.global_config_pb2 import GlobalConfig
from pb.tasks_pb2 import TaskConfig, TaskType, NextLocPredConfig
from pb.datasets_pb2 import DatasetsConfig
from pb.presentation_pb2 import PresentationConfig, GenHistoryPreConfig
from pb.runner_pb2 import RunnerConfig, DeepmoveRunnerConfig
from pb.models_pb2 import DeepmoveConfig
from pb.evaluate_pb2 import EvaluateConfig, EvalNextLocConfig

class Config:

    @classmethod
    def arguments_parser(cls):
        parser = ArgumentParser()
        parser.add_argument('-c', '--global_config', dest='global_config',
                            help='path to the global config file', required=False, default='global_config.json')
        # parser.add_argument('--config_dir_path', dest='config_dir_path',
        #                     help='config dirtory path', required=False)
        # parser.add_argument('-t', '--task_type', dest='task_type',
        #                     help='choose to run which task', required=False)
        # parser.add_argument('-d', '--datasets_path', dest='datasets_path',
        #                     help='datasets dir path', required=False)
        return parser

    def load_from_args(self):
        args = self.arguments_parser().parse_args()
        with open(args.global_config, 'r') as f:
            self.config_json = json.load(f)
        self.global_config = ParseDict(self.config_json, GlobalConfig())
        if not self.global_config.default_config:
            self.global_config.default_config = 'config'

    def load_config(self, message, relative_path):
        full_path = os.path.join(self.global_config.default_config, relative_path)
        with open(full_path, 'r') as f:
            config_json = json.load(f)
        return ParseDict(config_json, message)

    def load_from_default(self):
        self.task_config = TaskConfig()
        self.task_config.type = self.global_config.task.type
        if self.task_config.type == TaskType.NEXT_LOC_PRED:
            nlp_config = NextLocPredConfig()
            # nlp_config.datasets.MergeFrom(
            #     self.load_config(DatasetsConfig(), os.path.join('datasets', 'datasets.json'))
            # )

            nlp_config.presentation.gen_history_pre.MergeFrom(
                self.load_config(GenHistoryPreConfig(), os.path.join('presentation', 'gen_history.json'))
            )

            nlp_config.runner.deepmove.model_deepmove.MergeFrom(
                self.load_config(DeepmoveConfig(), os.path.join('model', 'deepMove.json'))
            )
            nlp_config.runner.deepmove.MergeFrom(
                self.load_config(DeepmoveRunnerConfig(), os.path.join('run', 'deepMove.json'))
            )

            nlp_config.evaluate.eval_next_loc.MergeFrom(
                self.load_config(EvalNextLocConfig(), os.path.join('evaluate', 'eval_next_loc.json'))
            )

            self.task_config.next_loc_pred.MergeFrom(nlp_config)

        self.task_config.MergeFrom(self.global_config.task)
        # print(self.task_config)
        # print(MessageToDict(self.task_config))


    def __init__(self):
        self.load_from_args()
        self.load_from_default()
