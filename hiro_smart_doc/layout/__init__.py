from typing import Any
from .model_runner_25.model_runner_25 import ModelRunner25
from .model_runner_9.model_runner_9 import ModelRunner9
from .model_runner_5.model_runner_5 import ModelRunner5
from .model_runner_chem.model_runner_chem import ModelRunnerChem
from pathlib import Path
from .backends.base import Backend


class LayoutRunner:

    def __init__(self, model_configs: list[dict[str, Any]]):
        self.model_runners = {}
        for model_config in model_configs:
            self.model_runners[model_config["model_id"]] = LayoutRunner.create_model(model_config["model_id"], model_config["backend"], model_config["model_path"], model_config["threads"])
    @staticmethod
    def create_model(model_id: str, backend: Backend, model_path: Path, threads: int):
        """
        根据model_id创建对应的模型实例
        
        Args:
            model_id: 模型ID ("25" 或 "5")
            backend: 后端类型
            model_path: 模型路径
            threads: 线程数
            
        Returns:
            对应的模型实例
        """
        match model_id:
            case "25":
                return ModelRunner25(backend, model_path, threads)
            case "9":
                return ModelRunner9(backend, model_path, threads)
            case "5":
                return ModelRunner5(backend, model_path, threads)
            case "chem":
                return ModelRunnerChem(backend, model_path, threads)
            case _:
                raise ValueError(f"Unsupported model_id: {model_id}. Supported values: '25', '9', '5', 'chem'")

    def __getitem__(self, model_id: str):
        """
        获取指定model_id的模型运行器
        
        Args:
            model_id: 模型ID ("25" 或 "5")
            
        Returns:
            模型运行器实例
        """
        return self.model_runners[model_id]