import wandb
import subprocess
from pathlib import Path
from typing import Union, Callable, Optional
import yaml

# インスタンス化した後は引数が必要なのはlog関数とdump_json関数のみ(いずれもdata(dict)を引数に取る)

class TrainLogger:
    def __init__(
        self,
        use_wandb: bool = True,
        project_name: Optional[str] = None,
        run_name: Optional[str] = None,
        entity: Optional[str] = None,
        config: Optional[dict] = None,
        tags: Optional[list] = None,
        log_yaml_path: Optional[str] = None,
        sweep_config: Optional[Union[dict, str]] = None,
        sweep_id: Optional[str] = None,
        sweep_function: Optional[Callable] = None,
        sweep_counts: Optional[int] = None,
        **kwargs,
    ):
        """
        Initializes the Logger class, which can log to both WandB and standard logging.

        Parameters:
        - use_wandb (bool): Whether to use WandB for logging (default: True).
        - project_name (str): The name of the WandB project. Required if `use_wandb=True`.
        - run_name (str): The name of the run. Default is None.
        - entity (str): The entity or team under which the project is created. Default is None.
        - config (dict): A dictionary of configuration parameters to log. Default is None.
        - sweep_config (dict | str): Sweep configuration dict or path to YAML file.
        - tags (list): A list of tags for the run. Default is None.
        - log_yaml_path: Destination YAML file path.
        - sweep_id (str): Sweep ID. Default is None.
        - sweep_function (callable): Function to run for each agent trial. Default is None.
        - sweep_counts (int): Number of agent runs. Default is None.
        - kwargs: Additional kwargs for wandb.init().
        """
        if use_wandb and project_name is None:
            raise ValueError("project_name must be specified when use_wandb=True")
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.run_name = run_name
        self.entity = entity
        self.config = config
        self.tags = tags
        self.log_yaml_path = log_yaml_path
        self.sweep_config = sweep_config
        self.sweep_id = sweep_id
        self.sweep_function = sweep_function
        self.sweep_counts = sweep_counts
        self.kwargs = kwargs
        self.run = None

    def start(self):
        """
        Starts a new WandB run and initializes sweep ID if configured.
        """
        if self.use_wandb:
            self.run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                config=self.config,
                name=self.run_name,
                tags=self.tags,
                **self.kwargs
            )
        
        # Create sweep ID if needed
        if self.sweep_id is None and self.sweep_config is not None:
            if isinstance(self.sweep_config, dict):
                self.sweep_id = wandb.sweep(
                    sweep=self.sweep_config,
                    project=self.project_name,
                    entity=self.entity
                )
            elif isinstance(self.sweep_config, str) and Path(self.sweep_config).exists():
                out = subprocess.check_output(["wandb", "sweep", self.sweep_config])
                self.sweep_id = out.decode().strip().split()[-1]
            else:
                raise ValueError("sweep_config must be a dict or an existing file path.")

    def log(self, data: dict):
        """
        Logs a dict of metrics to WandB (if enabled).
        
        Parameters:
        - data (dict): Dictionary to serialize.
        """
        if not isinstance(data, dict):
            raise ValueError("The data should be a dictionary.")
        
        if self.use_wandb and self.run:
            wandb.log(data)

    def finish(self):
        """
        Finishes the current WandB run.
        """
        if self.use_wandb and self.run:
            wandb.finish()
            self.run = None

    def agent(self):
        """
        Starts a WandB agent to run the sweep.
        """
        if not self.use_wandb:
            raise RuntimeError("W&B is disabled.")
        if self.sweep_id is None:
            raise RuntimeError("sweep_id not found. Call start() or set sweep_id manually.")
        wandb.agent(
            sweep_id=self.sweep_id,
            function=self.sweep_function,
            count=self.sweep_counts
        )

    def controller(self):
        """
        Returns a WandB controller for advanced sweep control.
        """
        if not self.use_wandb:
            raise RuntimeError("W&B is disabled.")
        if self.sweep_id is None:
            raise RuntimeError("sweep_id not found. Call start() or set sweep_id manually.")
        return wandb.controller(
            sweep_id_or_config=self.sweep_id,
            entity=self.entity,
            project=self.project_name
        )

    def dump_yaml(self, data: dict):
        """
        Dumps the provided dict to the specified YAML file path.

        Parameters:
        - data (dict): Dictionary to serialize.
        - filepath (str | Path): Destination YAML file path.
        """
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary.")
        path = Path(self.log_yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


# # 使用例
# if __name__ == "__main__":
#     config = {
#         'learning_rate': 0.001,
#         'epochs': 10,
#         'batch_size': 32
#     }
#     # WandbLoggerのインスタンスを作成
#     logger = WandbLogger(project_name="my_project", run_name="experiment_1", entity="my_team", config=config, tags=["baseline", "test"])
    
#     # ログを開始
#     logger.start()

#     # 学習ループ（外部に記述）
#     for epoch in range(config['epochs']):
#         # 学習コードをここに記述
#         # 例: トレーニングロスとバリデーションロスの計算
#         train_loss = 0.05 * epoch  # トレーニングロスを仮置き
#         valid_loss = 0.04 * epoch  # バリデーションロスを仮置き
#         accuracy = 0.8 + 0.02 * epoch  # 精度を仮置き

#         # 結果をWandbにログ
#         logger.log({
#             'epoch': epoch,
#             'train_loss': train_loss,
#             'valid_loss': valid_loss,
#             'accuracy': accuracy
#         })

#     # ログを終了
#     logger.finish()