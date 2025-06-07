import wandb
import logging
import json
import torch

class TrainLogger:
    def __init__(self, project_name=None, entity=None, config=None, log_level=logging.INFO, log_file=None, use_wandb=True, **kwargs):
        """
        Initializes the Logger class, which can log to both WandB and standard logging.

        Parameters:
        - project_name (str): The name of the WandB project. Required if `use_wandb=True`.
        - entity (str): The entity or team under which the project is created. Default is None.
        - config (dict): A dictionary of configuration parameters to log. Default is None.
        - log_level (int): Logging level (default: logging.INFO).
        - log_file (str): Optional file path to save logs. If None, logs will be printed to console.
        - use_wandb (bool): Whether to use WandB for logging (default: True).
        - kwargs: Additional keyword arguments for wandb.init() to allow further customization.
        """
        self.use_wandb = use_wandb
        self.run = None

        # WandB setup (only if enabled)
        if self.use_wandb and project_name is None:
            raise ValueError("project_name must be specified when use_wandb=True")
        
        self.project_name = project_name
        self.entity = entity
        self.config = config
        self.kwargs = kwargs

        # Setup Python logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        if log_file:
            handler = logging.FileHandler(log_file)
        else:
            handler = logging.StreamHandler()

        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        if config:
            self.logger.info("Logger initialized with configuration:")
            self.logger.info(json.dumps(config, indent=4))

    def start(self, run_name=None, tags=None):
        """
        Starts a new WandB run (if enabled) and logs the start of the run.
        
        Parameters:
        - run_name (str): The name of the run. Default is None.
        - tags (list): A list of tags for the run. Default is None.
        """
        if self.use_wandb:
            self.run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                config=self.config,
                name=run_name,
                tags=tags,
                **self.kwargs
            )
            self.logger.info(f"Started WandB run: {run_name}, Tags: {tags}")
        else:
            self.logger.info("WandB logging is disabled. Run started locally.")

    def log(self, data):
        """
        Logs data to the current WandB run (if enabled) and also logs it using Python's logging module.
        
        Parameters:
        - data (dict): A dictionary of data to log.
        """
        if isinstance(data, dict):
            if self.use_wandb and self.run:
                wandb.log(data)
            serializable_data = {
                k: (v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in data.items()
            }
            self.logger.info(f"Logged data: {json.dumps(serializable_data, indent=4)}")
        else:
            raise ValueError("The data should be a dictionary.")

    def finish(self):
        """
        Finishes the current WandB run (if enabled) and logs the completion.
        """
        if self.use_wandb and self.run:
            wandb.finish()
            self.logger.info("Finished WandB run.")
            self.run = None
        else:
            self.logger.info("Run finished (WandB was not used).")

# # 使用例
# if __name__ == "__main__":
#     config = {
#         'learning_rate': 0.001,
#         'epochs': 10,
#         'batch_size': 32
#     }
#     # WandbLoggerのインスタンスを作成
#     logger = WandbLogger(project_name="my_project", entity="my_team", config=config)
    
#     # ログを開始
#     logger.start(run_name="experiment_1", tags=["baseline", "test"])

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