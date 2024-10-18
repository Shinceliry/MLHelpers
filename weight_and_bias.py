import wandb

class WandbLogger:
    def __init__(self, project_name, entity=None, config=None, **kwargs):
        """
        Initializes the WandbLogger class with flexibility for various tasks.
        
        Parameters:
        - project_name (str): The name of the WandB project.
        - entity (str): The entity or team under which the project is created. Default is None.
        - config (dict): A dictionary of configuration parameters to log. Default is None.
        - kwargs: Additional keyword arguments for wandb.init() to allow further customization.
        """
        self.project_name = project_name
        self.entity = entity
        self.config = config
        self.kwargs = kwargs
        self.run = None

    def start(self, run_name=None, tags=None):
        """
        Starts a new WandB run with the specified name and tags.
        
        Parameters:
        - run_name (str): The name of the run. Default is None.
        - tags (list): A list of tags for the run. Default is None.
        """
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            config=self.config,
            name=run_name,
            tags=tags,
            **self.kwargs
        )

    def log(self, data):
        """
        Logs data to the current WandB run.
        
        Parameters:
        - data (dict): A dictionary of data to log.
        """
        if self.run:
            if isinstance(data, dict):
                wandb.log(data)
            else:
                raise ValueError("The data should be a dictionary.")
        else:
            raise RuntimeError("You need to start a run before logging data.")

    def finish(self):
        """
        Finishes the current WandB run.
        """
        if self.run:
            wandb.finish()
            self.run = None

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
