from configs.pathConfig import PathConfig
from configs.dataConfig import DataConfig
from configs.modelConfig import ModelConfig
from configs.trainConfig import TrainConfig
from trainers.GRU_trainer import GRUTrainer
from trainers.vTransformer_trainer import vTransformerTrainer
from utils.seed import set_seed
from utils.exceptions import ModelNotSupported

def main(args):
    set_seed(args.seed)
    root_path = args.root_path
    path_config = PathConfig(root_path)
    data_config = DataConfig()
    continue_learning = args.continue_learning
    train_config = TrainConfig()
    model_config = ModelConfig.create_model_config(args.model)
    if continue_learning:
        train_config.continue_learning = True
        train_config.continue_expr = args.expr
    if args.model == "GRU":
        trainer = GRUTrainer(model_config, train_config,
                             data_config, path_config)
    elif args.model == "vTransformer":
        trainer = vTransformerTrainer(model_config, train_config,
                                      data_config, path_config)
    else:
        raise ModelNotSupported(f"Model {args.model} not supported")
    trainer.train()
