from configs.pathConfig import PathConfig
from configs.dataConfig import DataConfig
from configs.modelConfig import ModelConfig
from configs.trainConfig import TrainConfig
from trainers.GRU_trainer import GRUTrainer
from trainers.vTransformer_trainer import vTransformerTrainer
from utils.seed import set_seed


def main(args):
    root_path = args.root_path

    path_config = PathConfig(root_path=root_path)
    data_config = DataConfig()
    train_config = TrainConfig()
    model_config = ModelConfig.create_model_config(args.model)

    if args.model == "GRU":
        trainer = GRUTrainer(model_config, train_config,
                             data_config, path_config)
    elif args.model == "vTransformer":
        trainer = vTransformerTrainer(model_config, train_config,
                                      data_config, path_config)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")

    trainer.test(expr=args.expr)
