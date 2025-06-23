from configs.pathConfig import PathConfig
from configs.dataConfig import DataConfig
from configs.modelConfig import ModelConfig
from configs.trainConfig import TrainConfig
from trainers.GRU_trainer import GRUTrainer
from utils.seed import set_seed


def main(args):
    root_path = args.root_path

    path_config = PathConfig(root_path=root_path)
    data_config = DataConfig()
    model_config = ModelConfig()
    train_config = TrainConfig()

    if args.model == "GRU":
        trainer = GRUTrainer(model_config, train_config,
                             data_config, path_config)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")

    test_loss = trainer.test(expr=args.expr)
    print(f"Test loss: {test_loss}")

    return
