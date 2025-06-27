from configs.pathConfig import PathConfig
from configs.dataConfig import DataConfig
from configs.modelConfig import ModelConfig
from configs.trainConfig import TrainConfig
from trainers.GRU_trainer import GRUTrainer
from trainers.vTransformer_trainer import vTransformerTrainer
from trainers.iTransformer_trainer import iTransformerTrainer
from utils.seed import set_seed
from utils.exceptions import ModelNotSupported

def main(args):
    set_seed(args.seed)
    root_path = args.root_path
    lookback_window = args.lookback_window
    batch_size = args.batch_size
    continue_learning = args.continue_learning
    learning_rate = args.learning_rate
    optimizer = args.optimizer

    feature_engineering = args.feature_engineering
    # TODO: check dimension
    d_feature = 895 + 2 if feature_engineering else 895

    model_kwargs = {
        "seq_len": lookback_window,
        "d_feature": d_feature,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "n_blocks": args.n_blocks,
        "activation": args.activation,
        "dropout": args.dropout
    }

    if args.model in ["vTransformer", "iTransformer"]:
        model_kwargs.update({
            "n_heads": args.n_heads,
            "attn_dropout": args.attn_dropout,
            "mask_flag": args.mask_flag
        })

    path_config = PathConfig(root_path)
    data_config = DataConfig(seq_len=lookback_window)
    train_config = TrainConfig(
        batch_size=batch_size, learning_rate=learning_rate, optimizer_name=optimizer)

    model_config = ModelConfig(modelname=args.model, **model_kwargs)

    if continue_learning:
        train_config.continue_learning = True
        train_config.continue_expr = args.expr

    if args.model == "GRU":
        trainer = GRUTrainer(model_config, train_config,
                             data_config, path_config)
    elif args.model == "vTransformer":
        trainer = vTransformerTrainer(model_config, train_config,
                                      data_config, path_config)
    elif args.model == "iTransformer":
        trainer = iTransformerTrainer(model_config, train_config,
                                      data_config, path_config)
    else:
        raise ModelNotSupported(f"Model {args.model} not supported")

    trainer.train()
