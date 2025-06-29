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

    data_kwargs = {
        "seq_len": args.lookback_window,
        "stride": args.stride,
        "feature_engineering": args.feature_engineering,
        "start_timestep": args.start_timestep
    }
    # TODO: check dimension
    d_feature = 895 + 2 if args.feature_engineering else 895

    model_kwargs = {
        "seq_len": data_kwargs["seq_len"],
        "d_feature": d_feature,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "n_blocks": args.n_blocks,
        "activation": args.activation,
        "dropout": args.dropout
    }
    train_kwargs = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer_name": args.optimizer,
        "use_early_stop": args.use_early_stop,
        "early_stop_patience": args.early_stop_patience,
        "grad_clip": args.grad_clip,
        "grad_norm": args.grad_norm,
        "continue_learning": args.continue_learning,
        "continue_expr": args.expr
    }

    if train_kwargs["continue_learning"]:
        train_kwargs["continue_expr"] = args.expr

    if args.model in ["vTransformer", "iTransformer"]:
        model_kwargs.update({
            "n_heads": args.n_heads,
            "attn_dropout": args.attn_dropout,
            "mask_flag": args.mask_flag
        })

    path_config = PathConfig(root_path)
    data_config = DataConfig(**data_kwargs)
    train_config = TrainConfig(**train_kwargs)
    model_config = ModelConfig(modelname=args.model, **model_kwargs)

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
