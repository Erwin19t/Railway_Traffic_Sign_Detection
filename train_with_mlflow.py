import dagshub
import mlflow
import mlflow.pytorch
import argparse
from yolov5 import train
from pathlib import Path

yolo_path = '/content/DagsHub_mlFlow_Playground/yolov5'
FILE = Path(yolo_path).resolve()
ROOT = FILE  # YOLOv5 root directory

def parse_opt(known=False):
    """Parses command-line arguments for YOLOv5 training, validation, and testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    parser.add_argument("--noautoanchor", action="store_true", help="disable AutoAnchor")
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")
    parser.add_argument("--evolve_population", type=str, default=ROOT / "data/hyps", help="location for loading population")
    parser.add_argument("--resume_evolve", type=str, default=None, help="resume evolve from last generation")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT / "runs/train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")

    # Logger arguments
    parser.add_argument("--entity", default=None, help="Entity")
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")

    # NDJSON logging
    parser.add_argument("--ndjson-console", action="store_true", help="Log ndjson to console")
    parser.add_argument("--ndjson-file", action="store_true", help="Log ndjson to file")

    return parser.parse_known_args()[0] if known else parser.parse_args()

def main():
    args = parse_opt()

    dagshub.init("DagsHub_mlFlow_Playground", "erwin19t", mlflow=True)
    # Start an MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(vars(args))

        # Train the model
        results = train.train(
            weights             =args.weights,
            cfg                 =args.cfg,
            data                =args.data,
            hyp                 =args.hyp,
            epochs              =args.epochs,
            batch_size          =args.batch_size,
            imgsz               =args.imgsz,
            rect                =args.rect,
            resume              =args.resume,
            nosave              =args.nosave,
            noval               =args.noval,
            noautoanchor        =args.noautoanchor,
            noplots             =args.noplots,
            evolve              =args.evolve,
            evolve_population   =args.evolve_population,
            resume_evolve       =args.resume_evolve,
            bucket              =args.bucket,
            cache               =args.cache,
            image_weights       =args.image_weights,
            device              =args.device,
            multi_scale         =args.multi_scale,
            single_cls          =args.single_cls,
            optimizer           =args.optimizer,
            sync_bn             =args.sync_bn,
            workers             =args.workers,
            project             =args.project,
            name                =args.name,
            exist_ok            =args.exist_ok,
            quad                =args.quad,
            cos_lr              =args.cos_lr,
            label_smoothing     =args.label_smoothing,
            patience            =args.patience,
            freeze              =args.freeze,
            save_period         =args.save_period,
            seed                =args.seed,
            local_rank          =args.local_rank,
            entity              =args.entity,
            upload_dataset      =args.upload_dataset,
            bbox_interval       =args.bbox_interval,
            artifact_alias      =args.artifact_alias,
            ndjson_console      =args.ndjson_console,
            ndjson_file         =args.ndjson_file
        )

        # Log metrics
        mlflow.log_metric('precision', results['precision'])
        mlflow.log_metric('recall', results['recall'])
        mlflow.log_metric('map', results['map'])
        mlflow.log_metric('map50', results['map50'])

        # Save and log the model
        #model_path = 'runs/train/exp/weights/best.pt'
        #mlflow.pytorch.log_model(pytorch_model=model_path, artifact_path="model")
        mlflow.end_run()

if __name__ == "__main__":
    main()
