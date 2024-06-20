import mlflow

def load():
    
    
    # Log metrics
    mlflow.log_metric('precision', results[0], step=epoch)
    mlflow.log_metric('recall', results[1], step=epoch)
    mlflow.log_metric('map_50', results[2], step=epoch)
    mlflow.log_metric('map_50-95', results[3], step=epoch)
    mlflow.log_metric('val_boxloss', results[4], step=epoch)
    mlflow.log_metric('val_objloss', results[5], step=epoch)
    mlflow.log_metric('val_clsloss', results[6], step=epoch)