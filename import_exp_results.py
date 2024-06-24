import mlflow
import os
import logging
import pandas as pd

def find_folders(starting_string):
    # Get the current working directory
    current_path = os.path.dirname(os.path.realpath(__file__))
    #current_path = os.getcwd()

    # Concatenate the folder name to the current path
    folder_name = "Experiments"
    directory = os.path.join(current_path, folder_name)
    
    matching_folders = []
    # Check if the directory exists
    if os.path.exists(directory):
        # Iterate over the items in the directory
        for dir_name in os.listdir(directory):
            # Create the full path
            full_path = os.path.join(directory, dir_name)
            # Check if it is a directory and starts with the given string
            if os.path.isdir(full_path) and dir_name.startswith(starting_string):
                matching_folders.append(full_path)
                           
    return matching_folders

def load_csv(current_experiment):
    csv_file = os.path.join(current_experiment, "results.csv")
        
    try:
        # Try to load CSV file into a pandas DataFrame
        csv_results = pd.read_csv(csv_file)
        # Display the first few rows of the DataFrame as an example
        print(f"Results from experiment in {current_experiment}:")
        
    except FileNotFoundError:
        print(f"CSV file 'results.csv' not found in {current_experiment}")
        return None
     
    return csv_results
    
def load(User_experiments):
    epoch = 0
    experiments = find_folders(User_experiments)
    print("Loading previous experiments results...")
    for current_experiment in experiments:
        results = load_csv(current_experiment)
        if results is not None:  # Check if results were successfully loaded
            for index, row in results.iterrows():
                
                # Log metrics
                mlflow.log_metric('precision', float(row.iloc[4]), step=epoch)
                mlflow.log_metric('recall', float(row.iloc[5]), step=epoch)
                mlflow.log_metric('map_50', float(row.iloc[6]), step=epoch)
                mlflow.log_metric('map_50-95', float(row.iloc[7]), step=epoch)
                mlflow.log_metric('val_boxloss', float(row.iloc[1]), step=epoch)
                mlflow.log_metric('val_objloss', float(row.iloc[2]), step=epoch)
                mlflow.log_metric('val_clsloss', float(row.iloc[3]), step=epoch)
                epoch+=1
    
    print("Previous expreriments results loaded :D")
    return epoch