import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def prepare_data():
    print("Fetching and splitting California Housing data...")
    california = fetch_california_housing() 
    X, y = california.data, california.target 
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42 
    )
    return X_train, X_val, y_train, y_val 

def execute_run(run_name, max_depth, learning_rate, X_train, X_val, y_train, y_val): 
    with mlflow.start_run(run_name=run_name): 
        params = {
            "max_depth": max_depth, 
            "learning_rate": learning_rate, 
            "objective": "reg:squarederror", 
            "eval_metric": "rmse" 
        }
        mlflow.log_params(params) 
        
        dtrain = xgb.DMatrix(X_train, label=y_train) 
        dval = xgb.DMatrix(X_val, label=y_val) 
        
        model = xgb.train(
            params=params, 
            dtrain=dtrain, 
            num_boost_round=100, 
            evals=[(dtrain, "train"), (dval, "val")],
            verbose_eval=False 
        )
        
        preds = model.predict(dval)
        
        # Calculate metrics required by the task
        rmse = np.sqrt(mean_squared_error(y_val, preds)) 
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Log the trained model
        mlflow.xgboost.log_model(model, artifact_path="xgboost-model") 
        print(f"{run_name} Complete | Val RMSE: {rmse:.4f}") 

def main(): 
    mlflow.set_tracking_uri("http://127.0.0.1:5000") 
    mlflow.set_experiment("California_Housing_Optimization") 
    
    X_train, X_val, y_train, y_val = prepare_data() 
    
    run_configs = [
        {"name": "Run 1", "depth": 3, "lr": 0.1},
        {"name": "Run 2", "depth": 5, "lr": 0.05}, 
        {"name": "Run 3", "depth": 7, "lr": 0.01}, 
    ]
    
    for config in run_configs: 
        execute_run(config["name"], config["depth"], config["lr"], X_train, X_val, y_train, y_val)

if __name__ == "__main__": 
    main() 