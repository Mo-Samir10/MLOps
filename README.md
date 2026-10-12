# Applied MLOps Pipeline: California Housing Predictor

## 1. Description of the Project
This project demonstrates an end-to-end Machine Learning Operations (MLOps) pipeline. It is divided into two main phases:
* **Phase 1: Experiment Tracking.** Training an XGBoost regression model on the California Housing dataset to predict house prices, using MLflow to track parameters, metrics (RMSE, MAE, R²), and model artifacts[cite: 1, 2].
* **Phase 2: Deployment.** Containerizing the best-performing model using FastAPI and Docker, and deploying it locally using a Kubernetes cluster via Minikube[cite: 8, 9, 10].

## 2. Repository Structure
* `train.py`: Script to load data, train three XGBoost experiments, and log metrics/artifacts to MLflow[cite: 3].
* `download_model.py`: Script to fetch the registered "staging" model from the MLflow registry[cite: 8].
* `app.py`: FastAPI application serving the loaded XGBoost model[cite: 8, 9].
* `requirements.txt`: Pinned Python dependencies for reproducible environments[cite: 9].
* `Dockerfile` & `.dockerignore`: Configuration for building the model-serving container[cite: 9].
* `k8s-deployment.yaml`: Kubernetes manifest for deploying the API with load balancing and replicas[cite: 9, 10].
* `mlflow_results.png`: Screenshot of the MLflow comparison UI[cite: 4].

---

## Phase 1: Experiment Tracking with MLflow

### How to Run the Training Pipeline
1. Create and activate a virtual environment:
   `python3 -m venv mlops-env`
   `source mlops-env/bin/activate` (On Windows: `mlops-env\Scripts\activate`)[cite: 5]
2. Install the dependencies:
   `pip install -r requirements.txt`[cite: 5]
3. Start the local MLflow server in a separate terminal:
   `mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 5000`[cite: 5]
4. Run the training script:
   `python train.py`[cite: 7]
5. Register the best model as `California_Housing_Predictor` and tag it with the `staging` alias in the MLflow UI[cite: 7, 8].

### MLflow Experiment Results

| Run | Max Depth | Learning Rate | RMSE | MAE | R² |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Run 1 | 3 | 0.1 | 0.539 | 0.368 | 0.779 |
| Run 2 | 5 | 0.05 | 0.522 | 0.355 | 0.792 |
| Run 3 | 7 | 0.01 | 0.700 | 0.533 | 0.626 |

### Selected Best Model
The best model is **Run 2** (Max Depth: 5, Learning Rate: 0.05). It was selected because it produced the lowest Root Mean Squared Error (RMSE) of 0.522, which was the primary metric designated for model selection[cite: 3].

---

## Phase 2: Deployment with Docker and Kubernetes

### How to Deploy the API
1. Download the registered staging model to your local directory:
   `python download_model.py`[cite: 8]
2. Start Minikube with the Docker driver:
   `minikube start --driver docker`[cite: 10]
3. Build the Docker image directly inside the Minikube registry:
   `minikube image build -t california-housing-api:v1 .`[cite: 10]
4. Apply the Kubernetes deployment and service resources:
   `kubectl apply -f k8s-deployment.yaml`[cite: 10]
5. Verify the pods are running (wait for 1/1 READY):
   `kubectl get pods`[cite: 11]

### Testing the Deployment
1. Expose the Kubernetes service to your local machine:
   `kubectl port-forward svc/housing-api-service 8080:80`[cite: 11]
2. Send a POST request to the `/predict` endpoint using `curl` or Postman[cite: 11]:
   ```bash
   curl -X 'POST' \
     '[http://127.0.0.1:8080/predict](http://127.0.0.1:8080/predict)' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{
     "MedInc": 8.3252,
     "HouseAge": 41.0,
     "AveRooms": 6.9841,
     "AveBedrms": 1.0238,
     "Population": 322.0,
     "AveOccup": 2.5556,
     "Latitude": 37.88,
     "Longitude": -122.23
   }'