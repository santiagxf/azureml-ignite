# Structuring an ML project with software engineering practices

## Running the project

### Run it on Azure ML
1. Clone the repository in your local machine.
1. Install Azure ML CLI v2. If you don't have it, follow the installation instructions at [Install and set up the CLI (v2)](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli).
1. Create a compute named `trainer-cpu` or rename the compute specified in [.aml/jobs/carpricer.job.yml](.aml/jobs/carpricer.job.yml).
1. Register the dataset:

    ```bash
    az ml data create -f .aml/data/product-reviews-train.yml
    az ml data create -f .aml/data/product-reviews-eval.yml
    ``` 
1. Create the training job:

    ```bash
    az ml job create -f .aml/jobs/carpricer.job.yml
    ```

(Optional)

6. Register the trained model in the registry:
    
    ```bash
    JOB_NAME=$(az ml job list --query "[0].name" | tr -d '"')
    az ml model create --name "carpricer" \
                       --type "mlflow_model" \
                       --path "azureml://jobs/$JOB_NAME/outputs/artifacts/pipeline"
    ```
6. Deploy the model in an online endpoint:

    ```bash
    az ml online-endpoint create -f .aml/endpoints/carpricer-online/endpoint.yml
    az ml online-deployment create -f .aml/endpoints/carpricer-online/deployments/default.yml --all-traffic
    ```


## Contributing

This project welcomes contributions and suggestions. Open an issue and start the discussion!
