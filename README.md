# cvsutils
Unofficial utility scripts for Microsoft Custom Vision Service.

## Setup
```sh
pip install cvsutils
export CVS_ENDPOINT='<endpoint>'
export CVS_TRAINING_KEY='<training_key>'
export CVS_PREDICTION_KEY='<prediction_key>'
export CVS_PREDICTION_RESOURCE_ID='<resource_id>'
```

Those keys and endpoint information can be found in the Custom Vision's settings page.

## Available commands

```sh
# Show a list of projects
cvs_list_projects [--verbose]

# Create a new project
cvs_create_project <dataset_filepath> [--project_name <name>] [--domain_id <domain_id>]

# Download dataset from a project
cvs_download_project <project_id> <output_dir>

# Train a model
cvs_train_project <project_id> [--domain_id <domain_id>] [--type {multiclass,multilabel}] [--force]

# Export a model
cvs_export_model <project_id> <iteration_id> {tensorflow,coreml,onnx} [--output_filepath <filepath>]
```

And
* cvs_evaluate_project
* cvs_get_domains
* cvs_predict_image
* cvs_remove_iteration
* cvs_validate_dataset

To see the detailed help, please run the command with "-h" option.

## Dataset file format
This tool uses the SIMPLE dataset format to upload/download datasets from Custom Vision Service.

For details, please see the [simpledataset](https://github.com/shonohs/simpledataset) repository.
