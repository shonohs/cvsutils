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

* cvs_create_project
* cvs_download_project
* cvs_evaluate_project
* cvs_export_model
* cvs_get_domains
* cvs_list_projects
* cvs_predict_image
* cvs_remove_iteration
* cvs_train_project
* cvs_validate_dataset

To see the detailed help, please run the command with "-h" option.

## Dataset file format
We use a custom dataset format to upload/download datasets from Custom Vision Service. A dataset consists of the following files:
* A txt file with a list of paths to image/label files
* [Optional] A txt file for label names. The file name should be "labels.txt".
* Zip files which contain image files
* [For object deteion] Zip files which contain label files

The main txt file contains references for the zip files. To upload a dataset, you can just specify the path to the main txt file.

### Image Classification
The format of the main txt file is:
```
<image_filename><space><comma-separated label ids>
```

### Object Detection
The format of the main txt file is:
```
<image_filename><space><label_filename>
```

The format of the label files is:
```
<label id> <left> <top> <right> <bottom>
```
Note that the coordinates of the bounding boxes are not normalized.