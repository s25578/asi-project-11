# Tasks

## Sprint 1

* Pick a topic (https://amanxai.com/2022/08/22/password-strength-checker-with-machine-learning/)
* Git repository
* DVC
* Jupyter Notebook with a simple model
* Python package

## Sprint 2

For your home assignment covering ASI Project 2 topics you should do the following:

1. Make an endpoint `/continue-train` that:
    * Accepts as an input:
        - `model-name` - name of a model that should be used for the continuation of the training.
        - `train-input` - a file that contains a data that shall be used for the continuation of the
          training. The file should contain the target column.
        - `new-model-name` - name of a model that was produced at the end of the training.
    * Returns as an output:
        - New model's quality variables (metrics) that were produced during the training.
2. Make an endpoint `/predict` that:
    * Accepts as an input:
        - `model-name` - name of a model that should be used for the predictions.
        - `input` - a file that contains a data that shall be used for the predictions. The file
          should NOT contain the target column.
    * Returns as an output:
        - Predictions.
3. Make an endpoint `/models` that:
    * Returns as an output:
        - A list of model names. You can use a `list[str]` type for that.

Deadline for the home assignment is: 26.12.2025 23:59:59.

## Sprint 3

Today (12.12.2025) we have completed ASI Sprint 3 with the following topics:

* PyCaret
* Optuna

Please find the files with sample PyCaret and Optuna use cases in the attachment.

For your home assignment covering ASI Sprint 3 topics you should do the following:

1. PyCaret - make a Python module that:
* trains PyCaret AutoML models on raw (not processed) data related to your topic.
* picks a best model and outputs predictions using the best model.
* computes scoring for the best model.
2. Optuna -  make a Python module that:
* finds hyperparameters (e.g. learning rate, number of neurons, number of layers, etc.) for your model.

Please ensure that the modules can be executed (e.g. you can create a simple CLI application with https://typer.tiangolo.com/).

Deadline for the home assignment is: 02.01.2026 23:59:59.

## Sprint 4

Today (16.01.2026) we are completing ASI Sprint 4 with the following topics:

* Docker
* Ansible

For your home assignment covering ASI Sprint 4 topics you should do the following:

1. Prepare a *Dockerfile* for your project.
    * After an image is built with the *Dockerfile*, the image should be available for execution
      with a container.
    * The running container should serve your application.
    * If you have a split into frontend and backend part, you can use separate *Dockerfile*s.
2. Prepare Ansible's  *inventory.yaml* and *playbook.yaml* that deploy your project.
    * The *playbook.yaml* should:
        - build images from *Dockerfile*s,
        - copy built images to the remote,
        - start containers with your application based on the built images,
        - configure your application.

Deadline for the home assignment is: 23.01.2026 23:59:59.