# Python Custom Trainer Plugin

Custom python trainer plugin for Unity


## Getting Started
---
### Installing the package:
1) Setting up virtual environment:
    1) Make a venv
    ```
    python -m venv .venv
    ```
    2) source the virtual environment
        Windows:
        ```
        source .venv/Scripts/activate
        ```
        linux:
        ```
        source .venv/bin/activate
        ```
2) Installing the requirements
    ```
    pip install -r requirements.txt
    ```
### Running the program:
1) Run the `mlagents-learn` function
    ```
    mlagents-learn <path_to_yaml>
                    --run-id <run_name>
                    --env <path_to_unity_environment_executable>
                    --torch-device cuda:0
    ```
    - example configuration
    ```
    mlagents-learn ./mlagents_trainer_plugin/a2c/a2c_3DBall.yaml
                    --run-id ExampleRun
                    --env /v/Unity/Unity\ Builds/UnityEnvironment.exe 
                    --torch-device cuda:0
    ```




