# DR2024

`python setup.py`


## Nice-to-have naming convention:
- files: `re-<algorithm>-<whatever you want>.py`
- notebooks: `re-<algorithm>-<whatever you want>.ipynb`


## DR on the spot
`BASE_STACK_NAME` = name of cloud formation stack to build  
`YOUR_IP` = public IP  
`TRAINING_STACK_NAME` = name assigned to training  
`TIME_TO_LIVE` = # min to run before termination

### AWS Console > CloudShell

### Create base resources
`./create-base-resources.sh <BASE_STACK_NAME> <YOUR_IP>`

### Update custom files: params and env
`cd custom-files`
- hyperparameters, model metadata, reward fn
- `run.env` - track (world), race type, model prefix

### Create resources: standard or spot (cheapest) instance
`./create-standard-instance.sh` | `./create-spot-instance.sh`  
Ex: `./create-spot-instance.sh <BASE_STACK_NAME> <TRAINING_STACK_NAME> <TIME_TO_LIVE>`

### Increment training (clone) - `custom-files/`
-   run - model prefix=new name, pretrained=True, pretrained_prefix=model to clone

### View stack menu (after running)
- Video feeds, grafana dashboards, config files and logs...
- `<instance public ip>:8100/menu.html`

### Other
- track names (for run.env); `<name>.npy` minus the npy: [here](https://github.com/aws-deepracer-community/deepracer-race-data/tree/main/raw_data/tracks)
- add users to stack: `./add-access.sh BASE-STACK-NAME ACCESS-STACK-NAME <public ip>`
- force shutdown of resource: `sudo shutdown now`

### References
- [tds f1 article](https://towardsdatascience.com/an-advanced-guide-to-aws-deepracer-2b462c37eea)
- [tds repo](https://github.com/dgnzlz/Capstone_AWS_DeepRacer/tree/master)
- [tds repo reference](https://github.com/cdthompson/deepracer-k1999-race-lines/blob/master/Race-Line-Calculation.ipynb)
- [log guru repo](https://github.com/aws-deepracer-community/deepracer-log-guru?tab=readme-ov-file)
- [dr log analysis repo](https://github.com/aws-deepracer-community/deepracer-analysis)
- [run dr locally on linux](https://aws-deepracer-community.github.io/deepracer-for-cloud/)
- [aws docs reward function params](https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-reward-function-input.html)
- [google colab](https://colab.google/)
