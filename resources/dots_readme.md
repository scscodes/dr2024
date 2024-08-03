# [DR on the spot](https://github.com/aws-deepracer-community/deepracer-on-the-spot?tab=readme-ov-file)
`BASE_STACK_NAME` = name of cloud formation stack to build  
`YOUR_IP` = public IP  
`TRAINING_STACK_NAME` = name assigned to training  
`TIME_TO_LIVE` = # min to run before termination

### AWS Console > CloudShell

### Create base resources
`./create-base-resources.sh <BASE_STACK_NAME> <YOUR_IP>`
- this process will create an s3 bucket
- if you are going to import to:from DR Console, the DOTS steps should use this created s3 bucket uri

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