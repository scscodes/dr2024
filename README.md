# DR2024

`python setup.py`


#### Nice-to-have naming convention: 
- files: `re-<algorithm>-<whatever you want>.py`
- notebooks: `re-<algorithm>-<whatever you want>.ipynb`


#### DR on the spot
AWS Console > CloudShell

Create base resources
`./create-base-resources.sh BASE-STACK-NAME YOUR-IP`
-   base stack name = name of cloudformation stack being built
-   your ip = public ip of your machine

Set custom files and params
 `cd custom-files`
- Update config files in this dir to set training
-   hyperparameters, model metadata, reward fn
-   run - world name, model prefix (unique name for each new model)

Create resources: standard or spot (cheapest) instance
`./create-standard-instance.sh` | `./create-spot-instance.sh`
`./create-spot-instance.sh BASE-STACK-NAME TRAINING-STACK-NAME TIME-TO-LIVE`
-   base stack name = from above
-   training stack name = name for the training
-   time to live = # min to keep running before automatic termination

View model on track (after waiting) `<instance ip>:8080`
- If needed, force shutdown via cli `sudo shutdown now`

Increment training (clone) - `custom-files/`
-   run - model prefix=new name, pretrained=True, pretrained_prefix=model to clone

Add users to stack 
`./add-access.sh BASE-STACK-NAME ACCESS-STACK-NAME <public ip>`

##### References
- [tds f1 article](https://towardsdatascience.com/an-advanced-guide-to-aws-deepracer-2b462c37eea)
- [tds repo](https://github.com/dgnzlz/Capstone_AWS_DeepRacer/tree/master)
- [tds repo reference](https://github.com/cdthompson/deepracer-k1999-race-lines/blob/master/Race-Line-Calculation.ipynb)
- [log guru repo](https://github.com/aws-deepracer-community/deepracer-log-guru?tab=readme-ov-file)
- [dr log analysis repo](https://github.com/aws-deepracer-community/deepracer-analysis)
- [run dr locally on linux](https://aws-deepracer-community.github.io/deepracer-for-cloud/)
- [aws docs reward function params](https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-reward-function-input.html)
- [google colab](https://colab.google/)
