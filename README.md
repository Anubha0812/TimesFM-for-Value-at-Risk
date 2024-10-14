
# TimesFM-for-Value-at-Risk
# Forecast Evaluation using TimesFM

This project evaluates forecast results using the TimesFM model. The TimesFM repository and model checkpoint are used to load and evaluate forecasts, allowing for customization of quantiles and input/output lengths as per user requirements.

## Overview

The project utilizes the [TimesFM repository](https://github.com/google-research/timesfm) by Google for forecast evaluation. You can adjust key parameters like quantiles and specify the input (`context_len`) and output (`pred_len`) lengths to suit your forecasting needs.

### Key Parameters
- Input Length (`context_len`): Set to 512 by default, which specifies the number of input time steps.
- Output Length (`pred_len`): Set to 1 by default, indicating the number of steps to forecast.
  
### Usage
To set up the TimesFM model for evaluation, use the following commands to download and load the checkpoint:

# Download model checkpoint
snapshot_download(local_dir="...specify", cache_dir="...specify", repo_id="google/timesfm-1.0-200m")

# Load the checkpoint
tfm.load_from_checkpoint("...specify", repo_id="google/timesfm-1.0-200m")


### Quantile Adjustments
The model allows you to evaluate forecasts across various quantiles. By default, the quantiles supported are 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, and 0.9. If other quantiles are required, you will need to modify the default quantile settings in the code.

### Finetuned Model
The fine-tuned model is saved in finetune folder. It can be further used for increamental finetuning.

### Requirements
Ensure you have the necessary dependencies from the TimesFM repository. Refer to the [TimesFM GitHub page](https://github.com/google-research/timesfm) for detailed setup instructions.

## License
This project uses the TimesFM repository from Google, which is subject to its respective license. Please review their license terms for compliance.


