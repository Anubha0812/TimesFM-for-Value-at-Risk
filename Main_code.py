import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PMAP_USE_TENSORSTORE'] = 'false'


import timesfm
import gc
import numpy as np
import pandas as pd
from timesfm import patched_decoder
from timesfm import data_loader
from tqdm import tqdm
import dataclasses
import IPython
import IPython.display
from huggingface_hub import snapshot_download


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


fix_gpu()



##################################################################################################
#os.environ['JAX_PLATFORMS'] = 'cpu'   # Set JAX to use CPU

################################################################################################
## specify the input length and output length as context_len and pred_len
##################################################################################################

context_len = 512
pred_len = 1

output_folder = "pre_train_results" ## it will store the pre trained forecasts
output_folder2 = "finetune_results" ## it will store the finetuned model forecasts

 # Calculate the indices for the boundaries
train_size = 0.7  # 70% of the total data for training
val_size = 0.1    # 10% of the total data for validation
test_size = 0.2   # 20% of the total data for testing

# Calculate the indices for boundaries
train_end_index = int(total_data_points * train_size)
val_end_index = int(total_data_points * (train_size + val_size))
boundaries = [train_end_index, val_end_index, total_data_points]

## or you can specify the boundaries manually  boundaries= [1819, 2600, 4876] 

####### mention the quantiles ###############################

QUANTILES = [0.01, 0.025, 0.05, 0.1, 0.5, 0.6, 0.7, 0.8, 0.9]
########################################################## Initialize the model
tfm = timesfm.TimesFm(
    context_len=context_len,
    horizon_len=pred_len,
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
    backend='gpu',  # 'jax' or 'pytorch'
    quantiles=QUANTILES,
)

snapshot_download(local_dir="...specify",cache_dir="...specify", repo_id="google/timesfm-1.0-200m")
tfm.load_from_checkpoint("...spceify",repo_id="google/timesfm-1.0-200m")# Load the checkpoint
##################################################################################################
data_path="datasets/returns.csv"


df=pd.read_csv(data_path)
ts_cols = [col for col in df.columns if col != "Date"]
num_cov_cols = None
cat_cov_cols = None

num_ts = len(ts_cols)
batch_size =16

dtl = data_loader.TimeSeriesdata(
      data_path=data_path,
      datetime_col="Date",
      num_cov_cols=num_cov_cols,
      cat_cov_cols=cat_cov_cols,
      ts_cols=np.array(ts_cols),
      train_range=[0, boundaries[0]],
      val_range=[boundaries[0], boundaries[1]],
      test_range=[boundaries[1], boundaries[2]],
      hist_len=context_len,
      pred_len=pred_len,
      batch_size=num_ts,
#      freq=freq,
      normalize=False,
      epoch_len=None,
      holiday=False,
      permute=True,
  )

train_batches1 = dtl.tf_dataset(mode="train", shift=1).batch(batch_size)
for tbatch in tqdm(train_batches1.as_numpy_iterator()):
    pass
print(tbatch[0].shape,tbatch[3].shape)

total_batches = sum(1 for _ in train_batches1)

train_batches = train_batches1.skip(0).take(total_batches - 1) 
for tbatch in tqdm(train_batches.as_numpy_iterator()):
    pass
print(tbatch[0].shape,tbatch[3].shape)

val_batches = dtl.tf_dataset(mode="val", shift=pred_len)

test_batches = dtl.tf_dataset(mode="test", shift=pred_len)

######################################################### Performing Inference
mae_losses = []
mae1_losses = []
batch_index=1

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


for batch in tqdm(test_batches.as_numpy_iterator()):
    past = batch[0]
    actuals = batch[3]
    _, forecasts1 = tfm.forecast(list(past), [0] * past.shape[0])

    # Save the forecasts for the first column without a suffix
    forecasts = forecasts1[:, 0 : actuals.shape[1], 0]  # First column
    forecasts_filename = os.path.join(output_folder, f'iforecasts_batch_{batch_index}.csv')
    forecast_df = pd.DataFrame(forecasts)
    forecast_df.to_csv(forecasts_filename, index=False)

    # Save the forecasts for the remaining 9 columns with suffixes q1 to q9
    for i in range(1, 10):  # Columns 1 to 9 (q1 to q9)
        forecasts = forecasts1[:, 0 : actuals.shape[1], i]
        forecasts_filename = os.path.join(output_folder, f'forecasts_q{i}_batch_{batch_index}.csv')
        forecast_df = pd.DataFrame(forecasts)
        forecast_df.to_csv(forecasts_filename, index=False)

    mae1_losses.append(np.abs(forecasts1[:, 0 : actuals.shape[1], 0] - actuals).mean())
    mae_losses.append(np.abs(forecasts1[:, 0 : actuals.shape[1], 5] - actuals).mean())

    # Save past data
    #past_filename = os.path.join(output_folder, f'index_2_1_input_batch_{batch_index}.csv')
    #past_df = pd.DataFrame(past)
    #past_df.to_csv(past_filename, index=False)

    # Save actuals data
    actuals_filename = os.path.join(output_folder, f'test_batch_{batch_index}.csv')
    actuals_df = pd.DataFrame(actuals)
    actuals_df.to_csv(actuals_filename, index=False)

    # Increment batch index
    batch_index += 1
    #print(batch_index)

#df.to_csv(f'stock_{i+1}.csv', index=False)
print(f"MAE: {np.mean(mae_losses)}")
print(f"MAE1: {np.mean(mae1_losses)}")
######################################################################################################################
print("Forecasts using pre-trained model")



###############################################################################################################################
### fine tuning part
import jax
from jax import numpy as jnp
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis import base_model
from praxis import optimizers
from praxis import schedules
from praxis import base_hyperparams
from praxis import base_layer
from paxml import tasks_lib
from paxml import trainer_lib
from paxml import checkpoints
from paxml import learners
from paxml import partitioning
from paxml import checkpoint_types

# PAX shortcuts
NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor
NpTensor = pytypes.NpTensor
WeightedScalars = pytypes.WeightedScalars
instantiate = base_hyperparams.instantiate
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
AuxLossStruct = base_layer.AuxLossStruct

AUX_LOSS = base_layer.AUX_LOSS
template_field = base_layer.template_field

# Standard prng key names
PARAMS = base_layer.PARAMS
RANDOM = base_layer.RANDOM

key = jax.random.PRNGKey(seed=1234)   

model = pax_fiddle.Config(
    patched_decoder.PatchedDecoderFinetuneModel,
    name='patched_decoder_finetune',
    core_layer_tpl=tfm.model_p,
)


@pax_fiddle.auto_config
def build_learner() -> learners.Learner:
  return pax_fiddle.Config(
      learners.Learner,
      name='learner',
      loss_name='avg_qloss',
      optimizer=optimizers.Adam(
          epsilon=1e-7,
          clip_threshold=1e2,
          learning_rate=1e-2,
          lr_schedule=pax_fiddle.Config(
              schedules.Cosine,
              initial_value=1e-3,
              final_value=1e-4,
              total_steps=40000,
          ),
          ema_decay=0.9999,
      ),
      # Linear probing i.e we hold the transformer layers fixed.
      bprop_variable_exclusion=['.*/stacked_transformer_layer/.*'],
  )


task_p = tasks_lib.SingleTask(
    name='ts-learn',
    model=model,
    train=tasks_lib.SingleTask.Train(
        learner=build_learner(),
    ),
)

task_p.model.ici_mesh_shape = [1, 1, 1]
task_p.model.mesh_axis_names = ['replica', 'data', 'mdl']

DEVICES = np.array(jax.devices()).reshape([1, 1, 1])
MESH = jax.sharding.Mesh(DEVICES, ['replica', 'data', 'mdl'])

num_devices = jax.local_device_count()
print(f'num_devices: {num_devices}')
print(f'device kind: {jax.local_devices()[0].device_kind}')


jax_task = task_p
key, init_key = jax.random.split(key)

# To correctly prepare a batch of data for model initialization (now that shape
# inference is merged), we take one devices*batch_size tensor tuple of data,
# slice out just one batch, then run the prepare_input_batch function over it.


def process_train_batch(batch):
    past_ts = batch[0].reshape(batch_size * num_ts, -1)
    actual_ts = batch[3].reshape(batch_size * num_ts, -1)
    return NestedMap(input_ts=past_ts, actual_ts=actual_ts)


def process_eval_batch(batch):
    past_ts = batch[0]
    actual_ts = batch[3]
    return NestedMap(input_ts=past_ts, actual_ts=actual_ts)


jax_model_states, _ = trainer_lib.initialize_model_state(
    jax_task,
    init_key,
    process_train_batch(tbatch),
    checkpoint_type=checkpoint_types.CheckpointType.GDA,
)

jax_model_states.mdl_vars['params']['core_layer'] = tfm._train_state.mdl_vars['params']
jax_vars = jax_model_states.mdl_vars
gc.collect()

jax_task = task_p


def train_step(states, prng_key, inputs):
  return trainer_lib.train_step_single_learner(
      jax_task, states, prng_key, inputs
  )


def eval_step(states, prng_key, inputs):
  states = states.to_eval_state()
  return trainer_lib.eval_step_single_learner(
      jax_task, states, prng_key, inputs
  )

key, train_key, eval_key = jax.random.split(key, 3)
train_prng_seed = jax.random.split(train_key, num=jax.local_device_count())
eval_prng_seed = jax.random.split(eval_key, num=jax.local_device_count())

p_train_step = jax.pmap(train_step, axis_name='batch')
p_eval_step = jax.pmap(eval_step, axis_name='batch')

replicated_jax_states = trainer_lib.replicate_model_state(jax_model_states)
replicated_jax_vars = replicated_jax_states.mdl_vars


best_eval_loss = 1e7
step_count = 0
patience = 0
NUM_EPOCHS = 100
PATIENCE = 5
TRAIN_STEPS_PER_EVAL = 1000
CHECKPOINT_DIR='finetune' ###########################################################################


def reshape_batch_for_pmap(batch, num_devices):
  def _reshape(input_tensor):
    bsize = input_tensor.shape[0]
    residual_shape = list(input_tensor.shape[1:])
    nbsize = bsize // num_devices
    return jnp.reshape(input_tensor, [num_devices, nbsize] + residual_shape)

  return jax.tree.map(_reshape, batch)


for epoch in range(NUM_EPOCHS):
    print(f"__________________Epoch: {epoch}__________________", flush=True)
    train_its = train_batches.as_numpy_iterator()
    if patience >= PATIENCE:
        print("Early stopping.", flush=True)
        break
    for batch in tqdm(train_its):
        train_losses = []
        if patience >= PATIENCE:
            print("Early stopping.", flush=True)
            break
        tbatch = process_train_batch(batch)
        tbatch = reshape_batch_for_pmap(tbatch, num_devices)
        replicated_jax_states, step_fun_out = p_train_step(
            replicated_jax_states, train_prng_seed, tbatch
        )
        train_losses.append(step_fun_out.loss[0])
        if step_count % TRAIN_STEPS_PER_EVAL == 0:
            print(
                f"Train loss at step {step_count}: {np.mean(train_losses)}",
                flush=True,
            )
            train_losses = []
            print("Starting eval.", flush=True)
            val_its = val_batches.as_numpy_iterator()
            eval_losses = []
            for ev_batch in tqdm(val_its):
                ebatch = process_eval_batch(ev_batch)
                ebatch = reshape_batch_for_pmap(ebatch, num_devices)
                _, step_fun_out = p_eval_step(
                    replicated_jax_states, eval_prng_seed, ebatch
                )
                eval_losses.append(step_fun_out.loss[0])
            mean_loss = np.mean(eval_losses)
            print(f"Eval loss at step {step_count}: {mean_loss}", flush=True)
            if mean_loss < best_eval_loss or np.isnan(mean_loss):
                best_eval_loss = mean_loss
                print("Saving checkpoint.")
                jax_state_for_saving = py_utils.maybe_unreplicate_for_fully_replicated(
                    replicated_jax_states
                )
                checkpoints.save_checkpoint(
                    jax_state_for_saving, CHECKPOINT_DIR, overwrite=True
                )
                patience = 0
                del jax_state_for_saving
                gc.collect()
            else:
                patience += 1
                print(f"patience: {patience}")
        step_count += 1
        
        
train_state = checkpoints.restore_checkpoint(jax_model_states, CHECKPOINT_DIR)
print(train_state.step)
tfm._train_state.mdl_vars['params'] = train_state.mdl_vars['params']['core_layer']
tfm.jit_decode()

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
mae_losses = []
mae1_losses = []
batch_index=1

if not os.path.exists(output_folder2):
    os.makedirs(output_folder2)


for batch in tqdm(test_batches.as_numpy_iterator()):
    past = batch[0]
    actuals = batch[3]
    _, forecasts1 = tfm.forecast(list(past), [0] * past.shape[0])

    # Save the forecasts for the first column without a suffix
    forecasts = forecasts1[:, 0 : actuals.shape[1], 0]  # First column
    forecasts_filename = os.path.join(output_folder2, f'forecasts_batch_{batch_index}.csv')
    forecast_df = pd.DataFrame(forecasts)
    forecast_df.to_csv(forecasts_filename, index=False)

    # Save the forecasts for the remaining 9 columns with suffixes q1 to q9
    for i in range(1, 10):  # Columns 1 to 9 (q1 to q9)
        forecasts = forecasts1[:, 0 : actuals.shape[1], i]
        forecasts_filename = os.path.join(output_folder2, f'forecasts_q{i}_batch_{batch_index}.csv')
        forecast_df = pd.DataFrame(forecasts)
        forecast_df.to_csv(forecasts_filename, index=False)

    mae1_losses.append(np.abs(forecasts1[:, 0 : actuals.shape[1], 0] - actuals).mean())
    mae_losses.append(np.abs(forecasts1[:, 0 : actuals.shape[1], 5] - actuals).mean())

    # Save past data
    #past_filename = os.path.join(output_folder2, f'index_2_1_input_batch_{batch_index}.csv')
    #past_df = pd.DataFrame(past)
    #past_df.to_csv(past_filename, index=False)

    # Save actuals data
    actuals_filename = os.path.join(output_folder2, f'test_batch_{batch_index}.csv')
    actuals_df = pd.DataFrame(actuals)
    actuals_df.to_csv(actuals_filename, index=False)

    # Increment batch index
    batch_index += 1
    print(batch_index)

#df.to_csv(f'stock_{i+1}.csv', index=False)
print(f"MAE: {np.mean(mae_losses)}")
print(f"MAE1: {np.mean(mae1_losses)}")
######################################################################################################################
######################################################################################################################
print("Finetune Completed")

