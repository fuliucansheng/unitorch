# Roberta Text Classification Tutorial

&nbsp;&nbsp;&nbsp;&nbsp;In this tutorial, we will learn how to perform text classification using the RoBERTa model. RoBERTa is a transformer-based model that has been pre-trained on a large corpus of text and can be fine-tuned for specific downstream tasks like text classification.

## Step 1: Prepare Dataset Files

&nbsp;&nbsp;&nbsp;&nbsp;Prepare a TSV (Tab-Separated Values) file as an example. You can also use JSON, PARQUET or HUB dataset formats. The TSV file should have three columns with the following dataset settings. The "image" column contains the image paths as strings.

## Step 2: Prepare config.ini File

Take this [config](https://github.com/fuliucansheng/unitorch/examples/configs/classification/roberta.ini) as a template.

### Task Settings

```ini
# optim
[core/optim/adamw]
learning_rate = 0.0001

# scheduler
[core/scheduler/linear_warmup]
num_warmup_rate = 0.001

# task
[core/task/supervised]
model = core/model/classification/roberta
optim = core/optim/adamw
scheduler = core/scheduler/linear_warmup
dataset = core/dataset/ast
loss_fn = core/loss/ce
score_fn = core/score/acc
monitor_fns = ['core/score/acc']
output_header = ['query', 'doc']
postprocess_fn = core/postprocess/classification/score
writer = core/writer/csv

from_ckpt_dir = ${core/cli:from_ckpt_dir}
to_ckpt_dir = ${core/cli:cache_dir}
output_path = ${core/cli:cache_dir}/output.txt
train_batch_size = 64
dev_batch_size = 64
test_batch_size = 256
```

!!! note
    * `model`: model to be used in the task.
    * `optim`: optim to be used in the task.
    * `scheduler`: scheduler to be used in the task.
    * `loss_fn`: loss to be used in the task.
    * `score_fn`: metric to be used for saving checkpoints.
    * `monitor_fns`: metrics for logging.
    * `output_header`: save these fields for inference.
    * `postprocess_fn`, `writer`: post-process functions and writer for inference.
    * `{train, dev, test}_batch_size`: batch size settings for train/eval/inference.

### Model Settings

```ini
[core/model/classification/roberta]
pretrained_name = roberta-base
num_classes = 2
```

!!! note
    The options in [core/model/classification/roberta] are settings for the Roberta model.

### Dataset Settings

```ini
[core/dataset/ast]
names = ['query', 'doc', 'label']

[core/dataset/ast/train]
data_files = ${core/cli:train_file}
preprocess_functions = [
    'core/process/roberta/classification(query, doc))',
    'core/process/label(label)'
  ]

[core/dataset/ast/dev]
data_files = ${core/cli:dev_file}
preprocess_functions = [
    'core/process/roberta/classification(query, doc))', 
    'core/process/label(label)'
  ]

[core/dataset/ast/test]
names = ['text', 'image']
data_files = ${core/cli:test_file}
preprocess_functions = [
    'core/process/roberta/classification(query, doc))'
  ]
```

!!! note
    * The options in [core/dataset/ast/train] are settings for the training data.
    * `names` specifies the fields in the dataset. For TSV files, this should be the header.
    * `preprocess_functions` are the preprocess functions used to convert the raw data into tensors as model inputs.
    * These options can be set independently for train, dev, and test.

### Processing Settings

```ini
[core/process/roberta]
pretrained_name = roberta-base
max_seq_length = 24
```

!!! note
    The options in [core/process/roberta] are settings for the Roberta processor used in dataset.


## Step 3: Run Training Command

Use the following command to run the training:

```bash
unitorch-train path/to/config.ini --train_file path/to/train.tsv --dev_file path/to/dev.tsv --core/task/supervised@train_batch_size 128
```

!!! note
    The --core/task/supervised@train_batch_size 128 part of the command overrides the parameter setting in the config file.

Use the following command to run the inference:

```bash
unitorch-infer path/to/config.ini --test_file path/to/test.tsv --from_ckpt_dir path/to/ckpt/folder --core/task/supervised@test_batch_size 128
```

!!! note
    The --core/task/supervised@test_batch_size 128 part of the command overrides the parameter setting in the config file.