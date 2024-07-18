# Blip Image Caption Tutorial

&nbsp;&nbsp;&nbsp;&nbsp;In this tutorial, we will learn how to perform image caption using the Blip model. Blip is a transformer-based model that has been pre-trained on a large corpus of text/image and can be fine-tuned for specific downstream tasks like multi-modal classification & image caption.

## Step 1: Prepare Dataset Files

&nbsp;&nbsp;&nbsp;&nbsp;Prepare a TSV (Tab-Separated Values) file as an example. You can also use JSON, PARQUET or HUB dataset formats. The TSV file should have three columns with the following dataset settings. The "caption" column is the model target. The "image" column contains the image path string in the zip files. You need to start an image service first.

## Step 2: Prepare config.ini File

Take this [config](https://github.com/fuliucansheng/unitorch/examples/configs/caption/blip.ini) as a template.

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
model = core/model/caption/blip
optim = core/optim/adamw
scheduler = core/scheduler/linear_warmup
dataset = core/dataset/ast
loss_fn = core/loss/lm
score_fn = core/score/bleu
monitor_fns = ['core/score/bleu', 'core/score/rouge1', 'core/score/rouge2', 'core/score/rougel']
output_header = ['image']
postprocess_fn = core/postprocess/blip/detokenize
writer = core/writer/csv

from_ckpt_dir = ${core/cli:from_ckpt_dir}
to_ckpt_dir = ${core/cli:cache_dir}
output_path = ${core/cli:cache_dir}/output.txt
train_batch_size = 4
dev_batch_size = 8
test_batch_size = 8
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
[core/model/caption/blip]
pretrained_name = blip-image-captioning-base
no_repeat_ngram_size = 3
max_gen_seq_length = 15
```

!!! note
    The options in [core/model/caption/blip] are settings for the Blip model.

### Dataset Settings

```ini
[core/dataset/ast]
names = ['caption', 'image']

[core/dataset/ast/train]
data_files = ${core/cli:train_file}
preprocess_functions = [
    'core/process/blip/generation(caption, core/process/image/read(image))',
  ]

[core/dataset/ast/dev]
data_files = ${core/cli:dev_file}
preprocess_functions = [
    'core/process/blip/image_classification(core/process/image/read(image))',
    'core/process/blip/generation/labels(caption)',
  ]

[core/dataset/ast/test]
names = ['text', 'image']
data_files = ${core/cli:test_file}
preprocess_functions = [
    'core/process/blip/image_classification(core/process/image/read(image))'
  ]
```

!!! note
    * The options in [core/dataset/ast/train] are settings for the training data.
    * `names` specifies the fields in the dataset. For TSV files, this should be the header.
    * `preprocess_functions` are the preprocess functions used to convert the raw data into tensors as model inputs.
    * These options can be set independently for train, dev, and test.

### Processing Settings

```ini
[core/process/blip]
pretrained_name = blip-image-captioning-base
max_gen_seq_length = 15

[core/process/image]
http_url = http://0.0.0.0:11230/?file={0}
```

!!! note
    * The options in [core/process/blip] are settings for the Blip processor used in dataset.
    * The options in [core/process/image] are settings for the Image processor to read image from the local service.


## Step 3: Run Training Command

Start local image service first:

```bash
unitorch-service start path/to/zip/image/service.ini --zip_folder path/to/zip/folder
```

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