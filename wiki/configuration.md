---
hide:
  - toc
---
# Configuration

&nbsp;&nbsp;&nbsp;&nbsp;unitorch command workflow for modeling training/inference is using a unified configuration system. In this example, we'll explore the configuration of the BartForGeneration class in unitorch.

&nbsp;&nbsp;&nbsp;&nbsp; Here is a training command example for bart generation with local files.

```bash
unitorch-train \
    configs/generation/bart.ini \
    --train_file train.tsv \
    --dev_file dev.tsv \
    --core/model/generation/bart@num_beams 20 \
    --core/model/generation/bart@no_repeat_ngram_size 0
```

&nbsp;&nbsp;&nbsp;&nbsp;This is the training command for bart generation. In this command, we provide the path to the configuration file `configs/core/generation/bart.ini` and specify additional parameters using the `--` syntax. The parameters provided after `--core/model/generation/bart@` override the corresponding values in the configuration file. In this example, we override `num_beams` to **20** and `no_repeat_ngram_size` to **0**. The configuration file using an INI file format is as follows.

```ini
# model
[core/model/generation/bart]
pretrained_name = bart-base
no_repeat_ngram_size = 3
max_gen_seq_length = 15

# dataset
[core/dataset/ast]
names = ['encode', 'decode']

# ...
```

In this configuration, we specify the following parameters:

* `pretrained_name`: The name of the pretrained model. In this example, it is set to **bart-base**.
* `no_repeat_ngram_size`: The size of n-grams to avoid repeating in the generated sequences. It is set to **3** in this example. Because we override this parameter in command line, it would be set to **0** finally.
* `max_gen_seq_length`: The maximum length of the generated sequences. It is set to **15** in this example.

Then let's check the BartForGenertaion model class code.

```python
class BartForGeneration(_BartForGeneration):
    def __init__(
        self,
        config_path: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        pass

    @classmethod
    @add_default_section_for_init("core/model/generation/bart")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/generation/bart")
        pretrained_name = config.getoption("pretrained_name", "default-bart")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_bart_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(config_path, gradient_checkpointing)
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_bart_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @add_default_section_for_function("core/model/generation/bart")
    def generate(
        self,
        input_ids: torch.Tensor,
        num_beams: Optional[int] = 5,
        decoder_start_token_id: Optional[int] = 2,
        decoder_end_token_id: Optional[int] = 2,
        num_return_sequences: Optional[int] = 1,
        min_gen_seq_length: Optional[int] = 0,
        max_gen_seq_length: Optional[int] = 48,
        repetition_penalty: Optional[float] = 1.0,
        no_repeat_ngram_size: Optional[int] = 0,
        early_stopping: Optional[bool] = True,
        length_penalty: Optional[float] = 1.0,
        num_beam_groups: Optional[int] = 1,
        diversity_penalty: Optional[float] = 0.0,
        do_sample: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 1.0,
    ):
        pass

```

&nbsp;&nbsp;&nbsp;&nbsp;The `from_core_configure` method is a class method used to create an instance of BartForGeneration based on a provided configuration object (config). It retrieves various options from the configuration and initializes the instance with the appropriate values. It also loads pretrained weights if a pretrained_weight_path is specified in the configuration. It also has `add_default_section_for_function` decorator to override the parameter value from coniguration object with specific section. The `num_beams` is set to **20** and `no_repeat_ngram_size` is set to 3 in this example.