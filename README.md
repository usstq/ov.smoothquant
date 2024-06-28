# ov.smoothquant

[SmoothQuant](https://github.com/mit-han-lab/smoothquant) is a full INT8(W8A8) LLM quantization methods which can potentially use the full power of INT8 acceleration capability of HW (for example, AMX-INT8 on SPR case).

But it's not easy to keep accuracy of the quantized model, Intel® Neural Compressor has to implement [a special enhancement](https://github.com/intel/neural-compressor/blob/master/docs/source/smooth_quant.md#our-enhancement) for the accuracy-drop of quantized model to be within 1%.

OpenVINO backend for HF pipeline is [optimum-intel](https://github.com/huggingface/optimum-intel/), and the model exported by `optimum-cli` only support weight-only quantization of int8(or int4).

according to [neural-compressor](https://github.com/intel/neural-compressor/blob/master/docs/source/smooth_quant.md#validated-models):

| Model/Last token accuracy |  FP32 Accuracy   | INT8 (w/ SmoothQuant) | Notes | ours |
|:----------:|:------:|:------:|-----------------------------------|:------:|
| bigscience/bloomz-560m  | 0.3947 | 0.3930 | alpha=0.8, Ipex 2.1  | 0.3914 `-a 0.8 -skip lm_head` |
| facebook/opt-2.7b       | 0.6365 | 0.6404 | alpha=0.5, Ipex 2.0  | 0.6390 `-a 0.5 -skip lm_head` |
| databricks/dolly-v1-6b* | 0.6866 | 0.6895 | alpha=0.8, Ipex 2.1  | 0.6975 `-a 0.85 -skip lm_head` |
| LLaMa-2-7b-hf*          | 0.7392 | 0.7335 | alpha=Auto, Ipex 2.1 | 0.7394 `-a 0.85 -skip_act to/Convert mlp.down_proj` |
| gpt2-medium             | 0.4306 |        |                      | 0.4002 `-a 0.8 -skip_act lm_ head .3.mlp.c_proj` |
validattion tools:
 - https://github.com/EleutherAI/lm-evaluation-harness
 - commit e5e5ee0cb629c9c88165292d1b4bf34623392d33

## openai-community/gpt2-medium
```bash
$ optimum-cli export openvino --fp16 --task text-generation-with-past -m  openai-community/gpt2-medium ./models/gpt2-medium-ov
$ lm_eval --model openvino --tasks lambada_openai --model_args pretrained=./models/gpt2-medium-ov,ov_config=ov_config.json
|lambada_openai|      1|none  |     0|acc       |↑  | 0.4306|±  |0.0069|
$ python ./ov_smoothquant/calibration.py -m models/gpt2-medium-ov act_scales/gpt2-medium.pickle
$ python ./ov_smoothquant/quant.py -m models/gpt2-medium-ov -s act_scales/gpt2-medium.pickle -o ./models/gpt2-medium-SQ
# -a 0.6 -skip_act lm_head
|lambada_openai|      1|none  |     0|acc       |↑  | 0.3862|±  |0.0068|
# -a 0.8 -skip_act lm_head
|lambada_openai|      1|none  |     0|acc       |↑  | 0.4000|±  |0.0068|
# -a 0.8 -skip_act lm_ head .3.mlp.c_proj
|lambada_openai|      1|none  |     0|acc       |↑  | 0.4002|±  |0.0068|
```

## bloomz-560m
```bash
$ optimum-cli export openvino --fp16 --task text-generation-with-past -m bigscience/bloomz-560m/ bloomz-560m-ov
$ python ./ov_smoothquant/calibration.py -m /home/tingqian/tingqian/models/bloomz-560m-ov act_scales/bloomz-560m.pickle
$ python ./ov_smoothquant/quant.py -m ./models/bloomz-560m-ov -s act_scales/bloomz-560m.pickle -o ./models/bloomz-560m-SQ -a 0.8 -skip lm_head
$ lm_eval --model openvino --model_args pretrained=./models/bloomz-560m-SQ --tasks lambada_openai
|lambada_openai|      1|none  |     0|acc       |↑  | 0.3128|±  |0.0065|
$ lm_eval --model openvino --model_args pretrained=./models/bloomz-560m-SQ,ov_config=ov_config.json --tasks lambada_openai
|lambada_openai|      1|none  |     0|acc       |↑  | 0.3914|±  |0.0068|
```

## facebook/opt-2.7b
```bash
$ optimum-cli export openvino --fp16 --task text-generation-with-past -m facebook/opt-2.7b ./models/opt-2.7b-ov
$ lm_eval --model openvino --tasks lambada_openai --model_args pretrained=./models/opt-2.7b-ov/,ov_config=ov_config.json
|lambada_openai|      1|none  |     0|acc       |↑  |0.6367|±  |0.0067|
$ python ./ov_smoothquant/calibration.py -m=./models/opt-2.7b-ov/ act_scales/opt-2.7b.pickle
$ python ./ov_smoothquant/quant.py -m=./models/opt-2.7b-ov/ -s act_scales/opt-2.7b.pickle -o ./models/opt-2.7b-SQ -a 0.5 -skip lm_head
|lambada_openai|      1|none  |     0|acc       |↑  |0.6390|±  |0.0067|
```

## databricks/dolly-v1-6b
```bash
$ optimum-cli export openvino --fp16 --task text-generation-with-past -m databricks/dolly-v1-6b ./models/dolly-v1-6b-ov
$ lm_eval --model openvino --tasks lambada_openai --model_args pretrained=./models/dolly-v1-6b-ov,ov_config=ov_config.json
|lambada_openai|      1|none  |     0|acc       |↑  |0.6866|±  |0.0065|
$ python ./ov_smoothquant/calibration.py -m=./models/dolly-v1-6b-ov/ act_scales/dolly-v1-6b.pickle
$ python ./ov_smoothquant/quant.py -m=./models/dolly-v1-6b-ov/ -s act_scales/dolly-v1-6b.pickle  -o ./models/dolly-v1-6b-SQ -a 0.8 -skip lm_head
|lambada_openai|      1|none  |     0|acc       |↑  |0.6788|±  |0.0065| 
$ python ./ov_smoothquant/quant.py -m=./models/dolly-v1-6b-ov/ -s act_scales/dolly-v1-6b.pickle  -o ./models/dolly-v1-6b-SQ -a 0.85 -skip lm_head
|lambada_openai|      1|none  |     0|acc       |↑  |0.6975|±  |0.0064|
```

## meta-llama/Llama-2-7b-hf
```bash
$ HF_TOKEN=xxx optimum-cli export openvino --fp16 --task text-generation-with-past -m meta-llama/Llama-2-7b-hf ./models/Llama-2-7b-hf-ov
$ lm_eval --model openvino --tasks lambada_openai --model_args pretrained=./models/Llama-2-7b-hf-ov,ov_config=ov_config.json
|lambada_openai|      1|none  |     0|acc       |↑  |0.7392|±  |0.0061|
python ./ov_smoothquant/ppl.py --f32 -x16 -m ./models/Llama-2-7b-hf-ov
PPL: 5.63 @ chunk 512/8192: 100%|████████████████| 41/41 [02:18<00:00,  3.38s/it]
$ python ./ov_smoothquant/calibration.py -m=./models/Llama-2-7b-hf-ov act_scales/Llama-2-7b-hf.pickle
$ python ./ov_smoothquant/quant.py -m=./models/Llama-2-7b-hf-ov -s act_scales/Llama-2-7b-hf.pickle  -o ./models/Llama-2-7b-hf-SQ -a 0.85 -skip to/Convert
$ python ./ov_smoothquant/ppl.py --f32 -x16 -m ./models/Llama-2-7b-hf-SQ
PPL: 6.04 @ chunk 512/8192: 100%|████████████████| 41/41 [00:32<00:00,  1.27it/s]

# mlp.down_proj is difficult to quantize (official smoothquant didn't apply smooth-quant to them at all before quantization, I don't know why),
# eliminate them from quantization list can boost accuracy, another tricky thing is per-token dynamic quantization of activations of down_proj
# seems to be also very helpful to preserve accuracy, but requires special kernels
# but mlp.down_proj & lm_head can still use weight-only compression
$ python ./ov_smoothquant/quant.py -m=./models/Llama-2-7b-hf-ov -s act_scales/Llama-2-7b-hf.pickle  -o ./models/Llama-2-7b-hf-SQ -a 0.85 -skip_act to/Convert mlp.down_proj
PPL: 5.65 @ chunk 512/8192: 100%|████████████████| 41/41 [00:51<00:00,  1.25s/it]
|lambada_openai|      1|none  |     0|acc       |↑  |0.7394|±  |0.0061|
```

## Llama-2-7b
To keep accuracy better, we need:
 - lm_head
 - weight must be per-OC INT8-quantized (symmetrically)
 - Using relatively high alpha `alpha=0.85`
 - must use per-token quantization for activation (at least for mlp.down_proj)
   or skip quantizing activations of mlp.down_proj layers:
 - mlp.down_proj has very large absmax (>100), and must be calculated separately using FP16/FP32/BF16 like [LLM.int8()](https://arxiv.org/abs/2208.07339)

## gpt-j-6b

## openai-community/gpt2-medium

```bash
# export text generation model
$ python3 ./tools/gpt2-textgen.py -e openai-community/gpt2-medium ./models/gpt2-medium-ov
model openai-community/gpt2-medium is exported to ./models/gpt2-medium-ov
# raw accuracy
$ python ./ov_smoothquant/ppl.py -m=./models/gpt2-medium-ov/ -c 128
PPL: 28.72 @ chunk 128/128: 100%|█████████████| 2237/2237 [01:12<00:00, 30.97it/s]
# calibration
python ./ov_smoothquant/calibration.py -m=./models/gpt2-medium-ov/ act_scales/gpt2.pickle
# quantize
python ./ov_smoothquant/quant.py -m=./models/gpt2-medium-ov/ -s ./act_scales/gpt2.pickle -o ./models/gpt2-med-SQ -a 0.6 -othr 100 -ppl ./wikitext-2-raw/wiki.test.raw
#new accuracy
python ./ov_smoothquant/ppl.py -m=./models/gpt2-med-SQ -c 128
PPL: 29.20
```

## Command lines

```bash
# download wikitext-2
wget https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip

# export ov LLM model IR (fp16)
optimum-cli export openvino --fp16 --task text-generation-with-past -m ./meta-llama/Llama-2-7b-hf/ Llama-2-7b-hf-ov

# calibration: generate abs max for each activations of Linear/FC layer
python ./ov_smoothquant/generate_act_scales.py -c=wikitext-2-raw/wiki.test.raw -m ~/models/Llama-2-7b-hf-ov/ -s 128 ./act_scales/Llama-2-7b-hf.pickle

# quantize using SmoothQuant
python ./ov_smoothquant/quant.py -m ~/models/Llama-2-7b-hf-ov/ -s ./act_scales/Llama-2-7b-hf.pickle -o ./models/Llama-2-7b-hf-SQ

python ./ov_smoothquant/quant.py -m /home/sdp/huyuan/dlboost_models/llama-2-7b-chat/pytorch/FP32/ -s ./act_scales/Llama-2-7b-hf.pickle -o ./models/Llama-2-7b-hf-SQ

# gpt-j-6b PPL: 13.18
python ./ov_smoothquant/quant.py -m /home/sdp/huyuan/dlboost_models/gpt-j-6b/pytorch/FP32/ -s ./act_scales/gpt-j-6b.pickle -o ./models/gpt-j-6b-SQ -a 0.85 --skip h.2.mlp.fc_out

# evaluate PPL
python ov_smoothquant/eval.py /home/sdp/huyuan/dlboost_models/gpt-j-6b/pytorch/FP32/ -ppl wikitext-2-raw/wiki.test.raw -c 128
PPL: 13.02 @ ppl-chunk 128: 

python ov_smoothquant/eval.py ./models/gpt-j-6b-SQ/ -ppl wikitext-2-raw/wiki.test.raw -c 128
PPL: 13.64 @ ppl-chunk 128  0.85
```
