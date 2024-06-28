# ov.smoothquant

[SmoothQuant](https://github.com/mit-han-lab/smoothquant) is a full INT8(W8A8) LLM quantization methods which can potentially use the full power of INT8 acceleration capability of HW (for example, AMX-INT8 on SPR case).

But it's not easy to keep accuracy of the quantized model, Intel® Neural Compressor has to implement [a special enhancement](https://github.com/intel/neural-compressor/blob/master/docs/source/smooth_quant.md#our-enhancement) for the accuracy-drop of quantized model to be within 1%.

OpenVINO backend for HF pipeline is [optimum-intel](https://github.com/huggingface/optimum-intel/), and the model exported by `optimum-cli` only support weight-only quantization of int8(or int4).


## 

```bash
# convert
optimum-cli export openvino --fp16 --task text-generation-with-past -m ./bigscience/bloomz-560m/ bloomz-560m-ov
# orginal PPL
python ./ov_smoothquant/ppl.py -m /home/tingqian/tingqian/models/bloomz-560m-ov --f32
PPL: 29.58 @ chunk 512/512: 100%|██████████████████| 557/557 [05:03<00:00,  1.83it/s]
python ./ov_smoothquant/ppl.py -m /home/tingqian/tingqian/models/bloomz-560m-ov --f32 -x16
PPL: 28.37 @ chunk 512/8192: 100%|█████████████████| 35/35 [00:19<00:00,  1.80it/s]

# calibration
python ./ov_smoothquant/calibration.py -m /home/tingqian/tingqian/models/bloomz-560m-ov act_scales/bloomz-560m.pickle

# quantize
python ./ov_smoothquant/quant.py -m /home/tingqian/tingqian/models/bloomz-560m-ov -s act_scales/bloomz-560m.pickle -o ./models/bloomz-560m-SQ -a 0.8 -othr 100000 -ppl ./wikitext-2-raw/wiki.test.raw

# validate
python ./ov_smoothquant/ppl.py -m=./models/ --f32 -x16
PPL: 30.82 @ chunk 512/8192: 100%|█████████████████| 35/35 [00:14<00:00,  2.43it/s]
python ./ov_smoothquant/ppl.py -m /home/tingqian/tingqian/models/bloomz-560m-ov --f32
PPL: 32.21 @ chunk 512/512: 100%|██████████████████| 557/557 [03:45<00:00,  2.47it/s]

```
## Llama-2-7b
To keep accuracy better, we need:
 - weight must be per-OC INT8-quantized (symmetrically)
 - Using relatively high alpha `alpha=0.85`
 - must use per-token quantization for activation (at least for mlp.down_proj)
   or skip some mlp.down_proj layers:
```bash
# __module.model.layers.1.mlp.down_proj/aten::linear/MatMul
# __module.model.layers.8.mlp.down_proj/aten::linear/MatMul
# __module.model.layers.10.self_attn.q_proj/aten::linear/MatMul & k & v
# __module.model.layers.26.mlp.down_proj/aten::linear/MatMul
# __module.model.layers.27.mlp.down_proj/aten::linear/MatMul
# __module.model.layers.28.mlp.down_proj/aten::linear/MatMul
# __module.model.layers.29.mlp.down_proj/aten::linear/MatMul
# __module.model.layers.30.mlp.down_proj/aten::linear/MatMul
# __module.model.layers.31.mlp.down_proj/aten::linear/MatMul
python ./ov_smoothquant/quant.py -m ~/tingqian/models/Llama-2-7b-hf-ov/ -s ./act_scales/Llama-2-7b-hf.pickle -o ./models/Llama-2-7b-hf-SQ -a 0.9 -skip .8.mlp.down_proj .31.mlp.down_proj .30.mlp.down_proj .1.mlp.down_proj .29.mlp.down_proj

```
 - very few channel has very large absmax (>100), and must be calculated separately using FP16/FP32/BF16 like [LLM.int8()](https://arxiv.org/abs/2208.07339)

## gpt-j-6b

## openai-community/gpt2-medium

export text generation model:
```bash
$ python3 ./tools/gpt2-textgen.py -e openai-community/gpt2-medium ./models/gpt2-medium-ov
model openai-community/gpt2-medium is exported to ./models/gpt2-medium-ov
```

raw accuracy
```bash
$ python ./ov_smoothquant/ppl.py -m=./models/gpt2-medium-ov/ -c 128
PPL: 28.72 @ chunk 128/128: 100%|█████████████| 2237/2237 [01:12<00:00, 30.97it/s]
```

calibration
```bash
python ./ov_smoothquant/calibration.py -m=./models/gpt2-medium-ov/ act_scales/gpt2.pickle
saving fc_observations to act_scales/gpt2.pickle...
```

quantize
```bash
python ./ov_smoothquant/quant.py -m=./models/gpt2-medium-ov/ -s ./act_scales/gpt2.pickle -o ./models/gpt2-med-SQ -a 0.6 -othr 100 -ppl ./wikitext-2-raw/wiki.test.raw
```

new accuracy
```bash
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


/home/spr_models/gpt-j-6b/pytorch/INT8_compressed_weights
2024-06-26 07:39:35,788 - root - INFO - |     Task     |Version|Metric|Value |   |Stderr|
2024-06-26 07:39:35,788 - root - INFO - |--------------|------:|------|-----:|---|-----:|
2024-06-26 07:39:35,788 - root - INFO - |lambada_openai|      0|ppl   |4.1192|±  |0.0889|
2024-06-26 07:39:35,788 - root - INFO - |              |       |acc   |0.6765|±  |0.0065|

/models/gpt-j-6b-SQ/
2024-06-26 08:06:30,120 - root - INFO - |     Task     |Version|Metric|Value |   |Stderr|
2024-06-26 08:06:30,120 - root - INFO - |--------------|------:|------|-----:|---|-----:|
2024-06-26 08:06:30,120 - root - INFO - |lambada_openai|      0|ppl   |4.0520|±  |0.0887|
2024-06-26 08:06:30,120 - root - INFO - |              |       |acc   |0.6852|±  |0.0065|
