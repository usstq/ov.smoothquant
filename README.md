# ov.smoothquant

[SmoothQuant](https://github.com/mit-han-lab/smoothquant) is a full INT8(W8A8) LLM quantization methods which can potentially use the full power of INT8 acceleration capability of HW (for example, AMX-INT8 on SPR case).

But it's not easy to keep accuracy of the quantized model, Intel® Neural Compressor has to implement [a special enhancement](https://github.com/intel/neural-compressor/blob/master/docs/source/smooth_quant.md#our-enhancement) for the accuracy-drop of quantized model to be within 1%.

OpenVINO backend for HF pipeline is [optimum-intel](https://github.com/huggingface/optimum-intel/), and the model exported by `optimum-cli` only support weight-only quantization of int8(or int4).


## Llama-2-7b
 - Using alpha=0.85
 - weight is per-OC quantized
 - must use per-token quantization for activation (at least for mlp.down_proj)
 - very few channel has very large absmax (>100), and must be calculated separately

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
python ./ov_smoothquant/quant.py -m /home/sdp/huyuan/dlboost_models/gpt-j-6b/pytorch/FP32/ -s ./act_scales/gpt-j-6b.pickle -o ./models/gpt-j-6b-SQ -a 0.5


# evaluate PPL
python ov_smoothquant/eval.py /home/sdp/huyuan/dlboost_models/gpt-j-6b/pytorch/FP32/ -ppl wikitext-2-raw/wiki.test.raw -c 128
PPL: 13.02 @ ppl-chunk 128: 

python ov_smoothquant/eval.py ./models/gpt-j-6b-SQ/ -ppl wikitext-2-raw/wiki.test.raw -c 128
PPL: 13.64 @ ppl-chunk 128  0.85

```