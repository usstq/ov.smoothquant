# ov.smoothquant

[SmoothQuant](https://github.com/mit-han-lab/smoothquant) is a full INT8(W8A8) LLM quantization methods which can potentially use the full power of INT8 acceleration capability of HW (for example, AMX-INT8 on SPR case).

But it's not easy to keep accuracy of the quantized model, Intel® Neural Compressor has to implement [a special enhancement](https://github.com/intel/neural-compressor/blob/master/docs/source/smooth_quant.md#our-enhancement) for the accuracy-drop of quantized model to be within 1%.

OpenVINO backend for HF pipeline is [optimum-intel](https://github.com/huggingface/optimum-intel/), and the model exported by `optimum-cli` only support weight-only quantization of int8(or int4).


##

```bash
# download wikitext-2
wget https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip

# export ov LLM model IR (fp16)
optimum-cli export openvino --fp16 --task text-generation-with-past -m ./meta-llama/Llama-2-7b-hf/ Llama-2-7b-hf-ov

# calibration: generate abs max for each activations of Linear/FC layer
python ./ov_smoothquant/generate_act_scales.py -c=wikitext-2-raw/wiki.test.raw -m ~/models/Llama-2-7b-hf-ov/ -s 128 ./act_scales/Llama-2-7b-hf.pickle


```