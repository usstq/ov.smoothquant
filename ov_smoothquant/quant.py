from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

from openvino.runtime import opset8 as opset
from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
import numpy as np
import openvino as ov
from openvino.runtime.passes import Manager, Matcher, MatcherPass, WrapType, AnyInput
from openvino.runtime.utils import replace_node
import tqdm
import pickle, sys, time, argparse


def to_smooth_quant_model(model, fc_observations, alpha = 0.8):
    pass


parser = argparse.ArgumentParser()
parser.add_argument("-a", "--alpha", type=float, default=0.5)
parser.add_argument("-m", "--model_path", type=str, required=True, help="raw openvino IR (OVModelForCausalLM) export by optimum-cli")
parser.add_argument("-s", "--act_scales_path", type=str, required=True, help="target pickle file storing calibration result",
                    default="act_scales/llama-2-7b.pickle")
parser.add_argument("-p", "--prompt", type=str, default="What's oxygen?")

parser.add_argument("-o", "--output_model_path", type=str, help="target model path", default=None)

args = parser.parse_args()

handle = open(args.act_scales_path, 'rb')
with handle:
    fc_act_ic_absmax = pickle.load(handle)

print("=== show outlier channel's max abs activations observed ===")
for fc_name in fc_act_ic_absmax:
    the_absmax = fc_act_ic_absmax[fc_name]

    mean_absmax = np.mean(the_absmax)
    if mean_absmax < 0.5:
        mean_absmax = 0.5
    outlier_idx = (the_absmax > 20 * mean_absmax)

    if (outlier_idx.sum() > 0):
        print(fc_name, mean_absmax, the_absmax[outlier_idx])

device = "CPU"
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": "", "AFFINITY":"CORE"}
ov_config["INFERENCE_PRECISION_HINT"] = "f32"

cfg=AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
if tok.pad_token is None:
    tok.add_special_tokens({'pad_token': '[PAD]'})
    #tok.pad_token = tok.eos_token_id
ov_model = OVModelForCausalLM.from_pretrained(
    args.model_path,
    device=device,
    ov_config=ov_config,
    config=cfg,
    trust_remote_code=True,
)

def test_one_shot(ov_model, new_tokens = 32):
    prompt_test = args.prompt
    result = []
    inputs = tok(prompt_test, return_tensors="pt", padding=True)
    answers = ov_model.generate(**inputs, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False, temperature=None, top_p=None)
    for answer in answers:
        out = tok.decode(answer, skip_special_tokens=True)
        result.append(out.encode("utf-8")[:512])
    return result

print("=== quantize fc layers using smooth-quant ===")
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": "", "AFFINITY":"CORE"}
ov_config["INFERENCE_PRECISION_HINT"] = "f32"
ov_model = OVModelForCausalLM.from_pretrained(
    args.model_path,
    device=device,
    ov_config=ov_config,
    config=cfg,
    trust_remote_code=True,
)
ref_answer = test_one_shot(ov_model)

to_smooth_quant_model(ov_model.model, fc_act_ic_absmax)

print("recompile...")
ov_model.request = None
ov_model.compile()

cur_answer = test_one_shot(ov_model)

print(f"========ref_answer: {ref_answer}")
print(f"========cur_answer: {cur_answer}")

if args.output_model_path is not None:
    print(f"saving to {args.output_model_path} ...")
    ov_model.save_pretrained(args.output_model_path)
    tok.save_pretrained(args.output_model_path)
