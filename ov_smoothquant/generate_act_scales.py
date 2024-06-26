from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

from openvino.runtime import opset8 as opset
from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
import numpy as np
import openvino as ov
from openvino.runtime.passes import Manager, Matcher, MatcherPass, WrapType, AnyInput
from openvino.runtime.utils import replace_node
import tqdm
import pickle, sys, time

def get_min_max_model(model):
    new_results = []
    new_result_names = []
    def pattern_replacement():
        act = AnyInput()
        wei = AnyInput()
        matmul = WrapType("opset8.MatMul", [act.output(0), wei.output(0)])

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            pvm = matcher.get_pattern_value_map()

            if not ("_proj/aten::linear/MatMul" in root.get_friendly_name()):
                return False

            #print(root, pvm[act].get_node())
            axes =  [0, 1]
            the_min = opset.reduce_min(pvm[act], reduction_axes = axes)
            the_max = opset.reduce_max(pvm[act], reduction_axes = axes)
            
            the_min.get_output_tensor(0).set_names({root.get_friendly_name() + "_min"})
            the_max.get_output_tensor(0).set_names({root.get_friendly_name() + "_max"})

            new_results.append(opset.result(the_min))
            new_results.append(opset.result(the_max))
            new_result_names.append(root.get_friendly_name())
            return False

        return Matcher(matmul, "SimpleReplacement"), callback
    
    manager = Manager()
    manager.register_pass(MatcherPass(*pattern_replacement()))
    manager.run_passes(model)

    model.add_results(new_results)
    return model, new_result_names

def get_fc_observations(model_path, calibration_txtfile, chunk_size = 128, total_size_limit = 0, device="CPU"):
    cfg=AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.add_special_tokens({'pad_token': '[PAD]'})
        #tok.pad_token = tok.eos_token_id

    ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": "", "AFFINITY":"CORE"}
    ov_config["INFERENCE_PRECISION_HINT"] = "f32"
    ov_model = OVModelForCausalLM.from_pretrained(
        model_path,
        device=device,
        ov_config=ov_config,
        config=cfg,
        trust_remote_code=True,
    )

    print("replace ov_model...")

    ov_model.model, fc_names = get_min_max_model(ov_model.model)

    print("recompile...")
    ov_model.request = None
    ov_model.compile()

    with open(calibration_txtfile) as f:
        text = f.read()

    print(f"tokenizing ...")
    inputs = tok(text, return_tensors="pt", return_token_type_ids=False)
    input_ids = inputs['input_ids']

    fc_observations = {}
    for fc_name in fc_names:
        fc_observations[fc_name + "_min"] = None
        fc_observations[fc_name + "_max"] = None

    if total_size_limit == 0:
        total_size_limit = input_ids.shape[1]
    else:
        total_size_limit = max(total_size_limit, input_ids.shape[1])
    progress_bar = tqdm.tqdm(range(0, total_size_limit, chunk_size))

    for i0 in progress_bar:
        input_ids_chunks = input_ids[:, i0:(i0+chunk_size)]
        input_ids_chunks[:, 0] = 1 # BOS

        result = ov_model.forward(input_ids_chunks, labels = input_ids_chunks, past_key_values=None, return_dict=True)
        ov_model.request.reset_state()

        update_cnt = 0
        for fc_name in fc_names:
            tag_min = fc_name + "_min"
            the_min = ov_model.request.get_tensor(tag_min).data
            if fc_observations[tag_min] is None:
                fc_observations[tag_min] = the_min
            else:
                new_min = np.minimum(fc_observations[tag_min], the_min)
                update_cnt += np.sum(new_min < fc_observations[tag_min])
                fc_observations[tag_min] = new_min

            tag_max = fc_name + "_max"
            the_max = ov_model.request.get_tensor(tag_max).data
            if fc_observations[tag_max] is None:
                fc_observations[tag_max] = the_max
            else:
                new_max = np.maximum(fc_observations[tag_max], the_max)
                update_cnt += np.sum(new_max > fc_observations[tag_max])
                fc_observations[tag_max] = new_max

        progress_bar.set_description(f"{update_cnt} ")
    
    fc_act_ic_absmax = {}
    for fc_name in fc_names:
        the_min = fc_observations[fc_name + "_min"]
        the_max = fc_observations[fc_name + "_max"]
        the_absmax = np.maximum(np.abs(the_min), np.abs(the_max))
        fc_act_ic_absmax[fc_name] = the_absmax

        mean_absmax = np.mean(the_absmax)
        if mean_absmax < 0.5:
            mean_absmax = 0.5
        outlier_idx = (the_max_abs > 20 * mean_absmax)

        if (outlier_idx.sum() > 0):
            print(fc_name, mean_absmax)
            print("  ", the_min[outlier_idx])
            print("  ", the_max[outlier_idx])
            print("  ", the_max_abs[outlier_idx])


    return fc_act_ic_absmax


import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--calibration_txtfile", type=str, required=True, default="wikitext-2-raw/wiki.test.raw", help="calibration text file")
parser.add_argument("-m", "--model_path", type=str, required=True, help="raw openvino IR (OVModelForCausalLM) export by optimum-cli")
parser.add_argument("-s", "--chunk_size", type=int, default=128, help="chunk size for calibration text")
parser.add_argument("-t", "--total_size_limit", type=int, default=0, help="total size limit for calibration text")

parser.add_argument("act_scales_path", type=str, help="target pickle file for storing calibration result",
                    default="act_scales/llama-2-7b.pickle")

args = parser.parse_args()

fc_act_ic_absmax = get_fc_observations(
    model_path = args.model_path,
    calibration_txtfile = args.calibration_txtfile,
    total_size_limit = args.total_size_limit,
    chunk_size = args.chunk_size
)

with open(args.act_scales_path, 'wb') as handle:
    pickle.dump(fc_act_ic_absmax, handle)
