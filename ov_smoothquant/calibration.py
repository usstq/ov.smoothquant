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
from datasets import load_dataset

def to_min_max_model(model):
    new_results = []
    fc_names = []
    def pattern_replacement():
        act = AnyInput()
        wei = AnyInput()
        matmul = WrapType("opset8.MatMul", [act.output(0), wei.output(0)])

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            pvm = matcher.get_pattern_value_map()

            const_weight = pvm[wei].get_node()
            if const_weight.get_type_name() == "Convert":
                const_weight = const_weight.input_value(0).get_node()

            if const_weight.get_type_name() != "Constant":
                print("\t skipped: ", root.get_friendly_name())
                return False

            #print(root, pvm[act].get_node())
            act_rank = len(pvm[act].get_partial_shape())
            axes =  [i for i in range(act_rank-1)]

            the_min = opset.reduce_min(pvm[act], reduction_axes = axes)
            the_max = opset.reduce_max(pvm[act], reduction_axes = axes)
            
            the_min.get_output_tensor(0).set_names({root.get_friendly_name() + "_min"})
            the_max.get_output_tensor(0).set_names({root.get_friendly_name() + "_max"})

            new_results.append(opset.result(the_min))
            new_results.append(opset.result(the_max))
            fc_names.append(root.get_friendly_name())
            return False

        return Matcher(matmul, "SimpleReplacement"), callback
    
    manager = Manager()
    manager.register_pass(MatcherPass(*pattern_replacement()))
    manager.run_passes(model)

    model.add_results(new_results)
    return fc_names

def get_fc_observations(model_path, dataset_path, seq_len = 512, num_samples = 512, device="CPU"):
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
        use_cache=False
    )

    print("replace ov_model...")
    fc_names = to_min_max_model(ov_model.model)

    print("recompile...")
    ov_model.request = None
    ov_model.compile()

    fc_observations = {}

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)

    for i in tqdm.tqdm(range(num_samples)):
        input_ids = tok(dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True).input_ids
        result = ov_model.forward(input_ids, labels = input_ids, past_key_values=None, return_dict=True)
        for fc_name in fc_names:
            the_min = ov_model.request.get_tensor(fc_name + "_min").data
            the_max = ov_model.request.get_tensor(fc_name + "_max").data
            if not (fc_name in fc_observations):
                fc_observations[fc_name] = {"min": the_min, "max": the_max}
            else:
                fc_observations[fc_name]["min"] = np.minimum(the_min, fc_observations[fc_name]["min"])
                fc_observations[fc_name]["max"] = np.maximum(the_max, fc_observations[fc_name]["max"])
        ov_model.request.reset_state()

    return fc_observations

import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset-path", type=str, default="./pile-rnd512.val.jsonl.zst", help="location of the calibration dataset, we use the validation set of the Pile dataset")
parser.add_argument("-m", "--model_path", type=str, required=True, help="raw openvino IR (OVModelForCausalLM) export by optimum-cli")
parser.add_argument("act_minmax_path", type=str, help="target pickle file for storing calibration result",
                    default="act_scales/llama-2-7b.pickle")

args = parser.parse_args()

print(f"calibrating {args.model_path} on {args.dataset_path} ...")
fc_observations = get_fc_observations(
    model_path = args.model_path,
    dataset_path = args.dataset_path,
)

print(f"saving fc_observations to {args.act_minmax_path}...")
with open(args.act_minmax_path, 'wb') as handle:
    pickle.dump(fc_observations, handle)

print("Done.")