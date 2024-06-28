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
from ppl import perplexity_ov

def to_smooth_quant_model(model, fc_observations, skip_act_quant_names=[], skip_quant_names=[], act_quant_sym = False, outlier_thr = 1e9, alpha = 0.8):
    # Simple: for Extensions. Without any classes and inheritance.
    def pattern_replacement():
        act = AnyInput()
        wei = AnyInput()
        matmul = WrapType("opset8.MatMul", [act.output(0), wei.output(0)])

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            pvm = matcher.get_pattern_value_map()
            transpose_a = False
            transpose_b = True

            if not root.get_friendly_name() in fc_observations:
                return False

            #if not "__module.model.layers.1.mlp.down_proj/aten::linear/MatMul" in root.get_friendly_name():
            #    return False
            # __module.model.transformer.h.2.mlp.fc_out/aten::linear/MatMul

            X_min = fc_observations[root.get_friendly_name()]["min"]
            X_max = fc_observations[root.get_friendly_name()]["max"]
            X_absmax = fc_observations[root.get_friendly_name()]["absmax"]

            # weight matrix: [OC, IC]
            const_weight = pvm[wei].get_node()
            if const_weight.get_type_name() == "Convert":
                const_weight = const_weight.input_value(0).get_node()
            assert(const_weight.get_type_name() == "Constant")

            weight = const_weight.data.astype(np.float32)

            act_rank = len(pvm[act].get_partial_shape())
            IC = pvm[act].get_partial_shape().get_dimension(act_rank-1).get_length()
            assert(IC == weight.shape[1])

            OC = weight.shape[0]

            # smooth-quant:
            weight_abs_max = abs(weight).max(0).clip(min=1e-5)
            X_absmax = X_absmax.clip(min=1e-5)
            # assert(X_absmax.min() > 0)

            # s : each IC channel of weight matrix * s
            # use big alpha, for bigger outlier |X|, so x_scale can scale it down further

            per_channel_alpha = (X_absmax * 0 + alpha)
            #per_channel_alpha[X_absmax > 10] = 0.7
            #per_channel_alpha[X_absmax > 100] = 0.8

            layer_id = 0 # int(root.get_friendly_name().split(".")[3])
            quant_activation = True
            if skip_act_quant_names is not None:
                for skip_name in skip_act_quant_names:
                    if skip_name in root.get_friendly_name():
                        quant_activation = False

            if skip_quant_names is not None:
                for skip_name in skip_quant_names:
                    if skip_name in root.get_friendly_name():
                        print(f"\t skipped {root.get_friendly_name()}")
                        return False

            '''
            if ".mlp.down_proj/aten::linear/MatMul" in root.get_friendly_name():
                layer_id = int(root.get_friendly_name().split(".")[3])
                # layer_id == 25 : PPL: 5.65
                # PPL: 5.67   0.85
                if layer_id >= 0:
                    per_channel_alpha = 0.85
                    quant_activation = True
                if layer_id == 1 or layer_id >= 25:
                    per_channel_alpha = 0.87
                    quant_activation = True

            if "__module.model.layers.1.mlp.down_proj/aten::linear/MatMul" in root.get_friendly_name():
                quant_activation = False
            #if "__module.model.layers.30.mlp.down_proj/aten::linear/MatMul" in root.get_friendly_name():
            #    quant_activation = False
            #if "__module.model.layers.31.mlp.down_proj/aten::linear/MatMul" in root.get_friendly_name():
                #quant_activation = False
            #    per_channel_alpha = 0.6
            if "__module.model.layers.8.mlp.down_proj/aten::linear/MatMul" in root.get_friendly_name():
                #quant_activation = False
                per_channel_alpha = 0.8
            '''

            px = pow(X_absmax, per_channel_alpha)
            pw = pow(weight_abs_max, 1 - per_channel_alpha)
            # pw = px
            smoothquant_w_scales = (px / pw).clip(min=1e-5)
            smoothquant_x_scales = 1/smoothquant_w_scales

            per_token_quant = False
            #if "mlp.down_proj" in root.get_friendly_name():
            #    #quant_activation = False
            #    per_token_quant = True
            #    pass

            # clear outlier channel from quantization branch
            outlier_idx = (X_absmax > outlier_thr)
            outlier_cnt = outlier_idx.sum()
            if quant_activation and outlier_cnt > 0 :
                smoothquant_x_scales[outlier_idx] = 0
                smoothquant_w_scales[outlier_idx] = 0
                outlier_gather = opset.gather(pvm[act], outlier_idx.nonzero()[0], np.array([-1], dtype=np.int32))
                outlier_result = opset.matmul(outlier_gather, op.Constant(weight[:, outlier_idx]), transpose_a, transpose_b, name = root.get_friendly_name()+"_outlier") 
            else:
                outlier_result = None

            # [OC, IC] * [IC]
            if quant_activation:
                weight = weight * smoothquant_w_scales
                act_smoothed = opset.multiply(pvm[act], op.Constant(smoothquant_x_scales))
                x_min_per_tensor = (X_min * smoothquant_x_scales).min()
                x_max_per_tensor = (X_max * smoothquant_x_scales).max()
            else:
                act_smoothed = pvm[act]
                x_min_per_tensor = X_min.min()
                x_max_per_tensor = X_max.max()

            # quantize weight to INT8 on per-OC basis
            # per-tensor quantized is not working
            # w_deq_scales = np.array(abs(weight).max() / 127, dtype=np.float32)
            # per-OC quantized works well
            w_deq_scales = abs(weight).max(1, keepdims=True) / 127
            weight_quant = (weight / w_deq_scales).round().astype(np.int8)
            w_deq = opset.multiply(opset.convert(op.Constant(weight_quant), Type.f32), op.Constant(w_deq_scales))

            # symmetrical quantization has lower accuracy than asymmetrical
            if act_quant_sym:
                absmax = max(abs(x_min_per_tensor), abs(x_max_per_tensor))
                x_min_per_tensor = -absmax
                x_max_per_tensor = absmax

            if quant_activation:
                levels = np.int32(256)

                if per_token_quant:
                    # per-token dynamic, need special impl
                    absmax_per_token = opset.reduce_max(opset.absolute(act_smoothed), [0, 1], keep_dims = True)
                    input_low = opset.negative(absmax_per_token)
                    output_low = opset.negative(absmax_per_token)
                    input_high = absmax_per_token
                    output_high = absmax_per_token
                else:
                    # per-tensor static (easier for impl to speed-up)
                    input_low = np.array(x_min_per_tensor, dtype=np.float32)
                    input_high = np.array(x_max_per_tensor, dtype=np.float32)
                    output_low = np.array(x_min_per_tensor, dtype=np.float32)
                    output_high = np.array(x_max_per_tensor, dtype=np.float32)

                act_fq = opset.fake_quantize(act_smoothed,
                                            input_low,
                                            input_high,
                                            output_low,
                                            output_high,
                                            levels)
                new_matmul = opset.matmul(act_fq, w_deq, transpose_a, transpose_b, name = root.get_friendly_name())
            else:
                new_matmul = opset.matmul(act_smoothed, w_deq, transpose_a, transpose_b, name = root.get_friendly_name())
            
            if outlier_result is not None:
                new_matmul = opset.add(new_matmul, outlier_result, name="AddOutlier")

            replace_node(root, new_matmul)

            X_maxabs_thr = max(10*X_absmax.mean(), 30)
            quant_flag = f"Q{'A' if quant_activation else '_'}W"

            print(f"{quant_flag} Outlier:{outlier_cnt} {layer_id} {root.get_friendly_name()} [x:{smoothquant_x_scales.min():.2f}~{smoothquant_x_scales.max():.2f} w:{smoothquant_w_scales.min():.2f}~{smoothquant_w_scales.max():.2f}]  {X_min.min():.2f}~{X_max.max():.2f} =>  {x_min_per_tensor:.2f}~{x_max_per_tensor:.2f} mean:{X_absmax.mean():.3f}  big:{X_absmax[X_absmax > X_maxabs_thr]}")
            return True
        return Matcher(matmul, "SimpleReplacement"), callback
    manager = Manager()
    manager.register_pass(MatcherPass(*pattern_replacement()))
    manager.run_passes(model)
    return

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--alpha", type=float, default=0.5)
parser.add_argument("-m", "--model_path", type=str, required=True, help="raw openvino IR (OVModelForCausalLM) export by optimum-cli")
parser.add_argument("-s", "--act_scales_path", type=str, required=True, help="target pickle file storing calibration result",
                    default="act_scales/llama-2-7b.pickle")
parser.add_argument("-p", "--prompt", type=str, default="What's oxygen?")

parser.add_argument("-skip", "--skip", type=str, nargs='*')
parser.add_argument("-skip_act", "--skip_act", type=str, nargs='*')

parser.add_argument("-othr", "--outlier_thr", type=float, default=1e9)
parser.add_argument("-o", "--output_model_path", type=str, help="target model path", default=None)
parser.add_argument("-uc", '--use_cache', type=int, default=1)
parser.add_argument("-ppl", type=str, default=None)

args = parser.parse_args()

handle = open(args.act_scales_path, 'rb')
with handle:
    fc_observations = pickle.load(handle)

print("=== absmax of activations observed ===")
for fc_name in fc_observations:
    the_min = fc_observations[fc_name]["min"]
    the_max = fc_observations[fc_name]["max"]

    the_absmax = np.maximum(np.abs(the_min), np.abs(the_max))
    fc_observations[fc_name]["absmax"] = the_absmax

    mean_absmax = max(0.5, np.mean(the_absmax))
    outlier_idx = (the_absmax > 20 * mean_absmax)
    if (outlier_idx.sum() > 0):
        print(fc_name, mean_absmax, the_absmax[outlier_idx])

device = "CPU"
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": "", "AFFINITY":"CORE"}
ov_config["INFERENCE_PRECISION_HINT"] = "f32"

cfg=AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
try:
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
except:
    print("No tokenizer")
    tok = None

if tok and tok.pad_token is None:
    tok.add_special_tokens({'pad_token': '[PAD]'})
    #tok.pad_token = tok.eos_token_id

print("=== quantize fc layers using smooth-quant ===")
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": "", "AFFINITY":"CORE"}
ov_config["INFERENCE_PRECISION_HINT"] = "f32"
ov_model = OVModelForCausalLM.from_pretrained(
    args.model_path,
    device=device,
    ov_config=ov_config,
    config=cfg,
    trust_remote_code=True,
    use_cache=bool(args.use_cache)
)

def test_one_shot(ov_model, new_tokens = 32):
    if tok is None:
        return "???"
    prompt_test = args.prompt
    result = []
    inputs = tok(prompt_test, return_tensors="pt", padding=True)
    answers = ov_model.generate(**inputs, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False, temperature=None, top_p=None)
    for answer in answers:
        out = tok.decode(answer, skip_special_tokens=True)
        result.append(out.encode("utf-8")[:512])
    return result

ref_answer = test_one_shot(ov_model)

to_smooth_quant_model(ov_model.model, fc_observations,
    alpha = args.alpha, 
    skip_act_quant_names=args.skip_act,
    skip_quant_names=args.skip,
    outlier_thr=args.outlier_thr)

print("recompile...")
ov_model.request = None
ov_model.compile()

cur_answer = test_one_shot(ov_model)

print(f"========ref_answer: {ref_answer}")
print(f"========cur_answer: {cur_answer}")

if args.ppl:
    perplexity_ov(tok, ov_model, args.ppl, chunk_size = 512, step_size = 8192)

if args.output_model_path is not None:
    print(f"saving to {args.output_model_path} ...")
    ov_model.save_pretrained(args.output_model_path)
    if tok:
        tok.save_pretrained(args.output_model_path)
