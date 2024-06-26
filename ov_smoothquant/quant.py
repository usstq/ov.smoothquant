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

def to_smooth_quant_model(model, fc_act_ic_absmax, alpha = 0.8):
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

            if not root.get_friendly_name() in fc_act_ic_absmax:
                return False
            
            #if not "__module.model.layers.1.mlp.down_proj/aten::linear/MatMul" in root.get_friendly_name():
            #    return False

            X_max_abs = fc_act_ic_absmax[root.get_friendly_name()]

            X_max_abs[X_max_abs > 200] = 200

            # weight matrix: [OC, IC]
            const_weight = pvm[wei].get_node()
            if const_weight.get_type_name() == "Convert":
                const_weight = const_weight.input_value(0).get_node()
            assert(const_weight.get_type_name() == "Constant")

            weight = const_weight.data.astype(np.float32)

            IC = pvm[act].get_partial_shape().get_dimension(2).get_length()
            assert(IC == weight.shape[1])

            OC = weight.shape[0]

            # smooth-quant:
            weight_abs_max = abs(weight).max(0).clip(min=1e-5)
            assert(X_max_abs.min() > 0)

            # s : each IC channel of weight matrix * s
            # use big alpha, for bigger outlier |X|, so x_scale can scale it down further

            per_channel_alpha = (X_max_abs * 0 + 0.85)
            #per_channel_alpha[X_max_abs > 10] = 0.7
            #per_channel_alpha[X_max_abs > 100] = 0.8

            layer_id = int(root.get_friendly_name().split(".")[3])
            quant_activation = True

            '''
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

            per_token_quant = False
            if "mlp.down_proj" in root.get_friendly_name():
                quant_activation = False
                #per_token_quant = True
                pass

            px = pow(X_max_abs, per_channel_alpha)
            pw = pow(weight_abs_max, 1 - per_channel_alpha)
            # pw = px
            smoothquant_w_scales = (px / pw).clip(min=1e-5)
            smoothquant_x_scales = 1/smoothquant_w_scales

            # clear outlier channel from quantization branch
            outlier_idx = (X_max_abs > 100)
            outlier_cnt = outlier_idx.sum()
            if quant_activation and outlier_cnt > 0 :
                smoothquant_x_scales[outlier_idx] = 0
                smoothquant_w_scales[outlier_idx] = 0
                outlier_gather = opset.gather(pvm[act], outlier_idx.nonzero()[0], np.array([-1], dtype=np.int32))
                outlier_result = opset.matmul(outlier_gather, op.Constant(weight[:, outlier_idx]), transpose_a, transpose_b, name = root.get_friendly_name()+"_outlier") 
            else:
                outlier_result = None

            # [OC, IC] * [IC]
            weight = weight * smoothquant_w_scales

            act_smoothed = opset.multiply(pvm[act], op.Constant(smoothquant_x_scales))

            xmax = (X_max_abs * smoothquant_x_scales).max()
            xmin = (X_max_abs * smoothquant_x_scales).min()

            # quantize weight to INT8 on per-OC basis
            # per-tensor quantized is not working
            # w_deq_scales = np.array(abs(weight).max() / 127, dtype=np.float32)
            # per-OC quantized works well
            w_deq_scales = abs(weight).max(1, keepdims=True) / 127
            weight_quant = (weight / w_deq_scales).round().astype(np.int8)
            w_deq = opset.multiply(opset.convert(op.Constant(weight_quant), Type.f32), op.Constant(w_deq_scales))

            if quant_activation:
                levels = np.int32(256)

                input_low = np.array(-xmax, dtype=np.float32)
                input_high = np.array(xmax, dtype=np.float32)
                output_low = np.array(-xmax, dtype=np.float32)
                output_high = np.array(xmax, dtype=np.float32)

                # per-token
                if per_token_quant:
                    absmax_per_token = opset.reduce_max(opset.absolute(act_smoothed), [0, 1], keep_dims = True)
                    input_low = opset.negative(absmax_per_token)
                    output_low = opset.negative(absmax_per_token)
                    input_high = absmax_per_token
                    output_high = absmax_per_token

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

            # new_matmul = opset.matmul(act_smoothed, op.Constant(weight), transpose_a, transpose_b, name = root.get_friendly_name())
            replace_node(root, new_matmul)

            print(f"{'Q' if quant_activation else ' '} Outlier:{outlier_cnt} {layer_id} {root.get_friendly_name()} {smoothquant_x_scales.shape} {smoothquant_w_scales.shape}  w_scales : {w_deq_scales.shape}  {smoothquant_x_scales.min():.3f}~{smoothquant_x_scales.max():.3f}  {X_max_abs.min():.3f}~{X_max_abs.max():.3f} =>  {xmin:.3f}~{xmax:.3f}")
            #print(f"\t mean:{X_max_abs.mean()}  big:{X_max_abs[X_max_abs > 10]}")
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
