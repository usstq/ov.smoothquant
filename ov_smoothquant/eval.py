import openvino
import openvino.runtime as ov

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
import time

import types
import inspect
import argparse

import sys, os
import psutil
import tqdm
import numpy as np

from ppl import perplexity_ov

process = psutil.Process()


'''
optimum-cli export openvino --fp16 --task text-generation-with-past -m /home/openvino-ci-58/tingqian/model/llama2-7bchat /home/openvino-ci-58/tingqian/model/llama2-7bchat/OV
  --weight-format {fp32,fp16,int8,int4_sym_g128,int4_asym_g128,int4_sym_g64,int4_asym_g64}

optimum-cli export openvino --task text-generation-with-past --weight-format int4_sym_g128 -m ./org-opt-125m/ ov-opt-125m


mpi4py has issue working with intel MPI, so we can only perform MPI API calls inside CPU plugin code.
https://github.com/mpi4py/mpi4py/issues/418

    export MPICC=`which mpicc`
    python -m pip install git+https://github.com/mpi4py/mpi4py

OV_CPU_PROFILE=1 mpirun   \
  -n 1 env RANK_INFO=0of2 numactl --all -C 0-7 -m 0 python ./testLLM.py /mnt/disk2/tingqian/models/Mistral-7B-Instruct-v0.1-OV/FP16 : \
  -n 1 env RANK_INFO=1of2 numactl --all -C 48-55 -m 1 python ./testLLM.py /mnt/disk2/tingqian/models/Mistral-7B-Instruct-v0.1-OV/FP16
'''


#rank_info = os.environ["RANK_INFO"]
rank_info = f"{openvino.__version__}"
print(rank_info)

'''
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()
rank_info = f"[rank: {rank}/{world_size}] "
isMaster = (rank == 0)

print(rank_info)
sys.exit(0)
'''


def hook_forward(model):
    model._org_forward = model.forward
    model._latencies = []
    def new_forward(self, *args, **kwargs):
        # Call the original method
        # print(args, kwargs)
        t0 = time.time()
        ret = self._org_forward(*args, **kwargs)
        t1 = time.time()
        self._latencies.append(t1 - t0)
        return ret
    # https://stackoverflow.com/questions/1409295/set-function-signature-in-python
    # package transformers uses inspect.signature to detect exact signature of
    # model's forward method and behave differently based on that, for example
    # only auto-generate attention-mask when signature contain such named parameter
    new_forward.__signature__ = inspect.signature(model.forward)
    model.forward = types.MethodType(new_forward, model)

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--new-tokens', type=int, default=32)
parser.add_argument('-b', '--batch-sizes', type=int, nargs='*')
parser.add_argument('--bf16', action="store_true")
parser.add_argument('--f32', action="store_true")
parser.add_argument('-p', '--prompt', type=str, default="What's oxygen?")
parser.add_argument('-d', '--dynamic_quantization_group_size', type=int, default = 0)
parser.add_argument('-v', '--verbose', action="store_true")
parser.add_argument('-s', '--smaps', action="store_true")
parser.add_argument('-r', '--repeats', type=int, default=1)
parser.add_argument('-e', '--export', action="store_true")

parser.add_argument('-kvq', '--kvquant', action="store_true")

parser.add_argument("-ppl", type=str, default=None)
parser.add_argument("-c", "--ppl-chunk", type=int, default=512)

parser.add_argument('model_path')

args = parser.parse_args()
prompt = args.prompt
if prompt.isdigit():
    prompt = "Hi"*(int(prompt) - 1)

if args.batch_sizes is None:
    args.batch_sizes = [1]

args.batch_sizes = args.batch_sizes * args.repeats

print(rank_info, " OV VERSION=", ov.get_version())

# model list
if args.verbose:
    for i, model_ir in enumerate(OV_IR):
        print(f"[{i}] : {model_ir}")

model_path = args.model_path
device = "CPU"


ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": "", "AFFINITY":"CORE"}
if args.bf16:
    ov_config["INFERENCE_PRECISION_HINT"] = "bf16"

if args.f32:
    ov_config["INFERENCE_PRECISION_HINT"] = "f32"
 
if args.kvquant:
    ov_config["KV_CACHE_PRECISION"] = "u8"

if args.dynamic_quantization_group_size > 0:
    ov_config["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = str(args.dynamic_quantization_group_size)

print(ov_config)
print(rank_info, "--> load tokenizer.")
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tok.pad_token is None:
    tok.add_special_tokens({'pad_token': '[PAD]'})
    #tok.pad_token = tok.eos_token_id
 
cfg=AutoConfig.from_pretrained(model_path, trust_remote_code=True)

if args.verbose:
    print(rank_info, f"--> config {cfg}.")

if args.export:
    print("exporting OV IR from HF model....")
    t1=time.time()
    ov_model = OVModelForCausalLM.from_pretrained(
        model_path,
        device=device,
        export=True,
        compile=False,
        load_in_8bit=False,
        trust_remote_code=True
    )
    ov.save_model(ov_model.model, model_path + "/openvino_model.xml")
    print(rank_info, f" Model export(convert) & compilation took {time.time()-t1:.2f} seconds.")
    sys.exit(0)
else:
    t1=time.time()
    ov_model = OVModelForCausalLM.from_pretrained(
        model_path,
        device=device,
        ov_config=ov_config,
        config=cfg,
        trust_remote_code=True,
    )
    print(rank_info, f" Model compilation took {time.time()-t1:.2f} seconds.")


if args.ppl is not None:
    perplexity_ov(tok, ov_model, args.ppl, chunk_size = args.ppl_chunk)
    sys.exit(0)

print(f"RSS {process.memory_info().rss*1e-9: .3f} GB")

hook_forward(ov_model)

print(rank_info, "prompt:", prompt[:32], "...")

print(f"============== {model_path} ==============")
for batch_size in args.batch_sizes:
    prompt_batch = [prompt] * batch_size
    inputs = tok(prompt_batch, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]

    print(rank_info, f" generating for prompt shape: {input_ids.shape[0]}x{input_ids.shape[1]}...", end="")

    # synchronize (can be skipped)
    # data = (rank+1)**2
    # print(rank_info, f"data={data}")
    # all_data = comm.allgather(data)
    # print(rank_info, f"all_data={all_data}")

    ov_model._latencies = []
    answers = ov_model.generate(**inputs, max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens, do_sample=False, temperature=None, top_p=None)

    print("\r", " " * 80, "\r", end="")
    
    output_str = ""
    last_out = None
    last_repeats = 0
    def show_last(last_out, last_repeats):
        global output_str
        if last_out:
            if last_out.startswith(prompt):
                last_out = "... " + last_out[len(prompt):]
            output_str += "\t Output=" + repr(last_out.encode("utf-8")[:512]) + f"... repeats {last_repeats} times;"

    for answer in answers:
        out = tok.decode(answer, skip_special_tokens=True)
        # changes will be printed
        if (not last_out) or (last_out != out):
            show_last(last_out, last_repeats)
            last_repeats = 1
            last_out = out
        else:
            last_repeats += 1

    show_last(last_out, last_repeats)

    l = ov_model._latencies
    second_tok_latency = sum(l[1:])/(len(l)-1) if len(l) > 1 else 0
    mem_info = process.memory_info()
    print(rank_info, f" prompt:{input_ids.shape[0]}x{input_ids.shape[1]}  {l[0]*1e3:6.1f} ms + {second_tok_latency*1e3:6.1f} ms x {len(l)-1}   outputs: {output_str}  RSS/VMS {mem_info.rss*1e-9: .3f}/{mem_info.vms*1e-9: .3f} GB")

print(f"ov_config={ov_config}")

if args.smaps:
    AnonHugePages_size = 0
    file = open("/proc/self/smaps")
    lines = file.readlines()
    for l in lines:
        ls = l.split()
        if ls[0].startswith("AnonHugePages") and int(ls[1]) > 0:
            assert(ls[2] == 'kB')
            AnonHugePages_size += int(ls[1])
            print(ls)

    with open("smaps.txt", "w") as text_file:
        for l in lines:
            text_file.write(l)
    
    print(f"smaps.txt is generated. totall AnonHugePages {AnonHugePages_size} kB")