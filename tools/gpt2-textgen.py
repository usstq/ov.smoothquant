from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import numpy as np
from openvino.runtime import Core, get_version, serialize
import openvino as ov
from optimum.bettertransformer import BetterTransformer
import argparse, sys, time, os
from optimum.intel import OVModelForCausalLM, OVConfig

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--new-tokens', type=int, default=32)
parser.add_argument('-b', '--batch-sizes', type=int, nargs='*')
parser.add_argument('-v', '--verbose', action="store_true")
parser.add_argument('-p', '--prompt', type=str, default="What's oxygen?")
parser.add_argument('-e', '--export', type=str, default=None)
parser.add_argument('model_path')

args = parser.parse_args()

if args.export:
    # python3 ./test.py -e openai-community/gpt2-medium ./ov-model/
    config = AutoConfig.from_pretrained(args.export, num_labels=2, max_position_embeddings=1024)
    config.pad_token_id = config.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(args.export)
    print(model)

    # Cannot handle batch sizes > 1 if no padding token is defined.
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False
    new_model = BetterTransformer.transform(model)

    os.makedirs(args.model_path, exist_ok=True)

    # convert the model
    ov_model = ov.convert_model(new_model, example_input={"input_ids": np.ones([2, 1024], dtype=np.int64), "attention_mask": np.ones([2, 1024], dtype=np.int64)})

    tok = AutoTokenizer.from_pretrained(args.export, trust_remote_code=True)
    if tok.pad_token is None:
        tok.add_special_tokens({'pad_token': '[PAD]'})
        #tok.pad_token = tok.eos_token_id

    # save model & config
    tok.save_pretrained(args.model_path)
    ov.save_model(ov_model, args.model_path + "/openvino_model.xml", True)
    config.save_pretrained(args.model_path)
    print(f"model {args.export} is exported to {args.model_path}")
    sys.exit(0)

prompt = args.prompt
if prompt.isdigit():
    prompt = "Hi"*(int(prompt) - 1)

model = OVModelForCausalLM.from_pretrained(args.model_path, use_cache=False)
tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
# print(dir(model))

rank_info = f"{ov.__version__}"
print(rank_info, "prompt:", prompt[:32], "...")

class Detokenizer:
    def __init__(self, tok):
        self.tok = tok
        self.output_str = ""
        pass

    def show_last(self, last_out, last_repeats):
        if last_out:
            if last_out.startswith(prompt):
                last_out = "... " + last_out[len(prompt):]
            self.output_str += "\t Output=" + repr(last_out.encode("utf-8")[:200]) + f"... repeats {last_repeats} times;"

    def __call__(self, answers):
        self.output_str = ""
        last_out = None
        last_repeats = 0

        for answer in answers:
            out = tok.decode(answer, skip_special_tokens=True)
            # changes will be printed
            if (not last_out) or (last_out != out):
                self.show_last(last_out, last_repeats)
                last_repeats = 1
                last_out = out
            else:
                last_repeats += 1

        self.show_last(last_out, last_repeats)
        return self.output_str

detok = Detokenizer(tok)

for batch_size in args.batch_sizes:
    prompt_batch = [prompt] * batch_size
    inputs = tok(prompt_batch, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]

    t0 = time.time()
    answers = model.generate(**inputs, max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens, do_sample=False)
    t1 = time.time()

    print(f"batch_size={batch_size} {(t1-t0)*1e3 : .2f} ms  {batch_size/(t1-t0):.2f} QPS {detok(answers)} ")