from transformers import AutoConfig, AutoModelForSequenceClassification
import numpy as np
from openvino.runtime import Core, get_version, serialize
import openvino as ov
from optimum.bettertransformer import BetterTransformer
import argparse, sys, time, os
from optimum.intel import OVModelForSequenceClassification, OVConfig

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-tokens', type=int, default=1024)
parser.add_argument('-b', '--batch-sizes', type=int, nargs='*')
parser.add_argument('-v', '--verbose', action="store_true")
parser.add_argument('-e', '--export', type=str, default=None)
parser.add_argument('-no-sdpa', action="store_true")
parser.add_argument('-reshape-batch', type=int, default=0)

parser.add_argument('model_path')

args = parser.parse_args()

if args.export:
    # python3 ./test.py -e openai-community/gpt2-medium ./ov-model/
    config = AutoConfig.from_pretrained(args.export, num_labels=2, max_position_embeddings=1024)
    config.pad_token_id = config.eos_token_id

    # create a fake model w/o pre-trained para from config
    model = AutoModelForSequenceClassification.from_pretrained(args.export)
    print(model)

    # Cannot handle batch sizes > 1 if no padding token is defined.
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False

    if args.no_sdpa:
        new_model = model
    else:
        new_model = BetterTransformer.transform(model)

    os.makedirs(args.model_path, exist_ok=True)

    # convert the model
    ov_model = ov.convert_model(new_model, example_input={"input_ids": np.ones([2, 1024], dtype=np.int64), "attention_mask": np.ones([2, 1024], dtype=np.int64)})

    if args.reshape_batch > 0:
        ov_model.reshape({
            "input_ids": [args.reshape_batch , 1024],
            "attention_mask" : [args.reshape_batch , 1024]
        })
    # save model & config
    ov.save_model(ov_model, args.model_path + "/openvino_model.xml", True)
    config.save_pretrained(args.model_path)
    print(f"model {args.export} is exported to {args.model_path}")
    sys.exit(0)
    '''
    # following standard method cannot generate ScaledDotProductAttention
    config = AutoConfig.from_pretrained(args.model_path, num_labels=2, max_position_embeddings=1024)
    config.pad_token_id = config.eos_token_id
    model = OVModelForSequenceClassification.from_pretrained(args.model_path, export=True)
    model.save_pretrained("./gpt2-ov")
    '''

model = OVModelForSequenceClassification.from_pretrained(args.model_path)
# print(dir(model))

for batch_size in args.batch_sizes:
    # fake token ids are generated, w/o using tokenizer, for testing purpose
    input_ids = np.ones([batch_size, args.input_tokens], dtype=np.int64)
    attention_mask = np.ones([batch_size, args.input_tokens], dtype=np.int64)

    t0 = time.time()
    result = model.forward(input_ids=input_ids, attention_mask=attention_mask)
    t1 = time.time()

    print(f"batch_size={batch_size} {(t1-t0)*1e3 : .2f} ms  {batch_size/(t1-t0):.2f} QPS ", str(result.logits.tolist())[:64] + "...")
