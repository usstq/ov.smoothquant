import numpy as np
import tqdm

class PPL:
    def __init__(self):
        self.nll = 0
        self.cnt = 0
    
    def __call__(self, all_logits, labels):
        '''
            all_logits [seq_length, vocab_size]
            labels     [seq_length]
        '''
        seq_length = all_logits.shape[0]
        for i in range(0, seq_length - 1):
            logits = all_logits[i, :]
            max_logit = np.amax(logits)
            sum_exp = np.sum(np.exp(logits - max_logit))

            # logits at time-step i is for predicting token at time-step (i+1)
            next_tok = labels[i + 1]
            log_softmax_of_tok = (logits[next_tok] - max_logit) - np.log(sum_exp)

            self.nll += -log_softmax_of_tok
            self.cnt += 1
        return np.exp(self.nll / self.cnt)

    def __str__(self):
        return f"PPL: {np.exp(self.nll / self.cnt):.2f}"

def perplexity_ov(tokenizer, ov_model, test_txt_file_path, chunk_size, step_size = 0):
    with open(test_txt_file_path) as f:
        text = f.read()
    
    step_size = max(step_size, chunk_size)

    print(f"tokenizing ...")
    inputs = tokenizer(text, return_tensors="pt", return_token_type_ids=False)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    ppl_evaluator = PPL()

    progress_bar = tqdm.tqdm(range(0, input_ids.shape[1], step_size))
    for i0 in progress_bar:
        input_ids_chunks = input_ids[:, i0:(i0+chunk_size)]
        input_ids_chunks[:, 0] = 1 # BOS

        result = ov_model.forward(input_ids_chunks, labels = input_ids_chunks, past_key_values=None, return_dict=True)
        ov_model.request.reset_state()
        # print(result.logits.shape)
        seq_len = result.logits.shape[1]
        ppl_evaluator(result.logits.numpy()[0, seq_len//2:, :], input_ids_chunks.numpy()[0, seq_len//2:])

        progress_bar.set_description(f"{ppl_evaluator} @ chunk {chunk_size}/{step_size}")

if __name__ == "__main__":
    import argparse
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    from optimum.intel.openvino import OVModelForCausalLM


    parser = argparse.ArgumentParser()
    parser.add_argument('--bf16', action="store_true")
    parser.add_argument('--f32', action="store_true")
    parser.add_argument('-d', '--dynamic_quantization_group_size', type=int, default = 0)
    parser.add_argument('-v', '--verbose', action="store_true")
    parser.add_argument("-ppl", type=str, default="./wikitext-2-raw/wiki.test.raw")
    parser.add_argument("-c", "--ppl-chunk", type=int, default=512)
    parser.add_argument("-m", '--model_path', type=str, required=True)
    parser.add_argument("-uc", '--use_cache', type=int, default=1)
    parser.add_argument("-x", '--speed_up', type=int, default=1)

    args = parser.parse_args()

    model_path = args.model_path
    device = "CPU"
    ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": "", "AFFINITY":"CORE"}
    if args.bf16:
        ov_config["INFERENCE_PRECISION_HINT"] = "bf16"

    if args.f32:
        ov_config["INFERENCE_PRECISION_HINT"] = "f32"

    if args.dynamic_quantization_group_size > 0:
        ov_config["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = str(args.dynamic_quantization_group_size)

    print(ov_config)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.add_special_tokens({'pad_token': '[PAD]'})
        #tok.pad_token = tok.eos_token_id
    cfg=AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    ov_model = OVModelForCausalLM.from_pretrained(
        args.model_path,
        device=device,
        ov_config=ov_config,
        config=cfg,
        trust_remote_code=True,
        use_cache=args.use_cache
    )

    perplexity_ov(tok, ov_model, args.ppl, chunk_size = args.ppl_chunk, step_size = args.ppl_chunk * args.speed_up)
