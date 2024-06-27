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
