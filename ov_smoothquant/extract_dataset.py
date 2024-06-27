from datasets import load_dataset

# extract from dataset : https://gitee.com/hf-datasets/pile-val-backup
num_samples = 512

dataset = load_dataset("json", data_files="./val.jsonl.zst", split="train")
calibration_dataset = dataset.shuffle(seed=42).select(range(num_samples))
calibration_dataset.to_json("pile-rnd512.val.jsonl.zst")

dataset = load_dataset("json", data_files="pile-rnd512.val.jsonl.zst")
print(dataset)
