import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--alpha", type=float, default=0.5)
parser.add_argument("-m", "--model_path", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("-s", "--act_scales_path", type=str, default="act_scales/llama-2-7b.pickle")

args = parser.parse_args()

