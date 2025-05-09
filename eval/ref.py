import os

os.environ["HUMANEVAL_OVERRIDE_PATH"] = "eval/data/HumanEvalPlus-v0.1.10.jsonl"
os.environ["MBPP_OVERRIDE_PATH"] = "eval/data/MbppPlus-v0.2.0.jsonl"

from evalplus import codegen, evaluate, sanitize
from evalplus.data import get_human_eval_plus, get_mbpp_plus


humaneval = get_human_eval_plus()
mbpp = get_mbpp_plus()

print("checking")
