#!/usr/bin/env python3
import argparse
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("--destination", default="artifacts/base")
args = parser.parse_args()
snapshot_download(
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    revision="f15c379fb32bb402fa06a7ae9aecb1febf4b79ec",
    local_dir=args.destination, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.model"],
)
