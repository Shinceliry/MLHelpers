import argparse
import importlib.util
import ast
import os
import yaml
import json
import torch
import torch.nn as nn
from torchinfo import summary
from torchviz import make_dot

class DictToObj:
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObj(value))
            else:
                setattr(self, key, value)

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize PyTorch Model Graph")
    parser.add_argument("--model-file", type=str, required=True, help="Path to the Python file that defines the model class.")
    parser.add_argument("--model-class", type=str, required=True, help="Name of the model class inside the specified Python file.")
    
    # Accepts model constructor arguments as a string in dict format (exsample: --model-init-args "{'config':'/path/to/config.yaml'}")
    parser.add_argument("--model-init-args", type=str, default="{}", help="Constructor args for the model in Python dict format, "'e.g. "{\'config\':\'/path/to/config.yaml\'}" or a direct dict like "{\'dim_rhy\':80}".')
    parser.add_argument("--obj", action="store_true", help="If specified, convert the loaded config dict into a dot-accessible object.")
    
    # Accept multiple input shapes (exsample:--input-shapes "1,80,10" "1,80,10")
    parser.add_argument("--input-shapes", type=str, nargs="+", default=["1,10"], help="One or more comma-separated shapes, e.g. '1,80,10' '1,80,10'. Each shape will produce one dummy tensor.")
    parser.add_argument("--device", type=str, default="cuda", help='Device to run the model on: "cpu" or "cuda" (default: "cuda").')
    parser.add_argument("--output-name", type=str, default="model_graph", help="Base name for the output graph file (no extension needed).")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the output graph file.")
    parser.add_argument("--output-format", type=str, default="png", help="Output file format, e.g. png, svg, pdf.")
    parser.add_argument("--rankdir", type=str, default="TB", help="Direction of graph layout: TB (top-bottom), LR (left-right), etc.")
    parser.add_argument("--node-color", type=str, default="lightblue", help="Node color in the rendered graph.")
    
    return parser.parse_args()

def load_config_file_if_needed(config_value):
    if isinstance(config_value, str) and os.path.isfile(config_value):
        ext = os.path.splitext(config_value)[1].lower()
        if ext in [".yml", ".yaml"]:
            with open(config_value, "r") as f:
                return yaml.safe_load(f)
        elif ext == ".json":
            with open(config_value, "r") as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config file extension: {ext}")
    return config_value

def main():
    args = parse_args()
    raw_dict = ast.literal_eval(args.model_init_args)
    
    if "config" in raw_dict:
        loaded_config = load_config_file_if_needed(raw_dict["config"])
        raw_dict["config"] = loaded_config
        
        if args.obj and isinstance(loaded_config, dict):
            raw_dict["config"] = DictToObj(loaded_config)
    
    spec = importlib.util.spec_from_file_location("model_module", args.model_file)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    model_class = getattr(model_module, args.model_class)
    model = model_class(**raw_dict)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    dummy_inputs = []
    for shape_str in args.input_shapes:
        shape = [int(x) for x in shape_str.split(",")]
        dummy_inputs.append(torch.randn(*shape, device=device))
    
    print("=== Model Summary ===")
    summary(model, input_data=tuple(dummy_inputs), device=device)
    output = model(*dummy_inputs)
    
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.attr(rankdir=args.rankdir)
    dot.node_attr.update(style="filled", color=args.node_color)
    
    output_path = os.path.join(args.output_dir, args.output_name)
    dot.render(output_path, format=args.output_format, cleanup=True)
    print(f"Graph saved as {output_path}.{args.output_format}")

if __name__ == "__main__":
    main()