import argparse
import importlib.util
import torch
import torch.nn as nn
from torchinfo import summary
from torchviz import make_dot

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize PyTorch Model Graph")

    # モデルクラスを含む Python ファイルのパス (例: path/to/model.py)
    parser.add_argument(
        "--model-file",
        type=str,
        required=True,
        help="Path to the Python file that defines the model class",
    )

    # ファイル内に定義されたクラス名 (例: SimpleModel)
    parser.add_argument(
        "--model-class",
        type=str,
        required=True,
        help="Name of the model class inside the specified Python file",
    )

    # 入力サイズを可変長引数で受け取れるようにする (例: --input-size 1 10)
    # 例だと (batch_size=1, features=10) を想定
    parser.add_argument(
        "--input-size",
        type=int,
        nargs="+",
        default=[1, 10],
        help="Input size for the model, e.g. --input-size 1 10 for a (1,10) shape",
    )

    # 出力ファイル名 (拡張子は自動で付与される)
    parser.add_argument(
        "--output-name",
        type=str,
        default="model_graph",
        help="Base name for the output graph file (no extension needed)",
    )

    # 出力フォーマット (png, svg, pdf, など)
    parser.add_argument(
        "--output-format",
        type=str,
        default="png",
        help="Format for the output file, e.g. png, svg, pdf, etc.",
    )

    # グラフのカスタマイズオプション
    parser.add_argument(
        "--rankdir",
        type=str,
        default="TB",
        help="Direction for graph layout: TB (top-bottom) or LR (left-right), etc.",
    )
    parser.add_argument(
        "--node-color",
        type=str,
        default="lightblue",
        help="Node color for the graph",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # 動的インポート
    spec = importlib.util.spec_from_file_location("model_module", args.model_file)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # モデルクラスの取得・インスタンス化
    model_class = getattr(model_module, args.model_class)
    model = model_class()  # 引数がある場合は適宜追加

    # モデルサマリを表示 (torchinfo)
    print("=== Model Summary ===")
    summary(model, input_size=tuple(args.input_size))

    # ダミー入力を用意して forward
    input_tensor = torch.randn(*args.input_size)
    output = model(input_tensor)

    # 計算グラフを作成
    dot = make_dot(output, params=dict(model.named_parameters()))

    # グラフをカスタマイズ
    dot.attr(rankdir=args.rankdir)  # 左右方向 or 上下方向
    dot.node_attr.update(style="filled", color=args.node_color)

    # グラフを保存
    dot.render(args.output_name, format=args.output_format, cleanup=True)
    print(f"Graph saved as {args.output_name}.{args.output_format}")

if __name__ == "__main__":
    main()