import numpy as np
from transformers import AutoConfig, PretrainedConfig


def parse_model_meta(model_name: str) -> dict:
    config = AutoConfig.from_pretrained(model_name)

    model_type = config.architectures[0].split("F")[0].lower()

    if model_type == "opt":
        d_model = config.hidden_size
        d_ffn = config.ffn_dim
        n_head = config.num_attention_heads
        n_layer = config.num_hidden_layers
        n_kv_head = 0

    elif model_type == "llama":
        d_model = config.hidden_size
        d_ffn = config.intermediate_size
        n_head = config.num_attention_heads
        n_layer = config.num_hidden_layers
        n_kv_head = config.num_key_value_heads

    else:
        raise ValueError(f"Model type not supported {model_type}")

    return {
        "model_type": model_type,
        "d_model": d_model,
        "d_ffn": d_ffn,
        "n_head": n_head,
        "n_layer": n_layer,
        "n_kv_head": n_kv_head,
    }


def parse_gemm_shapes(model_name: str, batch_size: int, seq_len: int) -> dict:
    model_meta = parse_model_meta(model_name)
    d_model = model_meta["d_model"]
    d_ffn = model_meta["d_ffn"]
    n_head = model_meta["n_head"]
    head_dim = d_model // n_head

    model_type = model_meta["model_type"]

    gemm_shapes = {
        "fw_attn_qkvo": (
            (batch_size, seq_len, d_model),
            (d_model, d_model),
        ),
        "fw_attn_weight": (
            (batch_size, n_head, seq_len, head_dim),
            (batch_size, n_head, head_dim, seq_len),
        ),
        "fw_attn_output": (
            (batch_size, n_head, seq_len, seq_len),
            (batch_size, n_head, seq_len, head_dim),
        ),
        "bw_input_attn_qkvo": (
            (batch_size, seq_len, d_model),
            (d_model, d_model),
        ),
        "bw_weight_attn_qkvo": (
            (batch_size, d_model, seq_len),
            (batch_size, seq_len, d_model),
        ),
        "bw_input_attn_weight": (
            (batch_size, n_head, seq_len, head_dim),
            (batch_size, n_head, head_dim, seq_len),
        ),
        "bw_weight_attn_weight": (
            (batch_size, n_head, seq_len, seq_len),
            (batch_size, n_head, seq_len, head_dim),
        ),
        "bw_input_attn_output": (
            (batch_size, n_head, seq_len, seq_len),
            (batch_size, n_head, seq_len, head_dim),
        ),
        "bw_weight_attn_output": (
            (batch_size, n_head, seq_len, head_dim),
            (batch_size, n_head, head_dim, seq_len),
        ),
    }

    gemm_shapes["fw_mlp_up"] = (
        (batch_size, seq_len, d_model),
        (d_model, d_ffn),
    )
    gemm_shapes["fw_mlp_down"] = (
        (batch_size, seq_len, d_ffn),
        (d_ffn, d_model),
    )

    gemm_shapes["bw_input_mlp_up"] = (
        (batch_size, seq_len, d_ffn),
        (d_ffn, d_model),
    )
    gemm_shapes["bw_weight_mlp_up"] = (
        (batch_size, d_ffn, seq_len),
        (batch_size, seq_len, d_model),
    )

    gemm_shapes["bw_input_mlp_down"] = (
        (batch_size, seq_len, d_model),
        (d_model, d_ffn),
    )
    gemm_shapes["bw_weight_mlp_down"] = (
        (batch_size, d_model, seq_len),
        (batch_size, seq_len, d_ffn),
    )

    if model_type == "llama":
        gemm_shapes["fw_mlp_gate"] = gemm_shapes["fw_mlp_up"]
        gemm_shapes["bw_input_mlp_gate"] = gemm_shapes["bw_input_mlp_up"]
        gemm_shapes["bw_weight_mlp_gate"] = gemm_shapes["bw_weight_mlp_up"]

    return gemm_shapes


def parse_shapes_rc_shard(
    model_name: str, batch_size: int, seq_len: int
) -> dict:
    shapes = parse_gemm_shapes(model_name, batch_size, seq_len)

    shard_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    shard_shapes = []
    for key, shape in shapes.items():
        input_shape = shape[0]
        weight_shape = shape[1]

        gemm_input_shape = input_shape[-2:]
        gemm_weight_shape = weight_shape[-2:]

        for shard in shard_list:
            if shard > gemm_input_shape[-2] or shard > gemm_weight_shape[-1]:
                continue
            shard_input_remainder = gemm_input_shape[-2] % shard
            shard_weight_remainder = gemm_weight_shape[-1] % shard
            shard_shapes.append(
                ((shard, gemm_input_shape[-1]), (gemm_weight_shape[-2], shard))
            )

            if shard_input_remainder != 0:
                shard_shapes.append(
                    (
                        (shard_input_remainder, gemm_input_shape[-1]),
                        (gemm_weight_shape[-2], shard),
                    )
                )
            if shard_weight_remainder != 0:
                shard_shapes.append(
                    (
                        (shard, gemm_input_shape[-1]),
                        (gemm_weight_shape[-2], shard_weight_remainder),
                    )
                )

            if shard_input_remainder != 0 and shard_weight_remainder != 0:
                shard_shapes.append(
                    (
                        (shard_input_remainder, gemm_input_shape[-1]),
                        (gemm_weight_shape[-2], shard_weight_remainder),
                    )
                )

    # get all the unique shapes
    unique_shapes = []
    for shape in shard_shapes:
        if shape not in unique_shapes:
            unique_shapes.append(shape)

    return unique_shapes


def parse_shapes_summa_shard(
    model_name: str, batch_size: int, seq_len: int
) -> dict:
    shapes = parse_gemm_shapes(model_name, batch_size, seq_len)
    shard_list = [128, 256, 512, 1024, 2048]

    shard_shapes = []
    for key, shape in shapes.items():
        input_shape = shape[0]
        weight_shape = shape[1]

        gemm_input_shape = input_shape[-2:]
        gemm_weight_shape = weight_shape[-2:]

        for shard in shard_list:
            if shard > gemm_input_shape[-2] or shard > gemm_weight_shape[-1]:
                continue

            shard_shapes.append(((shard, shard), (shard, shard)))

            shard_input_remainder = gemm_input_shape[-2] % shard
            shard_weight_remainder = gemm_weight_shape[-1] % shard

            if shard_input_remainder != 0:
                shard_shapes.append(
                    ((shard_input_remainder, shard), (shard, shard))
                )
            if shard_weight_remainder != 0:
                shard_shapes.append(
                    ((shard, shard), (shard, shard_weight_remainder))
                )

            if shard_input_remainder != 0 and shard_weight_remainder != 0:
                shard_shapes.append(
                    (
                        (shard_input_remainder, shard),
                        (shard, shard_weight_remainder),
                    )
                )

    # get all the unique shapes
    unique_shapes = []
    for shape in shard_shapes:
        if shape not in unique_shapes:
            unique_shapes.append(shape)

    return unique_shapes


if __name__ == "__main__":
    model_names = [
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
        "facebook/opt-6.7b",
        "facebook/opt-13b",
        "facebook/opt-30b",
        "facebook/opt-66b",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Llama-2-70b-hf",
    ]

    for model_name in model_names:
        print(f"Model: {model_name}")
        shapes = parse_shapes_summa_shard(model_name, 128, 1024)
        print(f"Number of unique shapes: {len(shapes)}")
        print(shapes)
        print("=" * 80)

        shapes = parse_shapes_rc_shard(model_name, 128, 1024)
        print(f"Number of unique shapes: {len(shapes)}")
        print(shapes)
        print("=" * 80)
