from functools import partial

def run_gemma_with_activations(module: Module, variables, *,
                               embedded, positions, mask, adarms_cond):
    """Run gemma.Module and return (outputs, kv_cache, action_expert_activations_per_layer)."""

    # Capture outputs of every Block.__call__
    def block_filter(mod, _, __):
        return isinstance(mod, Block)

    (outputs, kv_cache), intermediates = module.apply(
        variables,
        embedded,
        positions,
        mask,
        adarms_cond=adarms_cond,
        mutable=["intermediates"],
        capture_intermediates=block_filter,
    )

    # intermediates["intermediates"] is a nested dict of module paths -> list of values
    # For a scanned Block, each entry is shape [depth, ...].
    # Each Block.__call__ returns (xs, kv_cache). We captured the *output*,
    # which is the `xs` list at that layer.
    all_block_outputs = []
    for _, v in intermediates["intermediates"].items():
        # v is a list like [{"xs": [...], "kv_cache": ...}] or just [xs, kv_cache],
        # depending on Flax version. Most commonly capture_intermediates returns the
        # positional outputs; here we grabbed Block.__call__'s return value.
        # In this code, Block.__call__ returns (xs, kv_cache) -> we want xs.
        xs_per_layer, _ = v  # scanned over depth, so xs_per_layer has leading depth
        all_block_outputs.append(xs_per_layer)

    # If there is only one scanned Block, that'll be length-1. Take that.
    # xs_per_layer is of shape: [depth, num_experts, batch, seq, width]
    # or a pytree of length num_experts.
    # In our case xs is a list [x_expert0, x_expert1].
    # After scan, xs_per_layer is a list with length num_experts,
    # each being [depth, B, T, D].
    # So to get action expert (index 1):
    xs_expert0, xs_expert1 = all_block_outputs[0]  # unpack experts
    action_expert_acts = xs_expert1  # shape [depth, B, T, D]

    return outputs, kv_cache, action_expert_acts