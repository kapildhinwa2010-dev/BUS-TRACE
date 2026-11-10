"""Optional PyTorch adapter: capture block writes with forward hooks.

Requires torch, which is not installed by default. Use this only if
your model runs in PyTorch and exposes per-block submodules.
"""


def capture_writes(model, example_input, name_filter=None):
    """Run one forward pass and capture leaf-module outputs as writes.

    ``name_filter`` is an optional callable that takes a module name
    and returns True for the modules whose outputs should be captured.
    """
    import torch

    writes = []
    handles = []

    def hook(module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        if out is not None and hasattr(out, "detach"):
            writes.append(out.detach().float().cpu().numpy())

    for name, module in model.named_modules():
        if name_filter is not None and not name_filter(name):
            continue
        if len(list(module.children())) == 0:
            handles.append(module.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(example_input)
    if was_training:
        model.train()
    for h in handles:
        h.remove()
    return writes