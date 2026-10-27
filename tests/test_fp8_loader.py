import json
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM as HFLlama
from fastdms.models.llama import LlamaForCausalLM
from fastdms.utils.loader import load_model


def test_channel_fp8_checkpoint_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    config = LlamaConfig(vocab_size=32, hidden_size=56, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=14,
                         num_key_value_heads=2, head_dim=4)
    hf = HFLlama(config)
    tensors = {}
    for name, tensor in hf.state_dict().items():
        if name.endswith('_proj.weight'):
            scale = tensor.abs().amax(1, keepdim=True).float().clamp_min(1e-8) / 448
            tensors[name] = (tensor / scale).to(torch.float8_e4m3fn)
            tensors[name.replace('.weight', '.weight_scale')] = scale
        else:
            tensors[name] = tensor
    save_file(tensors, str(tmp_path / 'model.safetensors'))
    config.quantization_config = dict(quant_method='compressed-tensors', format='float-quantized',
        config_groups={'group_0': dict(weights=dict(strategy='channel', num_bits=8, type='float', symmetric=True),
            input_activations=dict(strategy='token', dynamic=True, num_bits=8, type='float'))})
    config.save_pretrained(tmp_path)
    model = LlamaForCausalLM(config)
    load_model(model, str(tmp_path))
    layer = model.model.layers[0]
    qkv = layer.self_attn.qkv_proj
    expected = torch.cat([tensors[f'model.layers.0.self_attn.{p}_proj.weight'].float() for p in ('q','k','v')])
    torch.testing.assert_close(qkv.weight_fp8_t.t().float(), expected, rtol=0, atol=0)
    assert qkv.weight.numel() == 0
    from fastdms.layers.linear import convert_linear_modules_to_fp8
    assert convert_linear_modules_to_fp8(model) == 0
    x = torch.randn(3, 56)
    actual = qkv(x)
    reference = torch.nn.functional.linear(x, expected * qkv.weight_scale_inv.t())
    torch.testing.assert_close(actual, reference)
    down = layer.mlp.down_proj
    x = torch.randn(3, 64)
    reference = torch.nn.functional.linear(x, tensors['model.layers.0.mlp.down_proj.weight'].float() * tensors['model.layers.0.mlp.down_proj.weight_scale'])
    torch.testing.assert_close(down(x), reference)
    # Long prefill chunks must preserve the same linear result.
    x = torch.randn(4101, 64)
    reference = torch.nn.functional.linear(x, tensors['model.layers.0.mlp.down_proj.weight'].float() * tensors['model.layers.0.mlp.down_proj.weight_scale'])
    torch.testing.assert_close(down(x), reference)
