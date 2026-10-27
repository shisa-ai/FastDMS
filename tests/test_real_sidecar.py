import torch
from fastdms.models.sidecar import exact_budget_decisions
from training.sidecar import LinearSidecar, future_attention_mass


def test_exact_budget_is_per_sequence_head_and_window_safe():
    scores=torch.ones(20,2)
    decisions=exact_budget_decisions(scores,torch.tensor([0,10,20]),2,2)
    assert decisions.sum(0).tolist()==[6,6]
    assert decisions[:3].all() and decisions[10:13].all()
    assert not decisions[7:10].any() and not decisions[17:].any()


def test_layer_optimizer_only_updates_selected_layer():
    model=LinearSidecar(2,2,3)
    parameters=[torch.nn.Parameter(model.weight[0].detach()),torch.nn.Parameter(model.bias[0].detach())]
    optimizer=torch.optim.AdamW(parameters,lr=0.01)
    torch.nn.functional.linear(torch.ones(4,3),*parameters).sum().backward()
    optimizer.step()
    assert model.weight[0].abs().sum()>0
    assert model.weight[1].abs().sum()==0


def test_query_stride_one_matches_default():
    q=torch.randn(1,4,20,4);k=torch.randn(1,2,20,4)
    torch.testing.assert_close(future_attention_mass(q,k,3),future_attention_mass(q,k,3,query_stride=1))
    assert torch.isfinite(future_attention_mass(q,k,3,query_stride=4)).all()
