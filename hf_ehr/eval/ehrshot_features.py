import numpy as np
import torch


def select_patient_representation(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    pad_token_id: int,
    idx: int,
    embed_strat: str,
) -> np.ndarray:
    if embed_strat == "last":
        return hidden_states[idx, -1, :].cpu().numpy()
    if embed_strat == "last_nonpad":
        mask = input_ids[idx] != pad_token_id
        if torch.any(mask):
            pos = torch.nonzero(mask, as_tuple=False)[-1].item()
        else:
            pos = input_ids.shape[1] - 1
        return hidden_states[idx, pos, :].cpu().numpy()
    if embed_strat == "mean":
        mask = input_ids[idx] != pad_token_id
        if torch.any(mask):
            return hidden_states[idx, mask].mean(dim=0).cpu().numpy()
        return hidden_states[idx].mean(dim=0).cpu().numpy()
    raise ValueError(f"Embedding strategy `{embed_strat}` not supported.")
