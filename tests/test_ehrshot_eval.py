import numpy as np
import torch

from hf_ehr.eval.ehrshot_features import select_patient_representation


def test_select_patient_representation_uses_last_nonpad_token():
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    input_ids = torch.tensor(
        [
            [10, 11, 12, 0, 0],
            [0, 0, 20, 21, 22],
        ]
    )

    first = select_patient_representation(hidden, input_ids, pad_token_id=0, idx=0, embed_strat="last_nonpad")
    second = select_patient_representation(hidden, input_ids, pad_token_id=0, idx=1, embed_strat="last_nonpad")

    np.testing.assert_array_equal(first, hidden[0, 2].numpy())
    np.testing.assert_array_equal(second, hidden[1, 4].numpy())
