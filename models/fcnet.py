import logging
import numpy as np
import gymnasium as gym

from torch import Tensor

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, List, ModelConfigDict

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

DIM_Y = 2


class SICPPOFCNet(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        dim_Y: int
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.dim_Y = dim_Y

        hiddens = list(model_config.get("fcnet_hiddens", []))
        activation = model_config.get("fcnet_activation")
        no_final_linear = model_config.get("no_final_linear")

        # The model must have a final layer.
        assert no_final_linear is False

        self.vf_share_layers = model_config.get("vf_share_layers")

        # Do not share parameters!
        assert self.vf_share_layers is False

        self.free_log_std = model_config.get("free_log_std")
        # Generate free-floating bias variables for the second half of
        # the outputs.
        if self.free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two",
                num_outputs,
            )
            num_outputs = num_outputs // 2

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = size

        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        if len(hiddens) > 0:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=hiddens[-1],
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = hiddens[-1]
        if num_outputs:
            self._logits = SlimFC(
                in_size=prev_layer_size,
                out_size=num_outputs,
                initializer=normc_initializer(0.01),
                activation_fn=None,
            )
        else:
            self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[
                -1
            ]

        # Layer to add the log std vars to the state-dependent means.
        if self.free_log_std and self._logits:
            self._append_free_log_std = AppendBiasLayer(num_outputs)

        self._hidden_layers = nn.Sequential(*layers)

        self._value_branch_separate = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            prev_vf_layer_size = int(np.product(obs_space.shape))
            vf_layers = []
            for size in hiddens:
                vf_layers.append(
                    SlimFC(
                        in_size=prev_vf_layer_size,
                        out_size=size,
                        activation_fn=activation,
                        initializer=normc_initializer(1.0),
                    )
                )
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = SlimFC(
            in_size=prev_vf_layer_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

        # Constraint Value Network
        # Build a parallel set of hidden layers for the constraint value net.
        prev_cvf_layer_size = int(np.product(obs_space.shape) + DIM_Y)
        cvf_layers = []
        for size in hiddens:
            cvf_layers.append(
                SlimFC(
                    in_size=prev_cvf_layer_size,
                    out_size=size,
                    activation_fn=activation,
                    initializer=normc_initializer(1.0),
                )
            )
            prev_cvf_layer_size = size
        self._constraint_value_branch_separate = nn.Sequential(*cvf_layers)

        self._constraint_value_branch = SlimFC(
            in_size=prev_cvf_layer_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )
        return

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, Tensor],
        state: List[Tensor],
        seq_lens: Tensor,
    ) -> (Tensor, List[Tensor]):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) if self._logits else self._features
        if self.free_log_std:
            logits = self._append_free_log_std(logits)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> Tensor:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            out = self._value_branch(
                self._value_branch_separate(self._last_flat_in)
            ).squeeze(1)
        else:
            out = self._value_branch(self._features).squeeze(1)
        return out

    def constraint_value_function(self, batch_y_tensor: Tensor, input_dict: Dict[str, Tensor]) -> Tensor:
        # Return shape (batch_size, num_y)
        # obs_flatten has not been computed here.
        obs = input_dict["obs"].float()
        if len(obs.shape) > 2:
            obs = torch.flatten(obs, start_dim=1)
        obs_expand = obs.reshape(obs.shape[0], -1).unsqueeze(1).expand(-1, batch_y_tensor.shape[0], -1)
        y_expand = batch_y_tensor.unsqueeze(0).expand(obs.shape[0], -1, -1)
        obs_y = torch.cat((obs_expand, y_expand), dim=2).reshape(-1, obs.shape[1] + batch_y_tensor.shape[1])
        predict_constraint_val = self._constraint_value_branch(
            self._constraint_value_branch_separate(obs_y)
        ).squeeze(1).reshape(obs.shape[0], batch_y_tensor.shape[0])
        return predict_constraint_val



