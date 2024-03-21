from diffusers.models.attention_processor import Attention
import torch
import math


class AttnProcessorAttnMapEdit:
    r"""
    Default processor for performing attention-related computations.
    Taken from diffusers package
    Modified to support attention map editing
    """

    @staticmethod
    def find_target_id(
        attention_probs: torch.Tensor, replace_id: int, num_attention_heads: int
    ):
        """
        determine the target id to be edited, for now just edit one target
        return: torch.tensor of shape (batch_size, 1) and the index of the second largest attention value
        """
        batch_size = attention_probs.shape[0]
        relevant_attn = attention_probs[batch_size // 2 :, :, 1:replace_id]

        # find max attention value for each token (max over attn heads and attn maps)
        max_relevant_attn = relevant_attn.max(dim=-2)[0]
        max_relevant_attn = max_relevant_attn.reshape(
            batch_size // (2 * num_attention_heads),
            num_attention_heads,
            max_relevant_attn.shape[1],
        )
        max_max_relevant_attn = max_relevant_attn.max(dim=-2)[0]

        # for now, apply a very simple heuristic, that is find the index of the largerst max attention value
        val, index = torch.topk(
            max_max_relevant_attn, k=max_max_relevant_attn.shape[-1], dim=-1
        )
        idx = index[:, 0]

        # if the second largerst value is comparable to the largerst, then we don't edit
        no_edit_flag = (val[:, 0] * 0.9 - val[:, 1]) < 0
        idx[no_edit_flag] = -2

        # recover the idx to be w.r.t. the original attntion probs
        idx += 1

        return idx[:, None]

    @staticmethod
    def _get_relevant_attn_and_replace_attn(
        attention_probs: torch.Tensor, eos_id: int, replace_with: str = "first_padding"
    ):
        """get the relevant attention (non-empty prompt part excluding eos, sos, padding), replace attention slice (attention to be replaced)"""
        batch_size = attention_probs.shape[0]
        relevant_attn = attention_probs[batch_size // 2 :, :, 1:eos_id]
        if replace_with == "eos":
            replace_id = eos_id
        elif replace_with == "first_padding":
            replace_id = eos_id + 1
        elif replace_with == "none":
            # this is for the optimisation approach
            replace_id = eos_id
            relevant_attn = attention_probs[batch_size // 2 :]
        else:
            raise NotImplementedError
        replace_attn = attention_probs[batch_size // 2 :, :, replace_id]
        return relevant_attn, replace_attn

    @staticmethod
    def similar_distribution(
        modified_maps: torch.tensor, target_ids: torch.tensor, threshold: float = 0.8
    ):
        # x is of shape (width, width, seq_len)
        width = modified_maps.shape[1]
        return torch.mean(
            (
                torch.sum(modified_maps / (width**2), dim=(-2, -3))
                - (threshold / target_ids.shape[0])
            )
            ** 2
        )

    def loss_fn(
        self,
        modified_maps: torch.tensor,
        target_ids: torch.tensor,
        weights: torch.tensor,
        threshold: float = 0.8,
        gamma: float = 1.0,
        delta: float = 1.0,
        epsilon: float = 1.0,
    ):
        """
        threshold: hyperparameter to scale the determine the target proportion
        gamma: hyperparameter to scale tgt_proportion_cost
        delta: hyperparameter to scale equal_proportion_cost
        epsilon: hyperparameter to scale weights_cost
        """
        # original_maps is of shape (batch_size, width, width, seq_len), and has been softmaxed
        batch_size, width, _, _ = modified_maps.shape
        # ensure attention maps have sufficiently high values
        c2 = torch.mean(
            (
                torch.quantile(
                    modified_maps.view(batch_size, -1, target_ids.shape[0]), 0.9, dim=-2
                )
                - 0.9
            )
            ** 2
        )
        tgt_proportion_cost = c2
        # ensure the target attention maps have similar distributions to each other
        sim_dist_cost = self.similar_distribution(modified_maps, target_ids, threshold)
        weights_cost = torch.mean(weights**2)
        return (
            gamma * tgt_proportion_cost + delta * sim_dist_cost + epsilon * weights_cost
        )

    def _optimise_attention_maps(
        self,
        attention_probs,
        relevant_attn,
        target_ids,
        internals,
        current_step,
        max_iter: int = 20,
        lr: float = 1e2,
        decay: int = 0.25,
        global_decay: float = 0.99,
    ):
        """
        Hyperparameters:
        - max_iter: maximum number of iterations for the optimisation
        - lr: learning rate for the optimisation
        """
        # attention_probs shape: (num_batch x num_heads, H, W, seq_len);

        target_ids = torch.tensor(target_ids[0]).to(
            relevant_attn.device
        )  # no need to account for SOS as this is the entire sequence length (77)
        batch_size, width, _, seq_len = relevant_attn.shape

        X = torch.arange(0, width, 1, dtype=torch.float32)
        y = X.view(-1, 1)
        # generate centres guassian additive patterns
        sigma = width // 16
        centres = []
        for i in range(2 * sigma, width, 2 * sigma):
            for j in range(2 * sigma, width, 2 * sigma):
                centres.append((i, j))

        temp = [
            torch.exp(-((X - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2)).unsqueeze(0)
            for (x0, y0) in centres
        ]
        gaussian_maps_tensor = torch.cat(temp, dim=0)
        gaussian_maps_tensor = (
            gaussian_maps_tensor.to(relevant_attn.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .requires_grad_(True)
        )

        best_loss = float("inf")

        torch.set_grad_enabled(True)
        torch.autograd.set_detect_anomaly(True)

        # generate weights
        weights = (
            torch.rand(
                (batch_size, len(centres), len(target_ids)),
                dtype=torch.float32,
                device=relevant_attn.device,
                requires_grad=True,
            )
            .unsqueeze(-2)
            .unsqueeze(-2)
        )
        additive_attn = torch.zeros_like(
            relevant_attn, dtype=torch.float32, device=relevant_attn.device
        )
        # threshold should depend on the number of targets
        threshold = min(1.0, 0.2 * target_ids.shape[0])

        if current_step != 0:
            ma_attn = internals["ma_attn"].pop(0).to(relevant_attn.device)
        else:
            if "ma_attn" not in internals:
                internals["ma_attn"] = []
            ma_attn = torch.sum(gaussian_maps_tensor * weights, dim=1)
        # optimise the weights
        for j in range(max_iter):
            additive_attn[:, :, :, target_ids] = torch.sum(
                gaussian_maps_tensor * weights, dim=1
            )
            modified_maps = relevant_attn + additive_attn
            # gamma: target proportion cost, delta: equal proportion cost, epsilon: weights regulation
            loss = self.loss_fn(
                modified_maps.softmax(dim=-1)[:, :, :, target_ids],
                target_ids,
                weights,
                threshold=threshold,
                gamma=1,
                delta=80,
                epsilon=9e-2,
            )
            if loss < best_loss:
                best_loss = loss
                best_additive_attn = (
                    additive_attn.clone().detach().requires_grad_(False)
                )

            grad = torch.autograd.grad(loss, [weights], retain_graph=False)[0]
            weights = weights - batch_size * lr * grad

            # clear the gradients
            weights.grad = None
        if internals is not None:
            internals["ma_attn"].append(
                best_additive_attn[:, :, :, target_ids].clone().detach().cpu()
            )

        torch.set_grad_enabled(False)
        torch.autograd.set_detect_anomaly(False)

        # regulate impact of edit
        best_additive_attn[:, :, :, target_ids] = ma_attn * decay + best_additive_attn[
            :, :, :, target_ids
        ] * (1 - decay)
        current_decay = global_decay**current_step
        best_relevant_attn = (relevant_attn + current_decay * best_additive_attn).to(
            torch.float16
        )
        attention_probs[attention_probs.shape[0] // 2 :] = best_relevant_attn
        torch.set_grad_enabled(False)
        return attention_probs

    def optimise_edit_attn(
        self,
        attention_probs: torch.tensor,
        target_ids: torch.Tensor,
        eos_id: int,
        internals: dict,
        current_step: int,
    ):
        relevant_attn, _ = self._get_relevant_attn_and_replace_attn(
            attention_probs, eos_id, replace_with="none"
        )
        attention_probs = self._process_attn_head_dim_3d_to_4d_mapsize(attention_probs)
        relevant_attn = self._process_attn_head_dim_3d_to_4d_mapsize(relevant_attn)
        attention_probs = self._optimise_attention_maps(
            attention_probs, relevant_attn, target_ids, internals, current_step
        )
        attention_probs = self._process_attn_head_dim_4d_mapsize_to_3d(attention_probs)

        return attention_probs

    @staticmethod
    def _process_attn_head_dim_3d_to_4d_mapsize(attention_probs: torch.tensor):
        """
        Converts attention probs from 3D tensor shape (true_batch_size x attn_heads,  map_size x map_size, seq_len)
        to a 4D tensor of shape (true_batch_size x attn_heads, map_size, map_size, seq_len)
        """
        num_batch, attn_map_dim_sq, seq_len = attention_probs.shape
        width = int(math.sqrt(attn_map_dim_sq))
        processed_attn_probs = attention_probs.reshape(num_batch, width, width, seq_len)
        return processed_attn_probs

    @staticmethod
    def _process_attn_head_dim_4d_mapsize_to_3d(attention_probs: torch.tensor):
        """
        Converts attention probs from 4D tensor of shape (true_batch_size x attn_heads, map_size, map_size, seq_len)
        to a 3D tensor shape (true_batch_size x attn_heads,  map_size x map_size, seq_len)
        """
        num_batch, width, _, seq_len = attention_probs.shape
        processed_attn_probs = attention_probs.reshape(num_batch, -1, seq_len)
        return processed_attn_probs

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        target_ids: list = None,
        replace_id=None,
        internals=None,
        current_step=None,
        num_edit_steps=None,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        num_heads = attn.heads

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        dtype = query.dtype
        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device,
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1
        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=query.size(-1) ** -0.5,
        )
        del baddbmm_input
        attention_probs = attention_scores.to(dtype)
        del attention_scores
        if target_ids is None:
            target_ids = self.find_target_id(attention_probs, replace_id, attn.heads)

        temp = attention_probs.softmax(dim=-1)
        sos, eos = temp[:, :, 0], temp[:, :, replace_id]
        width = int(math.sqrt(attention_probs.shape[-2]))
        del temp
        # stagger edit steps
        if current_step in [_ for _ in range(num_edit_steps)]:
            attention_probs = self.optimise_edit_attn(
                attention_probs=attention_probs,
                target_ids=target_ids,
                eos_id=replace_id,
                internals=internals,
                current_step=current_step,
            )

        # softmax; comment if editing after softmax
        attention_probs = attention_probs.softmax(dim=-1)

        attention_probs[:, :, 0] = sos
        attention_probs[:, :, replace_id] = eos

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states
