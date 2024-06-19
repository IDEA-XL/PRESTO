from typing import List, Dict
from abc import ABC, abstractmethod

from torch.nn.functional import conv1d
import torch
import logging

from presto.modalities.base_modality import Modality

class LMMMetaModel:
    def __init__(self, config):
        super(LMMMetaModel, self).__init__(config)

    def _load_projector_weights(self, weights: Dict):
        weights = {k.replace("base_model.", "").replace("model.", ""): v for k, v in weights.items()}
        logging.info(f"Loading pretrained weights: {list(weights.keys())}")
        load_result = self.load_state_dict(weights, strict=False)
        assert (
            len(load_result.unexpected_keys) == 0
        ), "Unexpected weights, is this the right model?"

    def initialize_pretrained_modules(self, modalities: List[Modality], weights: Dict):
        for m in modalities:
            projector = m.build_projector(self.config.hidden_size)
            setattr(self, m.name + "_lmm_projector", projector)

        self._load_projector_weights(weights)

    def initialize_modules(self, modalities: List[Modality], weights: Dict):
        names = [m.name for m in modalities]

        self.config.modalities = names

        for m in modalities:
            projector = m.build_projector(self.config.hidden_size)
            setattr(self, m.name + "_lmm_projector", projector)

        self._load_projector_weights(weights)


class LMMMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self) -> "LMMMetaForCausalLM":
        pass

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, **kwargs
    ):
        model = self.get_model()

        batch_size, seq_len = input_ids.shape
        
        # if no modality is present, we can just return the input_ids
        # for now, only support one modality
        for m in self.modalities:
            if kwargs.get(m.name) is None:
                return input_ids, attention_mask, past_key_values, None, labels

        # batch_size x seq_len x embedding_hidden_size
        inputs_embeds = torch.zeros(
            (batch_size, seq_len, self.config.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )

        # modality x batch_size x instance_idx x modality_token_width x embedding_hidden_size
        projected_tensors = []
        # assuming that if caching is enabled, we'll never have past_key_values AND need to encode the instruction modality values
        if past_key_values is None:
            for m in self.modalities:
                m_vals = m.forward(kwargs.get(m.name))
                mp_vals = []
                proj = getattr(model, m.name + "_lmm_projector")

                # project each batch into language model token space
                for m_val in m_vals:
                    # each m_val is 'instance_idx x modality_token_width x embedding_hidden_size', but 'modality_token_width' is variable
                    instance_val_list = []
                    for each_instance in m_val: # each_instance is 'modality_token_width x embedding_hidden_size'
                        if each_instance is not None:
                            instance_val = proj(each_instance) # 'modality_token_width x lm_hidden_size'
                        else:
                            # set zero value for instance
                            # molecule_2d node features as 300
                            each_instance = torch.zeros((1, 300), device=self.device, dtype=self.dtype)
                            instance_val = proj(each_instance)
                        instance_val_list.append(instance_val)
                    mp_vals.append(instance_val_list)
                    
                projected_tensors.append(mp_vals)

        indices = None
        for i, input_ids_sample in enumerate(input_ids):
            is_text_mask = input_ids_sample >= 0

            # fill in all the LLM-based text embeddings
            inputs_embeds[i, is_text_mask] = model.embed_tokens(
                input_ids_sample[is_text_mask]
            )

            # skip if all tokens are text tokens
            if is_text_mask.sum() == seq_len:
                continue
            assert (
                past_key_values is None
            ), "We shouldn't have cached keys if this is the first instruction pass"

            for mi, m in enumerate(self.modalities):
                # locate the group of tokens for this modality
                m_mask = (input_ids_sample == m.token_idx).float()
                
                # # Below: for constant token width
                # m_kernel = torch.tensor(
                #     [-1] * m.token_width, dtype=m_mask.dtype, device=m_mask.device
                # )
                # m_conv = conv1d(
                #     m_mask.unsqueeze(0).unsqueeze(0),
                #     m_kernel.unsqueeze(0).unsqueeze(0),
                # )

                # # where do we see `token_width`-tokens in a row?
                # indices = (m_conv[0, 0] == -m.token_width).nonzero(as_tuple=True)[0]
                
                instances_token_width = [instance.shape[0] for instance in projected_tensors[mi][i]] 
                # find start indices of each instance
                indices = []
                ii = 0
                while ii < len(m_mask):
                    if m_mask[ii] == 1:
                        indices.append(ii) # find one instance
                        ii += instances_token_width[len(indices) - 1]
                    else:
                        ii += 1       
                    if len(indices) == len(instances_token_width):
                        break # early stop if we've found all instances        

                # fill these embeddings with the projected modality tensor
                last_covered_idx = -1
                for k, possible_token_idx in enumerate(indices):
                    if possible_token_idx <= last_covered_idx:
                        # make sure we don't overwrite an instance we've already covered
                        # handles bug caused by back-to-back tokens
                        continue
                    batch_modality_tensor = projected_tensors[mi][i][k]
                    try:
                        inputs_embeds[
                            i, possible_token_idx : possible_token_idx + instances_token_width[k]
                        ] = batch_modality_tensor
                    except:
                        breakpoint()
                    last_covered_idx = possible_token_idx + instances_token_width[k] - 1

        del input_ids
        
        # @open-mol flatten the projected_tensors into a single tensor to avoid deepspeed all_reduce() hanging issue
        # why? since if no modality is present, the gradient of projected_tensors is None, which will cause deepspeed all_reduce() hanging issue
        try:
            projected_tensors = torch.stack(
                [
                    torch.cat([torch.cat(batch, dim=0) for batch in modality], dim=0)
                    for modality in projected_tensors
                ]
            )
        except:
            projected_tensors = None
        return None, attention_mask, past_key_values, inputs_embeds, labels, projected_tensors
