import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union


class SwinModel(nn.Module):
    def __init__(self, swin_model, device, feat_size=768):
        super(SwinModel, self).__init__()
        self.swin_model = swin_model
        self.device = device
        self.feat_size = feat_size

    def forward(self, pixel_values: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,):

        return_dict = return_dict if return_dict is not None else self.swin_model.config.use_return_dict

        outputs = self.swin_model.swin(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # bs, 49, 768
        pool_output = torch.mean(sequence_output, dim=1)  # bs, 768
        # Reshape to (batch_size, num_channels, height, width)
        sequence_output = sequence_output.transpose(1, 2)
        batch_size, num_channels, sequence_length = sequence_output.shape
        height = width = math.floor(sequence_length ** 0.5)
        sequence_output = sequence_output.reshape(batch_size, num_channels, height, width)

        # Reconstruct pixel values
        reconstructed_pixel_values = self.swin_model.decoder(sequence_output)

        masked_im_loss = None
        if bool_masked_pos is not None:
            size = self.swin_model.config.image_size // self.swin_model.config.patch_size
            bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
            mask = (
                bool_masked_pos.repeat_interleave(self.swin_model.config.patch_size, 1)
                .repeat_interleave(self.swin_model.config.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
            )
            reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
            masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.swin_model.config.num_channels

        return masked_im_loss, pool_output