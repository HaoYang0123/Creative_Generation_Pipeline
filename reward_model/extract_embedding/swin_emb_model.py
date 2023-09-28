import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union


class SwinModel(nn.Module):
    #--swin-model="microsoft/swin-tiny-patch4-window7-224"  ===> feat_size=768
    #--swin-model="microsoft/swin-base-patch4-window7-224"  ===> feat_size=1024
    def __init__(self, swin_model, image_processor, device, feat_size=768, class_num_1=1000, class_num_2=1000, class_num_3=1000):
        super(SwinModel, self).__init__()
        self.swin_model = swin_model
        self.image_processor = image_processor
        self.device = device
        self.fc_level_1 = nn.Linear(feat_size, class_num_1)
        self.fc_level_2 = nn.Linear(feat_size, class_num_2)
        self.fc_level_3 = nn.Linear(feat_size, class_num_3)

    def forward_swin(self, pixel_values: Optional[torch.FloatTensor] = None,
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
        # if not return_dict:
        #     output = (reconstructed_pixel_values,) + outputs[2:]
        #     return ((masked_im_loss,) + output) if masked_im_loss is not None else output
        #
        # return SwinMaskedImageModelingOutput(
        #     loss=masked_im_loss,
        #     logits=reconstructed_pixel_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        #     reshaped_hidden_states=outputs.reshaped_hidden_states,
        # )

    def forward(self, image, bool_masked_pos):
        masked_loss, img_feat = self.forward_swin(image, bool_masked_pos)
        level1_prob = self.fc_level_1(img_feat)  # bs, #level1
        level2_prob = self.fc_level_2(img_feat)  # bs, #level2
        level3_prob = self.fc_level_3(img_feat)  # bs, #level3
        return level1_prob, level2_prob, level3_prob, masked_loss, img_feat


# print("start")
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
#
# print("get image", image)
# image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-simmim-window6-192")
# model = SwinForMaskedImageModeling.from_pretrained("microsoft/swin-base-simmim-window6-192")
#
# num_patches = (model.config.image_size // model.config.patch_size) ** 2
# pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
# # create random boolean mask of shape (batch_size, num_patches)
# bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()
#
# outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
# loss, reconstructed_pixel_values = outputs.loss, outputs.logits
# list(reconstructed_pixel_values.shape)