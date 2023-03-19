from torch import nn
import torch
import sys
sys.path.insert(0, "/home/gamir/DER-Roei/alon/open_clip_ref/src")



from open_clip import create_model_and_transforms, trace_model, mark_only_lora_as_trainable, image_transform_vg

class Open_Clip_Wrapper(nn.Module):
    def __init__(
            self,
            model):
        super().__init__()
        self.model = model

    def encode_image(self, image, norm=True):
        x = self.model.encode_image(image)[0]

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    def encode_text(self, text, norm=True):
        x = self.model.encode_text(text)

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

def get_zeroshot_model(config, device):
    """
    Specify your model here
    """
    model, _, _ = create_model_and_transforms(
    config["MODEL"]["model"],
    config["MODEL"]["pretrained"],
    config["MODEL"]["lora"],
    config["MODEL"]["image_lora"],
    config["MODEL"]["text_lora"],
    config["MODEL"]["prompt_tokens"],
    config["MODEL"]["prompt_attention"],
    config["MODEL"]["prompt_attention_full"],
    config["MODEL"]["mask_attention"],
    precision=config["MODEL"]["precision"],
    device=device,
)
    if config["MODEL"]["checkpoint_path"] != "":
        checkpoint = torch.load(config["MODEL"]["checkpoint_path"], map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            sd = checkpoint["state_dict"]
            if True and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
    return Open_Clip_Wrapper(model)