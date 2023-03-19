from torch import nn
import torch
import sys
sys.path.insert(0, "/home/gamir/DER-Roei/alon/BLIP")
import torch.nn.functional as F


from models.blip_retrieval_vg import blip_retrieval_vg as blip
from argparse import Namespace

class BLIP_Wrapper(nn.Module):
    def __init__(
            self,
            model):
        super().__init__()
        self.model = model

    def encode_image(self, image, device,  norm=True):
        image_embeds = self.model.visual_encoder(image)         
        image_feat = F.normalize(self.model.vision_proj(image_embeds[:,0,:]),dim=-1)

        return image_feat

    def encode_text(self, text, device,  norm=True):
        text = self.model.tokenizer(text, padding='max_length', truncation=True, max_length=35, 
                                    return_tensors="pt").to(device)
        text_output = self.model.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_feat = F.normalize(self.model.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)

        return text_feat

def get_zeroshot_model(config, device):
    """
    Specify your model here

    """


    model = blip(
    pretrained = config["MODEL"]["pretrained"],
    med_config = config["MODEL"]["med_config"],
    image_size = config["MODEL"]["image_size"],
    vit = config["MODEL"]["vit"],
    vit_grad_ckpt=config["MODEL"]["vit_grad_ckpt"],
    vit_ckpt_layer = config["MODEL"]["vit_ckpt_layer"],
    queue_size=config["MODEL"]["queue_size"],
    negative_all_rank=config["MODEL"]["negative_all_rank"],
    args = Namespace(objects = config["MODEL"]["objects"],
                    object_tokens = config["MODEL"]["object_tokens"],
                    relations = config["MODEL"]["relations"],
                    relation_tokens = config["MODEL"]["relation_tokens"],
                    prompt_attention = config["MODEL"]["prompt_attention"],
                    prompt_attention_full = config["MODEL"]["prompt_attention_full"],
                    mask_layers = config["MODEL"]["mask_layers"],
                    text_lora = config["MODEL"]["text_lora"],
                    image_lora = config["MODEL"]["image_lora"],
                    lora = config["MODEL"]["lora"],
                    prompts_lora = config["MODEL"]["prompts_lora"],
                    loss_ce = config["MODEL"]["loss_ce"],
                    negatives = config["MODEL"]["negatives"],
                    random_graph_ablation = config["MODEL"]["random_graph_ablation"],
                    vg_loss_lambda = config["MODEL"]["vg_loss_lambda"],
                    vg_batch_size = config["MODEL"]["vg_batch_size"],
                    device=device
                    )
)
    if config["MODEL"]["checkpoint_path"] != "":
        checkpoint = torch.load(config["MODEL"]["checkpoint_path"], map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            sd = checkpoint["state_dict"]
            if True and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
    return BLIP_Wrapper(model)