from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision as tv
from peft import LoraConfig, TaskType, get_peft_model,PeftModel
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Trainer, TrainingArguments

from .base_vlm import BaseVLM
from .data import CaptionDataset, MultiChoiceQADataset
import time
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from collections import OrderedDict

import torch.nn.functional as F
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BF16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

torch.backends.cuda.matmul.allow_tf32 = True  # Enable tf32 for better performance
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  


def load(model_name: str = "clip_model"):
    from .base_vlm import BaseVLM
    path = Path(__file__).parent / model_name
    vlm = BaseVLM()
    clip = CLIP(vlm.model.model.vision_model, vlm.model.model.text_model)
    clip = PeftModel.from_pretrained(clip, str(path))
    clip.load_pretrained(str(path))
    clip.eval().to(device)
    if USE_BF16:
        clip = clip.to(dtype=torch.bfloat16)
    return clip

def get_image_transform(train: bool = True):
    if train:
        return tv.transforms.Compose([
            tv.transforms.Resize(384),
            tv.transforms.RandomResizedCrop(384, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
    else:
        return tv.transforms.Compose([
            tv.transforms.Resize(384),
            tv.transforms.CenterCrop(384),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])



def clip_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Custom data collator for CLIP training.
    """
    max_len = max(f["input_ids"].shape[0] for f in features)

    def pad_tensor(tensor, pad_value):
        return torch.cat([tensor, torch.full((max_len - tensor.shape[0],), pad_value, dtype=tensor.dtype)])

    input_ids = torch.stack([pad_tensor(f["input_ids"], pad_value=processor.tokenizer.pad_token_id) for f in features])
    attention_mask = torch.stack([pad_tensor(f["attention_mask"], pad_value=0) for f in features])
    pixel_values = torch.stack([f["pixel_values"] for f in features])


    labels = torch.arange(len(features))
    return {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.long(),
        "pixel_values": pixel_values.float(),
        "labels": labels.long(),  
    }


# generative AI is used to generate idea, code
class OptimizedCaptionDatasetForTraining(Dataset):
    def __init__(self, dataset, processor: AutoProcessor, max_samples=None, cache_size=3000):
        self.dataset = dataset
        self.max_samples = min(max_samples, len(dataset)) if max_samples else len(dataset)
        self.image_processor = get_image_transform(train=True)   # FIXED
        self.processor = processor
        self.cache_limit = cache_size
        self.cached_images = OrderedDict()

        print(f"Pre-tokenizing {self.max_samples} captions...")
        self.processed_texts = []
        for i in range(self.max_samples):
            item = dataset[i]
            text = item["caption"] + (processor.tokenizer.eos_token or "")
            inputs = processor(text=text, return_tensors="pt", padding=False, truncation=True, max_length=77)
            self.processed_texts.append({
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
            })
        print("Done.")

    def __len__(self): return self.max_samples

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img_path = item["image_path"]

        img = self.cached_images.get(img_path)
        if img is None:
            img = Image.open(img_path).convert("RGB")
            self.cached_images[img_path] = img
            if len(self.cached_images) > self.cache_limit:
                self.cached_images.popitem(last=False)

        pixel_values = self.image_processor(img)
        text = self.processed_texts[idx]

        return {
            "pixel_values": pixel_values,
            "input_ids": text["input_ids"].long(),
            "attention_mask": text["attention_mask"].long(),
            "labels": text["input_ids"].long(),  # dummy
        }



class CLIP(nn.Module):
    def __init__(
        self, vision_encoder: nn.Module, text_encoder: nn.Module, proj_dim: int = 768, temperature: float = 0.07
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        # TODO: implement the rest components
        #self.temperature = nn.Parameter(torch.tensor(0.0))
        #self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / temperature)))

        vhid = vision_encoder.config.hidden_size
        thid = text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vhid, proj_dim, bias=False)
        self.text_proj = nn.Linear(thid, proj_dim, bias=False)

        nn.init.normal_(self.vision_proj.weight, std=vhid ** -0.5)
        nn.init.normal_(self.text_proj.weight, std=thid ** -0.5)
        

    def encode_image(self, pixel_values):
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if USE_BF16 else contextlib.nullcontext()
        with ctx:
            out = self.vision_encoder(pixel_values=pixel_values)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                feat = out.pooler_output
            else:
                feat = out.last_hidden_state[:, 0]
            feat = self.vision_proj(feat)
        return F.normalize(feat, dim=-1)
        #return self.vision_encoder(image)

    def encode_text(self, input_ids, attention_mask):
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if USE_BF16 else contextlib.nullcontext()
        with ctx:
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            last = out.last_hidden_state
            eos_pos = attention_mask.sum(1) - 1
            eos_pos = eos_pos.clamp(min=0)
            batch_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
            feat = last[batch_idx, eos_pos]
            feat = self.text_proj(feat)
        return F.normalize(feat, dim=-1)
        #return self.text_encoder(text)

    def save_pretrained(self, save_directory: str, **kwargs):
        """Customize save method, save additional parameters"""

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
     
        additional_state = {k: v.cpu() for k, v in self.named_parameters() if "vision_encoder." not in k and "text_encoder." not in k}
        torch.save(additional_state, save_directory / "additional_weights.pt")

    def load_pretrained(self, load_directory: str, **kwargs):
        """Customize load method, load projection additional parameters"""

        additional = Path(load_directory) / "additional_weights.pt"
        if additional.exists():
            sd = torch.load(additional, map_location="cpu")
            for k, v in sd.items():
                if k in self.state_dict():
                    self.state_dict()[k].copy_(v)

    def set_trainable_parameters(self):
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                # Keep backbone frozen if LoRA is applied to it
                param.requires_grad = False
            else:
                # Ensure projection layers and logit_scale are trainable
                param.requires_grad = True
    def gradient_checkpointing_enable(self, **kwargs):
        """
        Enable gradient checkpointing for the vision and text backbones.
        (You don't need to touch this method)
        """
        self.vision_encoder.gradient_checkpointing_enable(**kwargs)
        self.text_encoder.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        """
        Enable input require grads for the vision and text backbones.
        (You don't need to touch this method)
        """

        # Reference: https://discuss.huggingface.co/t/peft-lora-gpt-neox-backward-pass-failing/35641
        def make_inputs_require_grads(module, input, output):  # noqa: A002
            output.requires_grad_(True)

        self.vision_encoder.embeddings.register_forward_hook(make_inputs_require_grads)
        self.text_encoder.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the CLIP model.
        Args:
            pixel_values: The pixel values of the image.
            input_ids: The input ids of the text.
            attention_mask: The attention mask of the text.
            labels: The labels for the text features.
            (NOTE: you don't need to use the variable `labels`, this is just for compatibility with the Trainer class)
            (Hint: refer to returned values of the __getitem__ method in the CaptionDatasetForTraining class)
        Returns:
            TODO: think about the what values should be returned
        """
        mg_feat = self.encode_image(pixel_values)
        txt_feat = self.encode_text(input_ids, attention_mask)

        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = logit_scale * img_feat @ txt_feat.t()

        if self.training:
            target = torch.arange(img_feat.shape[0], device=img_feat.device)
            loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))
            return {"loss": loss}

        return img_feat, txt_feat, logit_scale

def compute_clip_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    labels: torch.Tensor,  
    num_items_in_batch: int | None = None, 
) -> torch.Tensor:
    """
    Compute the loss for the CLIP model.
    Args:
        outputs: A tuple containing the outputs of CLIP.forward().
        labels: The labels for the text features.
        (NOTE: you don't need to use the variable `labels`, this is just for compatibility with the Trainer class)
        num_items_in_batch: The number of items in the batch.
        (NOTE: you don't need to use the variable `num_items_in_batch`, this is just for compatibility with Trainer)
    Returns:
        The loss for the CLIP model.
    """
    image_features, text_features, logit_scale = outputs
    
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    labels = torch.arange(image_features.shape[0], device=image_features.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


def get_target_modules_for_lora(model: nn.Module) -> list[str]:
    
    target_modules = []
    target_keywords = ["q_proj", "k_proj", "v_proj", "o_proj", "dense", "proj", "output"]
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if ("vision_encoder" in name or "text_encoder" in name) and ("projection" not in name):
                if any(k in name for k in target_keywords):
                    target_modules.append(name)
    if not target_modules:

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and ("vision_encoder" in name or "text_encoder" in name):
                target_modules.append(name)
    target_modules = list(set(target_modules))
    print(f"Found {len(target_modules)} LoRA target modules (example 5):", target_modules[:5])
    return target_modules

def get_peft_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,           
        r=64,                                  
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules="all-linear",           
        bias="none",
        modules_to_save=["vision_proj", "text_proj", "logit_scale"], 
    )

def train(
    data_dir: str = "data",
    output_dir: str = "clip_model",
    num_train_epochs: float = 2.0,
    per_device_train_batch_size: int = 64,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-4,           
    warmup_steps: int = 500,
    num_workers: int = 12,
    max_samples: int = 350000,
):
    
    start_time = time.time()
    print("Starting training")

    vlm = BaseVLM()
    model = CLIP(vlm.model.model.vision_model, vlm.model.model.text_model).to(device)
    if USE_BF16:
        model = model.bfloat16()

    model = get_peft_model(model, get_peft_config())
    model.print_trainable_parameters()

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    dataset = CaptionDataset("train", Path(data_dir))
    train_dataset = OptimizedCaptionDatasetForTraining(dataset, processor, max_samples=max_samples)

    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=0.1,
        bf16=USE_BF16,
        tf32=True,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=2,
        report_to="tensorboard",
        dataloader_num_workers=num_workers,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        fp16=False,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, data_collator=None)

    print(f"Dataset size: {len(train_dataset)}")
    print(f"Effective batch size (per step): {per_device_train_batch_size} * {gradient_accumulation_steps}")
    trainer.train()

    trainer.save_model(str(output_dir))
    model.save_pretrained(str(output_dir))
    print("Training complete")

    


def demo_train():
    train(
        train_dataset_name="train_demo",
        output_dir="demo_clip",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        num_workers=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-8,
    )


def test(ckpt_path: str, val_dataset: str = "valid_grader"):
    import tqdm

    testset = MultiChoiceQADataset(val_dataset)

    clip = load(ckpt_path)
    clip = clip.model.to(device)

    image_processor = tv.transforms.Compose(
        [
            tv.transforms.Resize(192),
            tv.transforms.CenterCrop(192),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    correct_count = 0
    total_count = 0

    for pair in tqdm.tqdm(testset):
        image = Image.open(pair["image_path"]).convert("RGB")
        pixel_values = image_processor(image).unsqueeze(0).to(device).bfloat16()
        text_inputs = processor(
            text=[s + processor.tokenizer.eos_token for s in pair["candidates"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = text_inputs["input_ids"].long().to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        vision_feature, text_feature, _ = clip(pixel_values, input_ids, attention_mask)
        prediction = torch.matmul(vision_feature, text_feature.T).argmax(dim=-1)
        if prediction == pair["correct_index"]:
            correct_count += 1
        total_count += 1

    print(f"Accuracy: {correct_count / total_count}")


def main():
    from fire import Fire

    Fire({"train": train, "test": test})


if __name__ == "__main__":
    main()
