import os
from pathlib import Path
from typing import Any, Optional, Tuple, Union
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Trainer, TrainingArguments

from .base_vlm import BaseVLM
from .data import CaptionDataset, MultiChoiceQADataset



def get_optimal_device():
    """Get the best available device - CUDA or CPU only"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # CUDA optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        return device
    else:
        # Use CPU only - NO MPS
        device = torch.device("cpu")
        torch.set_num_threads(max(1, os.cpu_count() // 2))
        print(f"Using CPU device with {torch.get_num_threads()} threads")
        print("   MPS disabled for stability")
        return device


# Global device
device = get_optimal_device()


def get_optimal_dtype():
    """Get optimal dtype for current device"""
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    else:
        # CPU uses float32
        return torch.float32


def is_cuda():
    return device.type == "cuda"


# Initialize processor
try:
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
except Exception as e:
    print(f"Warning: Could not load processor: {e}")
    processor = None

def load(model_name: str = "clip_model"):
    from pathlib import Path
    from peft import PeftModel

    model_path = Path(__file__).parent / model_name

    vlm = BaseVLM()
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model

    # Move encoders to device
    optimal_dtype = get_optimal_dtype()
    vision_encoder = vision_encoder.to(device, dtype=optimal_dtype)
    text_encoder = text_encoder.to(device, dtype=optimal_dtype)

    clip = CLIP(vision_encoder, text_encoder)

    try:
        if is_cuda():
            clip = PeftModel.from_pretrained(
                clip,
                model_path,
                device_map="auto",
                torch_dtype=optimal_dtype
            )
        else:
            clip = PeftModel.from_pretrained(clip, model_path, device_map=None)
            clip = clip.to(device, dtype=optimal_dtype)
    except Exception as e:
        print(f"Error loading PeftModel: {e}")
        try:
            state_dict = torch.load(model_path / "adapter_model.bin", map_location=device)
            clip.load_state_dict(state_dict, strict=False)
        except Exception as e2:
            print(f"Error loading state dict: {e2}")

    clip = clip.to(device, dtype=optimal_dtype)
    clip.model.load_pretrained(model_path)
    clip.model.eval()


    return clip

def clip_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if processor is None:
        raise ValueError("Processor not initialized")

    max_length = max(f["input_ids"].shape[0] for f in features)
    batch_size = len(features)

    input_ids = torch.full((batch_size, max_length), processor.tokenizer.eos_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)

    for i, f in enumerate(features):
        seq_len = f["input_ids"].shape[0]
        input_ids[i, :seq_len] = f["input_ids"]
        attention_mask[i, :seq_len] = f["attention_mask"]
        labels[i, :seq_len] = f["labels"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels,
    }

class CaptionDatasetForTraining(Dataset):
    def __init__(self, dataset, processor: AutoProcessor):
        self.dataset = dataset
        self.processor = processor

        self.image_processor = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            tv.transforms.RandomHorizontalFlip(0.5),
            tv.transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
            tv.transforms.RandAugment(num_ops=2, magnitude=7),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._preprocess_text()

    def _preprocess_text(self):
        self.text_cache = []
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            text = item["caption"] + self.processor.tokenizer.eos_token
            text_inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            self.text_cache.append({
                "input_ids": text_inputs["input_ids"].squeeze(0).long(),
                "attention_mask": text_inputs["attention_mask"].squeeze(0)
            })

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.image_processor(image)
        text_inputs = self.text_cache[idx]
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "labels": text_inputs["input_ids"],
        }

class CLIP(nn.Module):
    def __init__(
            self,
            vision_encoder: nn.Module,
            text_encoder: nn.Module,
            proj_dim: int = 768,
            temperature: float = 0.07,
            use_learnable_temperature: bool = True,
            dropout_rate: float = 0.1
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.proj_dim = proj_dim

        vision_feat_dim = self._get_encoder_dim(vision_encoder, 'vision')
        text_feat_dim = self._get_encoder_dim(text_encoder, 'text')

        self.vision_proj = nn.Sequential(
            nn.Linear(vision_feat_dim, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_feat_dim, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim)
        )

        self._init_weights(self.vision_proj)
        self._init_weights(self.text_proj)

        if use_learnable_temperature:
            self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / temperature)))
        else:
            self.register_buffer('logit_scale', torch.tensor(1.0 / temperature))

    def _get_encoder_dim(self, encoder, encoder_type):
        if hasattr(encoder, 'config'):
            if hasattr(encoder.config, 'hidden_size'):
                return encoder.config.hidden_size
            elif hasattr(encoder.config, 'projection_dim'):
                return encoder.config.projection_dim
        if encoder_type == 'vision':
            if hasattr(encoder, 'num_features'):
                return encoder.num_features
            elif hasattr(encoder, 'head'):
                return encoder.head.in_features
            elif hasattr(encoder, 'fc'):
                return encoder.fc.in_features
        else:
            if hasattr(encoder, 'get_input_embeddings'):
                return encoder.get_input_embeddings().embedding_dim
        return 512

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(next(self.vision_encoder.parameters()).device,
                         next(self.vision_encoder.parameters()).dtype)
        features = self.vision_encoder(image)
        return self._process_features(features, 'vision')

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        input_ids = input_ids.to(next(self.text_encoder.parameters()).device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(next(self.text_encoder.parameters()).device)
        if hasattr(self.text_encoder, 'config'):
            self.text_encoder.config.use_cache = False
        features = self.text_encoder(input_ids, attention_mask=attention_mask)
        return self._process_features(features, 'text')

    def _process_features(self, features, encoder_type):
        if isinstance(features, (tuple, list)):
            features = features[0]
        if hasattr(features, 'last_hidden_state'):
            last_hidden = features.last_hidden_state
            if attention_mask := getattr(features, 'attention_mask', None):
                mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                features = torch.sum(last_hidden * mask, dim=1) / torch.sum(mask, dim=1)
            else:
                features = last_hidden.mean(dim=1)
        elif hasattr(features, 'pooler_output'):
            features = features.pooler_output
        elif encoder_type == 'vision' and features.dim() > 2:
            if features.dim() == 4:
                features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
            else:
                features = features.mean(dim=1)
        return features


    def forward(
            self,
            pixel_values: torch.Tensor,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        pixel_values = pixel_values.to(next(self.parameters()).device,
                                       next(self.parameters()).dtype)
        input_ids = input_ids.to(next(self.parameters()).device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(next(self.parameters()).device)

        batch_size = pixel_values.shape[0]

        image_features = self.encode_image(pixel_values)
        text_features = self.encode_text(input_ids, attention_mask)

        image_embeddings = self.vision_proj(image_features)
        text_embeddings = self.text_proj(text_features)

        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1, eps=1e-8)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1, eps=1e-8)

        logit_scale = self.logit_scale.exp().clamp(max=100, min=1e-4)

        logits_per_image = logit_scale * torch.matmul(image_embeddings, text_embeddings.t())
        logits_per_text = logits_per_image.t()

        if self.training:
            targets = torch.arange(batch_size, device=pixel_values.device)
            loss_i = F.cross_entropy(logits_per_image, targets, label_smoothing=0.1)
            loss_t = F.cross_entropy(logits_per_text, targets, label_smoothing=0.1)
            loss = (loss_i + loss_t) / 2
            return (loss,)
        else:
            return image_embeddings, text_embeddings, logit_scale

    def compute_similarity(self, pixel_values: torch.Tensor, input_ids: torch.Tensor,
                           attention_mask: torch.Tensor = None) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(pixel_values, input_ids, attention_mask)
            if isinstance(outputs, tuple) and len(outputs) == 3:
                image_embeddings, text_embeddings, logit_scale = outputs
            else:
                raise ValueError("Model should be in eval mode for similarity computation")

            logits_per_image = logit_scale * torch.matmul(image_embeddings, text_embeddings.t())
            return logits_per_image

    def get_embedding_dim(self) -> int:
        return self.proj_dim

    def save_pretrained(self, save_directory: str, **kwargs):
        additional_state_dict = {}
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            additional_state_dict[name] = param.data.cpu()

        torch.save(additional_state_dict, Path(save_directory) / "additional_weights.pt")

    def load_pretrained(self, load_directory: str, **kwargs):
        additional_weights_path = Path(load_directory) / "additional_weights.pt"
        if additional_weights_path.exists():
            additional_state_dict = torch.load(additional_weights_path, map_location="cpu")
            for name, param in self.named_parameters():
                if "vision_encoder." in name or "text_encoder." in name:
                    continue
                if name in additional_state_dict:
                    param.data = additional_state_dict[name].to(device)

    def set_trainable_parameters(self, trainable_layers=None, freeze_all=False):
        # FULL UNFREEZING — all params require grad
        for p in self.parameters():
            p.requires_grad = True

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.vision_encoder, 'config'):
            self.vision_encoder.config.use_cache = False
        if hasattr(self.text_encoder, 'config'):
            self.text_encoder.config.use_cache = False

        if hasattr(self.vision_encoder, 'gradient_checkpointing_enable'):
            self.vision_encoder.gradient_checkpointing_enable(**kwargs)
        if hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
            self.text_encoder.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        if hasattr(self.vision_encoder, 'embeddings'):
            self.vision_encoder.embeddings.register_forward_hook(make_inputs_require_grads)
        elif hasattr(self.vision_encoder, 'patch_embed'):
            self.vision_encoder.patch_embed.register_forward_hook(make_inputs_require_grads)
        if hasattr(self.text_encoder, 'get_input_embeddings'):
            self.text_encoder.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
        elif hasattr(self.text_encoder, 'embeddings'):
            self.text_encoder.embeddings.register_forward_hook(make_inputs_require_grads)
def get_target_modules_for_lora(model: nn.Module) -> list[str]:
    target = []
    for name, module in model.named_modules():
        if any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            target.append(name)
    return target

# generative AI is used to optimize the training module 
def train(
        data_dir: str | Path | None = None, 
        output_dir: str = "clip_model",
        num_train_epochs: float = 6.0,
        per_device_train_batch_size: int = None,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        num_workers: int = None,
        force_cpu: bool = False, 
        max_samples: int = 15000, 
):
    global device
    if force_cpu:
        device = torch.device("cpu")
        print("Forcing CPU usage as requested")
        torch.set_num_threads(max(1, os.cpu_count() // 2))

  
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    elif data_dir is None:
        data_dir = Path("data")
    
    print(f"Using data directory: {data_dir.absolute()}")
    
 
    if per_device_train_batch_size is None:
        if is_cuda():
            per_device_train_batch_size = 32
        else:
            per_device_train_batch_size = 4

    if num_workers is None:
        if is_cuda():
            num_workers = 8
        else:
            num_workers = 2

    vlm = BaseVLM()
    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model


    optimal_dtype = get_optimal_dtype()
    vision_encoder = vision_encoder.to(device, dtype=optimal_dtype)
    text_encoder = text_encoder.to(device, dtype=optimal_dtype)

    if hasattr(vision_encoder, 'config'):
        vision_encoder.config.use_cache = False
    if hasattr(text_encoder, 'config'):
        text_encoder.config.use_cache = False

    model = CLIP(
        vision_encoder,
        text_encoder,
        proj_dim=768, 
        use_learnable_temperature=True,
        dropout_rate=0.1
    )


    model = model.to(device, dtype=optimal_dtype)
    for p in model.vision_encoder.parameters():
        p.requires_grad = True

    for p in model.text_encoder.parameters():
        p.requires_grad = True


    for p in model.vision_proj.parameters():
        p.requires_grad = True
    for p in model.text_proj.parameters():
        p.requires_grad = True


    model.logit_scale.requires_grad = True
  
    model.set_trainable_parameters(freeze_all=False)
    model.set_trainable_parameters()

    # LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=96,
        lora_alpha=96,
        lora_dropout=0.05,
        target_modules=get_target_modules_for_lora(model),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    model = model.to(device, dtype=optimal_dtype)
    model.train()

    model.enable_input_require_grads()

    train_dataset = CaptionDataset("train", data_dir)

    if max_samples and len(train_dataset) > max_samples:
        print(f" Downsampling dataset: {len(train_dataset)} → {max_samples} samples")
        indices = torch.randperm(len(train_dataset))[:max_samples].tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_dataset = CaptionDatasetForTraining(train_dataset, processor)


    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=is_cuda(),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        bf16=(is_cuda() and torch.cuda.is_bf16_supported()),
        fp16=(is_cuda() and not torch.cuda.is_bf16_supported()),
        dataloader_pin_memory=is_cuda(),
        dataloader_num_workers=num_workers,
        logging_steps=10,
        save_strategy="epoch",
        save_steps=100,
        save_total_limit=3,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        prediction_loss_only=True,
        no_cuda=not is_cuda(),
        dataloader_drop_last=True,
        eval_strategy="no",
        lr_scheduler_type="cosine",
        optim="adamw_torch",  
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=clip_data_collator,
        compute_metrics=None,
    )

    try:
        print(f"Starting training on {device.type.upper()}...")
        if is_cuda():
            print(f" GPU: {torch.cuda.get_device_name()}")
        else:
            print(f"CPU threads: {torch.get_num_threads()}")

        print(f"Batch size: {per_device_train_batch_size}")
        print(f"Workers: {num_workers}")

        print("Starting training loop...")

        if is_cuda():
            torch.cuda.empty_cache()

        trainer.train()
        trainer.save_model(output_dir)
        model.model.save_pretrained(output_dir)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed with error: {e}")
        traceback.print_exc()
        try:
            trainer.save_model(output_dir)
            model.model.save_pretrained(output_dir)
            print("Model saved despite training error.")
        except Exception as save_error:
            print(f"Could not save model: {save_error}")

    writer.close()
    return model, processor


def test(ckpt_path: str, val_dataset: str = "valid_grader"):
    import tqdm

    testset = MultiChoiceQADataset(val_dataset)
    clip = load(ckpt_path)
    clip = clip.model.to(device)

    # USE GRADER'S PREPROCESSING
    image_processor = tv.transforms.Compose([
        tv.transforms.Resize(192),  # Grader uses 192
        tv.transforms.CenterCrop(192),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Grader's normalization
    ])

    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for pair in tqdm.tqdm(testset):
            image = Image.open(pair["image_path"]).convert("RGB")
            pixel_values = image_processor(image).unsqueeze(0).to(device).bfloat16()  # Use bfloat16
            
            # Use grader's text processing (no max_length limit)
            text_inputs = processor(
                text=[s + processor.tokenizer.eos_token for s in pair["candidates"]],
                return_tensors="pt",
                padding=True,
                truncation=True,
                # REMOVE max_length=128
            )
            input_ids = text_inputs["input_ids"].long().to(device)
            attention_mask = text_inputs["attention_mask"].to(device)

            vision_feature, text_feature, _ = clip(pixel_values, input_ids, attention_mask)
            
            # Use grader's similarity computation
            prediction = torch.matmul(vision_feature, text_feature.T).argmax(dim=-1)
            
            if prediction == pair["correct_index"]:
                correct_count += 1
            total_count += 1

    accuracy = correct_count / total_count
    print(f" Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")

    return accuracy
def main():
    from fire import Fire
    Fire({"train": train, "test": test})


if __name__ == "__main__":
    main()
