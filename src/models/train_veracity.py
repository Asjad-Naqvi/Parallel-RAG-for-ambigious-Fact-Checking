#!/usr/bin/env python3
"""
Train AMBIFC veracity classifier.

Experiment 1A:
    Original AMBIFC full-text SINGLE model.
    Loss: Cross-Entropy over hard veracity labels.

Experiment 1B:
    Original AMBIFC full-text DISTILL model.
    Loss: Soft Cross-Entropy over human soft-label distributions.

Input:
    data/processed/train.jsonl
    data/processed/dev.jsonl
    data/processed/test.jsonl

Each row should contain:
    id
    claim
    passage
    entity
    section_title
    soft_label
    hard_label
    category

Label order:
    0 = refuting
    1 = neutral
    2 = supporting
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_all_metrics, format_metrics


LABEL_NAMES = ["refuting", "neutral", "supporting"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path, max_rows: int | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            rows.append(json.loads(line))

            if max_rows is not None and len(rows) >= max_rows:
                break

    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_evidence_text(row: Dict[str, Any], use_rag_context: bool = False) -> str:
    """
    Original AMBIFC paper-style input:

        claim [SEP] evidence @ entity @ section_title [SEP]

    The tokenizer creates the [SEP] separation automatically because
    we pass claim as text A and evidence string as text B.
    """

    if use_rag_context and row.get("rag_context"):
        passage = str(row.get("rag_context", ""))
    else:
        passage = str(row.get("passage", row.get("original_passage", "")))

    entity = str(row.get("entity", ""))
    section_title = str(row.get("section_title", ""))

    extras = []

    if entity.strip():
        extras.append(entity.strip())

    if section_title.strip():
        extras.append(section_title.strip())

    if extras:
        return passage.strip() + " @ " + " @ ".join(extras)

    return passage.strip()


class AmbifcVeracityDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], use_rag_context: bool = False):
        self.rows = rows
        self.use_rag_context = use_rag_context

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]

        claim = str(row.get("claim", "")).strip()
        evidence = build_evidence_text(row, use_rag_context=self.use_rag_context)

        soft_label = row.get("soft_label")

        if not isinstance(soft_label, list) or len(soft_label) != 3:
            raise ValueError(f"Invalid soft_label in row id={row.get('id')}")

        hard_label = int(row.get("hard_label", int(np.argmax(soft_label))))

        return {
            "id": row.get("id", str(index)),
            "claim": claim,
            "evidence": evidence,
            "soft_label": torch.tensor(soft_label, dtype=torch.float32),
            "hard_label": torch.tensor(hard_label, dtype=torch.long),
            "category": row.get("category", ""),
            "hard_label_name": row.get("hard_label_name", LABEL_NAMES[hard_label]),
        }


class AmbifcCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        claims = [item["claim"] for item in batch]
        evidences = [item["evidence"] for item in batch]

        encoded = self.tokenizer(
            claims,
            evidences,
            padding=True,
            truncation="only_second",
            max_length=self.max_length,
            return_tensors="pt",
        )

        encoded["soft_labels"] = torch.stack([item["soft_label"] for item in batch])
        encoded["hard_labels"] = torch.stack([item["hard_label"] for item in batch])
        encoded["ids"] = [item["id"] for item in batch]
        encoded["categories"] = [item["category"] for item in batch]
        encoded["gold_label_names"] = [item["hard_label_name"] for item in batch]

        return encoded


def compute_training_loss(
    logits: torch.Tensor,
    soft_labels: torch.Tensor,
    hard_labels: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    """
    ce:
        Standard cross-entropy over hard labels.
        This is used for the original-paper SINGLE model.

    soft_ce:
        Soft cross-entropy over human annotation distributions.
        This is used for original-paper DISTILL models.
    """

    if loss_type == "ce":
        return F.cross_entropy(logits, hard_labels)

    if loss_type == "soft_ce":
        log_probs = F.log_softmax(logits, dim=-1)
        return -(soft_labels * log_probs).sum(dim=-1).mean()

    raise ValueError(f"Unknown loss type: {loss_type}")


def get_autocast_settings(args: argparse.Namespace, device: torch.device) -> Tuple[bool, torch.dtype]:
    if device.type != "cuda":
        return False, torch.float32

    if args.bf16:
        return True, torch.bfloat16

    if args.fp16:
        return True, torch.float16

    return False, torch.float32


@torch.no_grad()
def evaluate(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
    loss_type: str,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    model.eval()

    total_loss = 0.0
    total_examples = 0

    all_gold_probs = []
    all_pred_probs = []

    prediction_rows = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        ids = batch.pop("ids")
        categories = batch.pop("categories")
        gold_label_names = batch.pop("gold_label_names")

        soft_labels = batch.pop("soft_labels").to(device)
        hard_labels = batch.pop("hard_labels").to(device)

        model_inputs = {key: value.to(device) for key, value in batch.items()}

        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            outputs = model(**model_inputs)
            logits = outputs.logits
            loss = compute_training_loss(logits, soft_labels, hard_labels, loss_type)

        probs = torch.softmax(logits, dim=-1)

        batch_size = soft_labels.size(0)

        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

        gold_np = soft_labels.detach().cpu().numpy()
        pred_np = probs.detach().cpu().numpy()

        all_gold_probs.append(gold_np)
        all_pred_probs.append(pred_np)

        for i in range(batch_size):
            gold_label = int(gold_np[i].argmax())
            pred_label = int(pred_np[i].argmax())

            prediction_rows.append(
                {
                    "id": ids[i],
                    "category": categories[i],
                    "gold_label": gold_label,
                    "gold_label_name": LABEL_NAMES[gold_label],
                    "gold_label_name_original": gold_label_names[i],
                    "pred_label": pred_label,
                    "pred_label_name": LABEL_NAMES[pred_label],
                    "gold_probs": gold_np[i].tolist(),
                    "pred_probs": pred_np[i].tolist(),
                }
            )

    gold_probs = np.concatenate(all_gold_probs, axis=0)
    pred_probs = np.concatenate(all_pred_probs, axis=0)

    metrics = compute_all_metrics(gold_probs, pred_probs)
    metrics["loss"] = total_loss / max(total_examples, 1)

    gold_hard = gold_probs.argmax(axis=1)
    pred_hard = pred_probs.argmax(axis=1)

    metrics["hard_accuracy"] = float(accuracy_score(gold_hard, pred_hard))
    metrics["hard_macro_f1"] = float(
        f1_score(
            gold_hard,
            pred_hard,
            average="macro",
            labels=[0, 1, 2],
            zero_division=0,
        )
    )

    return metrics, prediction_rows


def is_better_metric(
    metric_name: str,
    current_value: float,
    best_value: float | None,
) -> bool:
    if best_value is None:
        return True

    lower_is_better = {"loss", "entce", "kl"}

    if metric_name in lower_is_better:
        return current_value < best_value

    return current_value > best_value


def print_train_config(args: argparse.Namespace, device: torch.device, total_steps: int, warmup_steps: int) -> None:
    print("\n=== Training Configuration ===")
    print(f"Model:                 {args.model_name}")
    print(f"Loss:                  {args.loss}")
    print(f"Best metric:           {args.best_metric}")
    print(f"Max length:            {args.max_length}")
    print(f"Learning rate:         {args.lr}")
    print(f"Epochs:                {args.epochs}")
    print(f"Batch size/device:     {args.per_device_batch_size}")
    print(f"Gradient accumulation: {args.grad_accum_steps}")
    print(f"Effective batch size:  {args.per_device_batch_size * args.grad_accum_steps}")
    print(f"Eval batch size:       {args.eval_batch_size}")
    print(f"Weight decay:          {args.weight_decay}")
    print(f"Warmup ratio:          {args.warmup_ratio}")
    print(f"Warmup steps:          {warmup_steps}")
    print(f"Total update steps:    {total_steps}")
    print(f"Max grad norm:         {args.max_grad_norm}")
    print(f"FP16 autocast:         {args.fp16}")
    print(f"BF16 autocast:         {args.bf16}")
    print(f"Gradient checkpoint:   {args.gradient_checkpointing}")
    print(f"Device:                {device}")


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(Path(args.train), args.max_train_rows)
    dev_rows = read_jsonl(Path(args.dev), args.max_dev_rows)
    test_rows = read_jsonl(Path(args.test), args.max_test_rows)

    print("\n=== Data ===")
    print(f"Train rows: {len(train_rows)}")
    print(f"Dev rows:   {len(dev_rows)}")
    print(f"Test rows:  {len(test_rows)}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
    )

    model.config.num_labels = 3
    model.config.id2label = {i: name for i, name in enumerate(LABEL_NAMES)}
    model.config.label2id = {name: i for i, name in enumerate(LABEL_NAMES)}

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    model.to(device)

    # Keep actual model weights in FP32. Autocast handles activation precision.
    model.float()

    autocast_enabled, autocast_dtype = get_autocast_settings(args, device)

    train_dataset = AmbifcVeracityDataset(train_rows, use_rag_context=args.use_rag_context)
    dev_dataset = AmbifcVeracityDataset(dev_rows, use_rag_context=args.use_rag_context)
    test_dataset = AmbifcVeracityDataset(test_rows, use_rag_context=args.use_rag_context)

    collator = AmbifcCollator(tokenizer=tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_training_steps = update_steps_per_epoch * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    config_to_save = vars(args).copy()
    config_to_save["device"] = str(device)
    config_to_save["autocast_enabled"] = autocast_enabled
    config_to_save["autocast_dtype"] = str(autocast_dtype)
    config_to_save["total_training_steps"] = total_training_steps
    config_to_save["warmup_steps"] = warmup_steps

    save_json(output_dir / "training_args.json", config_to_save)

    print_train_config(args, device, total_training_steps, warmup_steps)

    best_score: float | None = None
    best_epoch = -1
    history = []

    global_step = 0

    print("\n=== Training ===")

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_loss = 0.0
        epoch_examples = 0

        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(progress, start=1):
            batch.pop("ids")
            batch.pop("categories")
            batch.pop("gold_label_names")

            soft_labels = batch.pop("soft_labels").to(device)
            hard_labels = batch.pop("hard_labels").to(device)

            model_inputs = {key: value.to(device) for key, value in batch.items()}

            with torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                outputs = model(**model_inputs)
                logits = outputs.logits
                loss = compute_training_loss(
                    logits=logits,
                    soft_labels=soft_labels,
                    hard_labels=hard_labels,
                    loss_type=args.loss,
                )
                loss_to_backprop = loss / args.grad_accum_steps

            loss_to_backprop.backward()

            batch_size = hard_labels.size(0)
            epoch_loss += float(loss.item()) * batch_size
            epoch_examples += batch_size

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm,
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

            running_loss = epoch_loss / max(epoch_examples, 1)
            progress.set_postfix(loss=f"{running_loss:.4f}")

        train_loss = epoch_loss / max(epoch_examples, 1)

        print(f"\nEpoch {epoch} train loss: {train_loss:.6f}")

        dev_metrics, dev_predictions = evaluate(
            model=model,
            dataloader=dev_loader,
            device=device,
            loss_type=args.loss,
            autocast_enabled=autocast_enabled,
            autocast_dtype=autocast_dtype,
        )

        print(f"\nEpoch {epoch} dev metrics")
        print(format_metrics(dev_metrics))
        print(f"{'loss':15s}: {dev_metrics['loss']:.4f}")
        print(f"{'hard_accuracy':15s}: {dev_metrics['hard_accuracy']:.4f}")
        print(f"{'hard_macro_f1':15s}: {dev_metrics['hard_macro_f1']:.4f}")

        if args.best_metric not in dev_metrics:
            raise ValueError(
                f"best_metric={args.best_metric} not found in dev metrics. "
                f"Available metrics: {sorted(dev_metrics.keys())}"
            )

        current_score = float(dev_metrics[args.best_metric])

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_metrics": dev_metrics,
        }

        history.append(record)
        save_json(output_dir / "train_history.json", {"history": history})

        if is_better_metric(args.best_metric, current_score, best_score):
            best_score = current_score
            best_epoch = epoch

            print(
                f"\nNew best model at epoch {epoch}: "
                f"{args.best_metric}={current_score:.4f}"
            )

            best_dir = output_dir / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

            write_jsonl(output_dir / "dev_predictions.jsonl", dev_predictions)
            save_json(output_dir / "best_dev_metrics.json", dev_metrics)

    print("\n=== Loading best model for test evaluation ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Best dev {args.best_metric}: {best_score:.4f}")

    best_model_dir = output_dir / "best_model"

    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    model.to(device)
    model.float()

    test_metrics, test_predictions = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        loss_type=args.loss,
        autocast_enabled=autocast_enabled,
        autocast_dtype=autocast_dtype,
    )

    print("\n=== Test metrics ===")
    print(format_metrics(test_metrics))
    print(f"{'loss':15s}: {test_metrics['loss']:.4f}")
    print(f"{'hard_accuracy':15s}: {test_metrics['hard_accuracy']:.4f}")
    print(f"{'hard_macro_f1':15s}: {test_metrics['hard_macro_f1']:.4f}")

    write_jsonl(output_dir / "test_predictions.jsonl", test_predictions)
    save_json(output_dir / "test_metrics.json", test_metrics)

    final_summary = {
        "best_epoch": best_epoch,
        "best_dev_metric": args.best_metric,
        "best_dev_score": best_score,
        "test_metrics": test_metrics,
    }

    save_json(output_dir / "final_summary.json", final_summary)

    print(f"\nSaved outputs to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AMBIFC veracity classifier with DeBERTaV3-large."
    )

    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--dev", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)

    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/deberta-v3-large",
    )

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["ce", "soft_ce"],
        help="ce = hard-label cross entropy, soft_ce = soft-label cross entropy",
    )

    parser.add_argument(
        "--best_metric",
        type=str,
        default="accuracy",
        help="Metric used to save the best model. For SINGLE use accuracy. For DISTILL use loss or distcs.",
    )

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=6e-6)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)

    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 autocast. Model weights remain FP32.",
    )

    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 autocast. Do not enable with fp16.",
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Reduce activation memory by recomputing intermediate activations.",
    )

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use_rag_context", action="store_true")

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_train_rows", type=int, default=None)
    parser.add_argument("--max_dev_rows", type=int, default=None)
    parser.add_argument("--max_test_rows", type=int, default=None)

    args = parser.parse_args()

    if args.fp16 and args.bf16:
        raise ValueError("Use either --fp16 or --bf16, not both.")

    return args


if __name__ == "__main__":
    train(parse_args())