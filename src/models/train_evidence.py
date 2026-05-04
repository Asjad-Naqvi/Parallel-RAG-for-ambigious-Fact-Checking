#!/usr/bin/env python3
"""
Train AMBIFC evidence selector.

Experiment 2A:
    Binary evidence selection.

Input:
    data/processed/train_evidence.jsonl
    data/processed/dev_evidence.jsonl
    data/processed/test_evidence.jsonl

Each row should contain:
    id
    claim
    sentence
    entity
    section_title
    binary_evidence_label
    soft_label
    evidence_prob
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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


LABEL_NAMES = ["non_evidence", "evidence"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path, max_rows: int | None = None) -> List[Dict[str, Any]]:
    rows = []

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


def build_sentence_text(row: Dict[str, Any]) -> str:
    sentence = str(row.get("sentence", ""))
    entity = str(row.get("entity", ""))
    section_title = str(row.get("section_title", ""))

    extras = []

    if entity.strip():
        extras.append(entity.strip())

    if section_title.strip():
        extras.append(section_title.strip())

    if extras:
        return sentence.strip() + " @ " + " @ ".join(extras)

    return sentence.strip()


class AmbifcEvidenceDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]

        claim = str(row.get("claim", "")).strip()
        sentence_text = build_sentence_text(row)

        label = int(row.get("binary_evidence_label", 0))

        return {
            "id": row.get("id", str(index)),
            "parent_id": row.get("parent_id", ""),
            "claim": claim,
            "sentence_text": sentence_text,
            "label": torch.tensor(label, dtype=torch.long),
            "sentence_id": row.get("sentence_id", ""),
        }


class EvidenceCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        claims = [item["claim"] for item in batch]
        sentences = [item["sentence_text"] for item in batch]

        encoded = self.tokenizer(
            claims,
            sentences,
            padding=True,
            truncation="only_second",
            max_length=self.max_length,
            return_tensors="pt",
        )

        encoded["labels"] = torch.stack([item["label"] for item in batch])
        encoded["ids"] = [item["id"] for item in batch]
        encoded["parent_ids"] = [item["parent_id"] for item in batch]
        encoded["sentence_ids"] = [item["sentence_id"] for item in batch]

        return encoded


def get_autocast_settings(args: argparse.Namespace, device: torch.device):
    if device.type != "cuda":
        return False, torch.float32

    if args.bf16:
        return True, torch.bfloat16

    if args.fp16:
        return True, torch.float16

    return False, torch.float32


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    model.eval()

    total_loss = 0.0
    total_examples = 0

    all_gold = []
    all_pred = []
    all_scores = []

    prediction_rows = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        ids = batch.pop("ids")
        parent_ids = batch.pop("parent_ids")
        sentence_ids = batch.pop("sentence_ids")

        labels = batch.pop("labels").to(device)
        model_inputs = {key: value.to(device) for key, value in batch.items()}

        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            outputs = model(**model_inputs)
            logits = outputs.logits
            loss = F.cross_entropy(logits, labels)

        probs = torch.softmax(logits, dim=-1)
        pred_labels = probs.argmax(dim=-1)
        evidence_scores = probs[:, 1]

        batch_size = labels.size(0)

        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

        gold_np = labels.detach().cpu().numpy()
        pred_np = pred_labels.detach().cpu().numpy()
        score_np = evidence_scores.detach().cpu().numpy()

        all_gold.append(gold_np)
        all_pred.append(pred_np)
        all_scores.append(score_np)

        for i in range(batch_size):
            prediction_rows.append(
                {
                    "id": ids[i],
                    "parent_id": parent_ids[i],
                    "sentence_id": sentence_ids[i],
                    "gold_label": int(gold_np[i]),
                    "pred_label": int(pred_np[i]),
                    "evidence_score": float(score_np[i]),
                }
            )

    gold = np.concatenate(all_gold)
    pred = np.concatenate(all_pred)
    scores = np.concatenate(all_scores)

    precision, recall, f1, _ = precision_recall_fscore_support(
        gold,
        pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    metrics = {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": float(accuracy_score(gold, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate_gold": float(gold.mean()),
        "positive_rate_pred": float(pred.mean()),
        "avg_evidence_score": float(scores.mean()),
    }

    return metrics, prediction_rows


def is_better(current: float, best: float | None, metric_name: str) -> bool:
    if best is None:
        return True

    if metric_name == "loss":
        return current < best

    return current > best


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
        num_labels=2,
    )

    model.config.num_labels = 2
    model.config.id2label = {0: "non_evidence", 1: "evidence"}
    model.config.label2id = {"non_evidence": 0, "evidence": 1}

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    model.to(device)
    model.float()

    autocast_enabled, autocast_dtype = get_autocast_settings(args, device)

    train_dataset = AmbifcEvidenceDataset(train_rows)
    dev_dataset = AmbifcEvidenceDataset(dev_rows)
    test_dataset = AmbifcEvidenceDataset(test_rows)

    collator = EvidenceCollator(tokenizer, args.max_length)

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

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_training_steps = update_steps_per_epoch * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    save_json(
        output_dir / "training_args.json",
        {
            **vars(args),
            "device": str(device),
            "total_training_steps": total_training_steps,
            "warmup_steps": warmup_steps,
            "autocast_enabled": autocast_enabled,
            "autocast_dtype": str(autocast_dtype),
        },
    )

    print("\n=== Training Configuration ===")
    print(f"Model:                 {args.model_name}")
    print(f"Task:                  binary evidence selection")
    print(f"Best metric:           {args.best_metric}")
    print(f"Max length:            {args.max_length}")
    print(f"Learning rate:         {args.lr}")
    print(f"Epochs:                {args.epochs}")
    print(f"Batch size/device:     {args.per_device_batch_size}")
    print(f"Gradient accumulation: {args.grad_accum_steps}")
    print(f"Effective batch size:  {args.per_device_batch_size * args.grad_accum_steps}")
    print(f"Eval batch size:       {args.eval_batch_size}")
    print(f"Warmup steps:          {warmup_steps}")
    print(f"Total update steps:    {total_training_steps}")
    print(f"Device:                {device}")

    best_score = None
    best_epoch = -1
    history = []

    print("\n=== Training ===")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_examples = 0

        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(progress, start=1):
            batch.pop("ids")
            batch.pop("parent_ids")
            batch.pop("sentence_ids")

            labels = batch.pop("labels").to(device)
            model_inputs = {key: value.to(device) for key, value in batch.items()}

            with torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                outputs = model(**model_inputs)
                logits = outputs.logits
                loss = F.cross_entropy(logits, labels)
                loss_to_backprop = loss / args.grad_accum_steps

            loss_to_backprop.backward()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss = total_loss / max(total_examples, 1)
            progress.set_postfix(loss=f"{running_loss:.4f}")

        train_loss = total_loss / max(total_examples, 1)

        print(f"\nEpoch {epoch} train loss: {train_loss:.6f}")

        dev_metrics, dev_predictions = evaluate(
            model=model,
            dataloader=dev_loader,
            device=device,
            autocast_enabled=autocast_enabled,
            autocast_dtype=autocast_dtype,
        )

        print(f"\nEpoch {epoch} dev metrics")
        for key, value in dev_metrics.items():
            print(f"{key:20s}: {value:.4f}")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_metrics": dev_metrics,
            }
        )
        save_json(output_dir / "train_history.json", {"history": history})

        if args.best_metric not in dev_metrics:
            raise ValueError(f"best_metric={args.best_metric} not found in dev metrics.")

        current_score = dev_metrics[args.best_metric]

        if is_better(current_score, best_score, args.best_metric):
            best_score = current_score
            best_epoch = epoch

            print(f"\nNew best model at epoch {epoch}: {args.best_metric}={current_score:.4f}")

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
        autocast_enabled=autocast_enabled,
        autocast_dtype=autocast_dtype,
    )

    print("\n=== Test metrics ===")
    for key, value in test_metrics.items():
        print(f"{key:20s}: {value:.4f}")

    write_jsonl(output_dir / "test_predictions.jsonl", test_predictions)
    save_json(output_dir / "test_metrics.json", test_metrics)

    save_json(
        output_dir / "final_summary.json",
        {
            "best_epoch": best_epoch,
            "best_dev_metric": args.best_metric,
            "best_dev_score": best_score,
            "test_metrics": test_metrics,
        },
    )

    print(f"\nSaved outputs to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AMBIFC binary evidence selector.")

    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--dev", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--best_metric", type=str, default="f1")

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=6e-6)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--per_device_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--grad_accum_steps", type=int, default=1)

    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--cpu", action="store_true")

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