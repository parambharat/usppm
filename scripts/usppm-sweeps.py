import wandb
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedGroupKFold
from datasets.utils.logging import set_verbosity_error

set_verbosity_error()
import warnings
import logging
import torch

warnings.simplefilter("ignore")
logging.disable(logging.WARNING)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import datasets

from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.dataset import proba_to_label


def enrich_metadata(data: pd.DataFrame) -> pd.DataFrame:
    context_mapping = {
        "A": "Human Necessities",
        "B": "Operations and Transport",
        "C": "Chemistry and Metallurgy",
        "D": "Textiles",
        "E": "Fixed Constructions",
        "F": "Mechanical Engineering",
        "G": "Physics",
        "H": "Electricity",
        "Y": "Emerging Cross-Sectional Technologies",
    }
    data.loc[:, "context_desc"] = data["context"].apply(lambda x: context_mapping[x[0]])
    data.loc[:, "anchor_length"] = data["anchor"].str.split().map(len)
    data.loc[:, "target_length"] = data["target"].str.split().map(len)
    return data


def log_col_stats(df: pd.DataFrame, col: str, stage: str):
    stats_df = (
        pd.DataFrame(df[col].value_counts())
        .reset_index()
        .rename({"index": col, col: "count"}, axis=1)
        .sort_values("count", ascending=False)
    )
    table = wandb.Table(dataframe=stats_df)
    return wandb.log({f"{stage}_{col}_counts": table})


def convert_df_to_dataset(data_df, tokenizer, stage="train", problem_type="regression"):
    if stage == "train":
        if problem_type == "single_label_classification":
            data_df.loc[:, "score"] = (
                data_df["score"] * (data_df["score"].nunique() - 1)
            ).astype(int)

        elif problem_type == "multi_label_classification":
            data_df.loc[:, "score"] = (
                data_df["score"] * (data_df["score"].nunique() - 1)
            ).astype(int)
            data_df.loc[:, "score"] = pd.Series(
                levels_from_labelbatch(data_df.score, data_df["score"].nunique())
                .numpy()
                .tolist()
            )

        cols = ["anchor", "target", "context_desc", "score"]
        data_df = data_df[cols].rename({"score": "label"}, axis=1)
    else:
        cols = ["anchor", "target", "context_desc"]
        data_df = data_df[cols]
    data_df["inputs"] = (
        f"{tokenizer.sep_token} "
        + data_df["context_desc"]
        + f" {tokenizer.sep_token} "
        + data_df["anchor"]
        + f" {tokenizer.sep_token} "
        + data_df["target"]
    )

    dataset = datasets.Dataset.from_pandas(data_df, preserve_index=False)
    return dataset


def load_tokenizer_fn(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    def tokenizer_fn(examples):
        return tokenizer(examples["inputs"])

    return tokenizer, tokenizer_fn


def load_and_log_datasets(config):
    train_data = pd.read_csv(config.data_dir + "/train.csv").sample(
        frac=1, random_state=config.random_state
    )
    test_data = pd.read_csv(config.data_dir + "/test.csv").sample(
        frac=1, random_state=config.random_state
    )

    train_data = enrich_metadata(train_data)
    test_data = enrich_metadata(test_data)

    train_table = wandb.Table(dataframe=train_data)
    test_table = wandb.Table(dataframe=test_data)

    wandb.log({"train_dataset": train_table, "test_dataset": test_table})

    for col in train_data.columns:
        if col not in ["id"]:
            log_col_stats(train_data, col, stage="train")

    for col in test_data.columns:
        if col not in ["id"]:
            log_col_stats(test_data, col, stage="test")

    sgkf = StratifiedGroupKFold(
        n_splits=int(1 / config.val_size),
        random_state=config.random_state,
        shuffle=True,
    )
    train_index, val_index = next(
        sgkf.split(train_data, train_data["score"] * 100, train_data.anchor)
    )
    tokenizer, tokenizer_fn = load_tokenizer_fn(config)

    train_dataset = convert_df_to_dataset(
        train_data.copy(), tokenizer, stage="train", problem_type=config.problem_type
    )
    test_dataset = convert_df_to_dataset(
        test_data.copy(), tokenizer, stage="test", problem_type=config.problem_type
    )

    experiment_datasets = datasets.DatasetDict(
        {
            "train": train_dataset.select(train_index),
            "validation": train_dataset.select(val_index),
            "test": test_dataset,
        }
    )

    dropped_cols = [
        col
        for col in ["inputs", "anchor", "target", "context_desc", "label"]
        if not col == "label"
    ]

    processed_datasets = experiment_datasets.map(
        tokenizer_fn, batched=True, remove_columns=dropped_cols, num_proc=10,
    )
    return processed_datasets, tokenizer


def pearson_corr_reg(eval_pred):
    predictions = eval_pred.predictions.flatten()
    labels = eval_pred.label_ids
    return {"pearson": np.corrcoef(labels, predictions)[0][1]}


def pearson_corr_clf(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1).flatten().astype(float)
    labels = eval_pred.label_ids.astype(float)
    return {"pearson": np.corrcoef(predictions, labels)[0][1]}


def pearson_corr_ord(eval_pred):
    predictions = proba_to_label(torch.Tensor(expit(eval_pred.predictions))).numpy()
    labels = eval_pred.label_ids
    labels = (labels == 0).argmax(axis=1)
    return {"pearson": np.corrcoef(predictions, labels)[0][1]}


class CoralTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = coral_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


def get_trainer(config, tokenizer, dataset):
    training_args = TrainingArguments(
        config.output_dir,
        learning_rate=config.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        report_to="wandb",
        fp16=True,
        evaluation_strategy="steps",
        logging_strategy="steps",
        eval_accumulation_steps=10,
        logging_steps=100,
    )

    if config.problem_type == "regression":
        config.num_labels = 1
        metric_fn = pearson_corr_reg

    elif config.problem_type == "single_label_classification":
        config.num_labels = 5
        metric_fn = pearson_corr_clf

    elif config.problem_type == "multi_label_classification":
        config.num_labels = 4
        metric_fn = pearson_corr_ord

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path,
        num_labels=config.num_labels,
        problem_type=config.problem_type,
    )
    if config.problem_type == "multi_label_classification":
        trainer = CoralTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            compute_metrics=metric_fn,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            compute_metrics=metric_fn,
        )
    return trainer


defaults = {
    "batch_size": 64,
    "data_dir": "../inputs",
    "entity": "parambharat",
    "epochs": 5,
    "learning_rate": 0.000003,
    "log_dir": "../logs",
    "lr_scheduler_type": "cosine",
    "model_name_or_path": "anferico/bert-for-patents",
    "output_dir": "../outputs",
    "problem_type": "regression",
    "project_name": "usppm",
    "random_state": 42,
    "val_size": 0.25,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
}

sweep_config = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "eval/pearson"},
    "parameters": {
        "batch_size": {
            "distribution": "q_log_uniform_values",
            "max": 64,
            "min": 32,
            "q": 4,
        },
        "data_dir": {"value": "../inputs"},
        "dropout": {"values": [0.3, 0.4, 0.5]},
        "epochs": {"values": [4, 6, 8, 10]},
        "learning_rate": {"distribution": "uniform", "max": 5e-4, "min": 8e-6},
        "log_dir": {"value": "../logs"},
        "model_name_or_path": {
            "values": [
                "microsoft/deberta-v3-small",
                "microsoft/deberta-base",
                "AI-Growth-Lab/PatentSBERTa",
                "anferico/bert-for-patents",
            ]
        },
        "output_dir": {"value": "../outputs"},
        "problem_type": {
            "values": [
                "regression",
                "single_label_classification",
                #         "multi_label_classification"
            ]
        },
        "random_state": {"value": 42},
        "val_size": {"value": 0.25},
        "warmup_ratio": {"value": 0.1},
        "weight_decay": {"values": [0.01, 0.03, 0.001, 0.003]},
    },
}


def train_fn(config=defaults):

    with wandb.init(project=defaults["project_name"], config=config) as run:
        config = run.config
        dds, tokenizer = load_and_log_datasets(config)
        trainer = get_trainer(config, tokenizer, dds)
        trainer.train()
        del trainer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project=defaults["project_name"])
    wandb.agent(sweep_id, train_fn, count=25)
