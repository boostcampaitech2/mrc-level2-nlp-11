import logging
import os
import sys

from typing import List, Callable, NoReturn, NewType, Any
import dataclasses
from datasets import load_from_disk, Dataset, DatasetDict

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

from transformers import (
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
)

from tokenizers import Tokenizer
from tokenizers.models import WordPiece

from utils_qa import *
from trainer_qa import *

from arguments import ModelArguments, DataTrainingArguments, TrainerArguments
from mrc_reader import *
import wandb

logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainerArguments))
    model_args, data_args, trainer_args = parser.parse_args_into_dataclasses()
    training_args = TrainingArguments(
        output_dir=trainer_args.output_dir,
        learning_rate=trainer_args.lr,
        save_total_limit=10,
        per_device_train_batch_size=trainer_args.train_batch_size,
        per_device_eval_batch_size=trainer_args.eval_batch_size,
        num_train_epochs=trainer_args.epochs,
        logging_steps=trainer_args.logging_steps,
        do_train=True,
        do_eval=True,
        save_strategy="steps",
        report_to="wandb",
        run_name=trainer_args.run_name,
        save_steps=trainer_args.save_steps,
        evaluation_strategy="steps",
        eval_steps=trainer_args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_exact_match",
    )
    print(model_args.model_name_or_path)

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    if data_args.use_preprocess:
        datasets = get_preprocess_dataset()
    else:
        datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # Reader Class 생성
    reader = Reader(model_args=model_args, data_args=data_args, datasets=datasets)

    model, tokenizer = reader.get_model_tokenizer()

    # do_train mrc model 혹은 do_eval mrc model
    if training_args.do_train or training_args.do_eval:
        run_mrc(
            data_args,
            training_args,
            model_args,
            trainer_args,
            datasets,
            tokenizer,
            model,
            reader,
        )


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    trainer_args: TrainerArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
    reader: Reader,
) -> NoReturn:

    # dataset을 전처리합니다.
    # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
    reader.set_column_name(training_args.do_train)

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )
    reader.set_max_seq_length(max_seq_length)

    # Train preprocessing / 전처리를 진행합니다.

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = reader.get_train_dataset()

    # Validation preprocessing

    if training_args.do_eval:
        eval_dataset = reader.get_validation_dataset()

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        max_answer_length=data_args.max_answer_length,
        dataset=datasets,
        answer_column_name=reader.answer_column_name,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        # if last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        # elif os.path.isdir(model_args.model_name_or_path):
        #     checkpoint = model_args.model_name_or_path
        # else:
        #     checkpoint = None
        # train_result = trainer.train(resume_from_checkpoint=checkpoint)
        train_result = trainer.train()

        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # best_model dir에 모델 저장
        if model_args.model_name_or_path.split("_")[0] == "pre":
            model.save_pretrained(trainer_args.best_model_dir)
        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
