from dataclasses import dataclass, field
import logging
import json
import os

from datasets import load_from_disk, load_dataset, Dataset as HFDataset
import transformers
import torch
import tqdm

from bioagent.training import ModelArguments
from bioagent.inference import load_trained_lora_model
from bioagent.data_tools import encode_chat, parse_chat_output
from bioagent.chemistry_tools import EVALUATOR_BUILDERS


@dataclass
class EvaluationArguments(ModelArguments):
    dataset_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    max_new_tokens: int = field(default=2048, metadata={"help": "Maximum number of new tokens to generate."})
    temperature: float = field(default=1.0, metadata={"help": "Temperature to use for sampling."})
    load_bits: int = field(default=16, metadata={"help": "Quantization bits to use."})
    parser: str = field(default='base', metadata={"help": "Parser for the generated output."})
    evaluator: str = field(default='smiles', metadata={"help": "Evaluator to use for the generated output."})
    cache_dir: str = field(default=None, metadata={"help": "Path to the cache directory."})
    output_dir: str = field(default=None, metadata={"help": "Path to the output file."})
    verbose: bool = field(default=False, metadata={"help": "Print verbose output."})


def _save_rows(rows: list, cache_dir: str, file_name: str = "rows.txt"):
    os.makedirs(cache_dir, exist_ok=True)
    if file_name.endswith(".txt"):
        with open(os.path.join(cache_dir, file_name), "w") as file:
            file.write("\n".join(rows))
    elif file_name.endswith(".json"):
        with open(os.path.join(cache_dir, file_name), "w") as file:
            json.dump(rows, file)
    else:
        raise ValueError(f"Unknown file format: {file_name}")

def _save_score(score: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "score.json"), "w") as file:
        json.dump(score, file)


def _resolve_dataset(path: str) -> HFDataset:
    if os.path.exists(path):
        return load_from_disk(path)
    else:
        return load_dataset(path, split="train", data_files="*.arrow")


def _evaluate(model, tokenizer, dataset, args):
    predictions = []
    references = []
    evaluator = EVALUATOR_BUILDERS[args.evaluator]()

    dataset = tqdm.tqdm(dataset, desc="Evaluating", total=len(dataset))
    for entry in dataset:
        ground_truth = entry['ground_truth']
        encoded_dict = encode_chat(entry, tokenizer, model.modalities)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=encoded_dict["input_ids"].unsqueeze(0).to(model.device),
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                do_sample=True,
                temperature=args.temperature,
                modality_inputs={
                    m.name: [encoded_dict[m.name]] for m in model.modalities
                },
            )

        generated_output = tokenizer.decode(
            output_ids[0, encoded_dict["input_ids"].shape[0]:],
            skip_special_tokens=True,
        ).strip() 

        if args.parser:
            try:
                generated_output = parse_chat_output(generated_output, args.parser)["output"]
            except:
                pass

        if args.verbose:
            print(f"Ground Truth: {ground_truth}")
            print(f"Generated Output: {generated_output}")
            print(f"Score: {evaluator.evaluate([generated_output], [ground_truth])}")

        predictions.append(generated_output)
        references.append(ground_truth)

    if args.cache_dir:
        _save_rows(predictions, args.cache_dir, "predictions.txt")
        _save_rows(references, args.cache_dir, "references.txt")

    return evaluator.evaluate(predictions, references, verbose=True)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = transformers.HfArgumentParser((EvaluationArguments,))
    eval_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    model, tokenizer = load_trained_lora_model(
        model_name_or_path=eval_args.model_name_or_path,
        model_lora_path=eval_args.model_lora_path,
        load_bits=eval_args.load_bits,
    )

    dataset = _resolve_dataset(eval_args.dataset_path) 
    score = _evaluate(model, tokenizer, dataset, eval_args)

    if eval_args.output_dir:
        _save_score(score, eval_args.output_dir)
