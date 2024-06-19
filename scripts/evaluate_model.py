from dataclasses import dataclass, field
import logging
import json
import os

from datasets import load_from_disk, load_dataset, Dataset as HFDataset
import transformers
import torch
import tqdm

from presto.training import ModelArguments
from presto.inference import load_trained_lora_model, load_trained_model
from presto.data_tools import encode_chat, parse_chat_output, encode_interleaved_data
from presto.chemistry_tools import EVALUATOR_BUILDERS


@dataclass
class EvaluationArguments(ModelArguments):
    dataset_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lora_enable: bool = field(default=True, metadata={"help": "Enable LoRA."})
    max_new_tokens: int = field(default=2048, metadata={"help": "Maximum number of new tokens to generate."})
    temperature: float = field(default=0.2, metadata={"help": "Temperature to use for sampling."})
    top_k: int = field(default=50, metadata={"help": "Top k to use for sampling."})
    top_p: float = field(default=0.8, metadata={"help": "Top p to use for sampling."})
    do_sample: bool = field(default=True, metadata={"help": "Whether to sample from the output distribution."})
    load_bits: int = field(default=16, metadata={"help": "Quantization bits to use."})
    parser: str = field(default='base', metadata={"help": "Parser for the generated output."})
    evaluator: str = field(default='smiles', metadata={"help": "Evaluator to use for the generated output."})
    cache_dir: str = field(default=None, metadata={"help": "Path to the cache directory."})
    output_dir: str = field(default=None, metadata={"help": "Path to the output file."})
    is_icl: bool = field(default=False, metadata={"help": "Whether ICL testing is enabled."})
    verbose: bool = field(default=False, metadata={"help": "Print verbose output."})


def _save_rows(rows: list, cache_dir: str, file_name: str = "rows.txt"):
    rows = [str(row) for row in rows]
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
        try:
            return load_from_disk(path)
        except:
            return load_dataset(path, split="test")
    else:
        return load_dataset(path, split="test", data_files="*.arrow")


def _evaluate(model, tokenizer, dataset, args):
    predictions = []
    references = []
    evaluator = EVALUATOR_BUILDERS[args.evaluator]()

    dataset = tqdm.tqdm(dataset, desc="Evaluating", total=len(dataset))
    for entry in dataset:
        ground_truth = entry['ground_truth']
        if args.is_icl:
            encoded_dict = encode_interleaved_data(entry, tokenizer, model.modalities)
        else:
            encoded_dict = encode_chat(entry, tokenizer, model.modalities)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=encoded_dict["input_ids"].unsqueeze(0).to(model.device),
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=args.do_sample,
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
            # print(f"Score: {evaluator.evaluate([generated_output], [ground_truth])}")

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
    
    # Load the dataset
    dataset = _resolve_dataset(eval_args.dataset_path) 

    if eval_args.lora_enable:
        model, tokenizer = load_trained_lora_model(
            model_name_or_path=eval_args.model_name_or_path,
            model_lora_path=eval_args.model_lora_path,
            load_bits=eval_args.load_bits,
        )
    else:
        model, tokenizer = load_trained_model(
            model_name_or_path=eval_args.model_name_or_path,
            pretrained_projectors_path=eval_args.projectors_path,
            load_bits=eval_args.load_bits,
        )
    
    score = _evaluate(model, tokenizer, dataset, eval_args)

    if eval_args.output_dir:
        _save_score(score, eval_args.output_dir)
