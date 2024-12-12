import argparse
import pandas as pd
import pathlib
from tqdm import tqdm
from utils import read_json, write_json
from itertools import chain
from sage.spelling_correction import AvailableCorrectors
from sage.spelling_correction import T5ModelForSpellingCorruption
import torch
from metrics import get_sentences

def load_corrector():
    corrector = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.ent5_large.value)
    corrector.model.to(torch.device("cuda"))
    return corrector

def correct_spelling(corrector, text):
    corrected_text = corrector.correct(text, prefix="grammar: ")
    return corrected_text[0]

def polish_text(text):
    sentences = get_sentences(text)
    polished_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
        words = sentence.split(" ")
        words = [word.strip() for word in words]
        words = [word.upper() if word == "i" else word for word in words if word]
        polished_sentences.append(" ".join(words))
    text = " ".join(polished_sentences)
    return text

def main():
    parser = argparse.ArgumentParser(description="Process human results")
    parser.add_argument("-r", "--results-path", type=str, help="Path to the human results file in csv")
    parser.add_argument("-c", "--config-path", type=str, help="Path to the config file in json")
    parser.add_argument("-e", "--experiment", type=str, help="Experiment name")
    
    args = parser.parse_args()

    results_path = pathlib.Path(args.results_path)

    human_results = pd.read_csv(results_path)
    config = read_json(args.config_path)

    metadata = {
        "source": args.results_path,
        "num_participants": len(human_results)
    }
    write_json({"metadata": metadata, "data": human_results.to_dict(orient="records")}, results_path.with_suffix(".json"))

    corrector = load_corrector()

    for _, row in tqdm(human_results.iterrows(), total=len(human_results), desc="Processing human results"):
        if row["Include"] == 0:
            continue
        human_id = row["ResponseId"]
        results = []
        source_data = read_json(config["source_data"])
        source_data_map = {sample["id"]: sample for sample in source_data["data"]}

        for q_index in range(1, 5):
            item_id = config["id_map"][f"Q{q_index}"]
            source_sample = source_data_map[item_id]
            output = row[f"Q{q_index}"]
            corrected_output = correct_spelling(corrector, output)
            polished_output = polish_text(corrected_output)
            results.append({
                **source_sample,
                "output": polished_output,
                "original_output": output,
                "result_id": f"{human_id}_Q{q_index}"
            })
        
        metadata = {
            "source": args.results_path,
            "num_questions": 4,
            "participant_id": human_id,
            "model": "human"
        }

        output_path = results_path.parent / args.experiment / f"{human_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json({"metadata": metadata, "data": results}, output_path)

if __name__ == '__main__':
    main()