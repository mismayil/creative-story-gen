import argparse
import pathlib
from tqdm import tqdm

from utils import read_json, write_json, get_template_keys
from prompts import *

SYSTEM_INSTRUCTION_TEMPLATES = {
    "default": DEFAULT_SYSTEM_INSTRUCTION_TEMPLATE,
    "test": TEST_SYSTEM_INSTRUCTION_TEMPLATE,
    "summary": SUMMARY_SYSTEM_INSTRUCTION_TEMPLATE,
}

USER_INSTRUCTION_TEMPLATES = {
    "default": DEFAULT_USER_INSTRUCTION_TEMPLATE,
    "test": TEST_USER_INSTRUCTION_TEMPLATE,
    "summary": SUMMARY_USER_INSTRUCTION_TEMPLATE,
}

SHOT_TEMPLATES = {
    "summary": SUMMARY_SHOT_TEMPLATE
}

def prepare_template_value(value):
    if isinstance(value, list):
        return ", ".join(value)
    return value

def prepare_template(sample, template):
    template_keys = get_template_keys(template)
    format_args = {k: prepare_template_value(sample[k]) for k in template_keys}
    return template.format(**format_args)

def prepare_system_instruction(sample, template):
    instruction_template = SYSTEM_INSTRUCTION_TEMPLATES[template]
    return prepare_template(sample, instruction_template)

def prepare_user_instruction(sample, template):
    instruction_template = USER_INSTRUCTION_TEMPLATES[template]
    return prepare_template(sample, instruction_template)

def prepare_summary_shot_prompt(sample, template, num_shots=1, shot_data=None):
    if not shot_data:
        return ""

    shot_samples = [s for s in shot_data["data"] if s["id"] == sample["item_id"]][:num_shots]
    shot_template = SHOT_TEMPLATES[template]
    shots = []

    for i, shot_sample in enumerate(shot_samples):
        shots.append(
            prepare_template({**shot_sample, "index": i+1}, shot_template).strip()
        )

    final_shot = prepare_template({**sample, "summary": "", "index": len(shot_samples)+1}, shot_template).strip()
    return "\n\n".join(shots + [final_shot]).strip()

SYSTEM_INSTRUCTION_PROCESSORS = {
    "default": prepare_system_instruction
}

USER_INSTRUCTION_PROCESSORS = {
    "default": prepare_user_instruction
}

SHOT_PROCESSORS = {
    "default": lambda *args, **kwargs: "",
    "summary": prepare_summary_shot_prompt
}

def prepare_sample_for_eval(sample, template, num_shots=1, shot_data=None):
    system_instr_processor = SYSTEM_INSTRUCTION_PROCESSORS.get(
        template, SYSTEM_INSTRUCTION_PROCESSORS["default"]
    )
    user_instr_processor = USER_INSTRUCTION_PROCESSORS.get(
        template, USER_INSTRUCTION_PROCESSORS["default"]
    )
    shot_processor = SHOT_PROCESSORS.get(
        template, SHOT_PROCESSORS["default"]
    )
    
    eval_data = []
        
    system_prompt = system_instr_processor(sample, template)
    user_prompt = user_instr_processor(sample, template)
    shot_prompt = shot_processor(sample, template, num_shots=num_shots, shot_data=shot_data)

    if shot_prompt:
        user_prompt += "\n\n" + shot_prompt

    eval_data.append(
        {
            **sample,
            "system_prompt": system_prompt.strip(),
            "user_prompt": user_prompt.strip(),
            "template": template
        }
    )

    return eval_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datapath", type=str, help="Path to task data in json", required=True)
    parser.add_argument("-t", "--template", type=str, default="default", help="Template name")
    parser.add_argument(
        "-s",
        "--suffix",
        type=str,
        default="",
        help="Custom suffix for output file path.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory path. Defaults to input directory path.",
    )
    parser.add_argument("-sp", "--shot-path", type=str, default=None, help="Path to shot examples in json")
    parser.add_argument("-n", "--num-shots", type=int, default=1, help="Number of shot examples to include")

    args = parser.parse_args()
    input_data = read_json(args.datapath)
    shot_data = read_json(args.shot_path) if args.shot_path is not None else None

    eval_data = []

    for sample in tqdm(input_data["data"], desc="Preparing task data for evaluation"):
            eval_data.extend(
                prepare_sample_for_eval(
                    sample,
                    template=args.template,
                    num_shots=args.num_shots,
                    shot_data=shot_data
                )
            )

    datapath = pathlib.Path(args.datapath)
    output_dir = pathlib.Path(args.output_dir) if args.output_dir is not None else datapath.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_data_path = output_dir / f"{datapath.stem}_eval_{args.template}{args.suffix}.json"

    output_data = {
        "metadata": {
            "source": args.datapath,
            "template": args.template,
            "size": len(eval_data),
            "shot_path": args.shot_path,
        },
        "data": eval_data
    }
    write_json(output_data, eval_data_path)

    print(f"Output data saved to {eval_data_path}")


if __name__ == "__main__":
    main()