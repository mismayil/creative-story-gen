import argparse
import pathlib
from tqdm import tqdm

from utils import read_json, write_json, get_template_keys
from prompts import *

SYSTEM_INSTRUCTION_TEMPLATES = {
    "pilot": PILOT_SYSTEM_INSTRUCTION_TEMPLATE,
}

USER_INSTRUCTION_TEMPLATES = {
    "pilot": PILOT_USER_INSTRUCTION_TEMPLATE,
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

SYSTEM_INSTRUCTION_PROCESSORS = {
    "default": prepare_system_instruction
}

USER_INSTRUCTION_PROCESSORS = {
    "default": prepare_user_instruction
}

def prepare_sample_for_eval(sample, template):
    system_instr_processor = SYSTEM_INSTRUCTION_PROCESSORS.get(
        template, SYSTEM_INSTRUCTION_PROCESSORS["default"]
    )
    user_instr_processor = USER_INSTRUCTION_PROCESSORS.get(
        template, USER_INSTRUCTION_PROCESSORS["default"]
    )
    
    eval_data = []
        
    system_prompt = system_instr_processor(sample, template)
    user_prompt = user_instr_processor(sample, template)

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
    parser.add_argument("-t", "--template", type=str, default="pilot", help="Template name")
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

    args = parser.parse_args()
    input_data = read_json(args.datapath)

    eval_data = []

    for sample in tqdm(input_data["data"], desc="Preparing task data for evaluation"):
            eval_data.extend(
                prepare_sample_for_eval(
                    sample,
                    template=args.template
                )
            )

    datapath = pathlib.Path(args.datapath)
    output_dir = pathlib.Path(args.output_dir) if args.output_dir is not None else datapath.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_data_path_stem = output_dir / f"{datapath.stem}_eval_{args.template}{args.suffix}"

    output_data = {
        "metadata": {
            "source": args.datapath,
            "template": args.template,
            "size": len(eval_data)
        },
        "data": eval_data
    }
    write_json(
        output_data, eval_data_path_stem.with_suffix(".json"), ensure_ascii=False
    )

    print(f"Output data saved to {eval_data_path_stem.with_suffix('.json')}")


if __name__ == "__main__":
    main()