import argparse
import gc
import torch

import writting
import train_and_evaluate

# Supported VLM model names
SUPPORTED_MODELS = ["qwen", "llama", "llava"]



# Map model names to their module directory names
MODEL_DIR_MAP = {
    "qwen": "Qwen",
    "llama": "Llama",
    "llava": "LLaVA",
}


# All available dataset names
ALL_DATASET_NAMES = [
    "Sticks", "Fruits", "Tools", "Cookies", "Tapes",
    "Stationery", "Ropes", "Blocks", "Dishes", "Balls",
]

# All available conditions
ALL_CONDITIONS = ["Original", "Cable_BG", "Mesh_BG", "Low-light_CD", "Blurry_CD"]

# Map CLI condition names to actual directory suffixes
CONDITION_SUFFIX_MAP = {
    "Original": "",
    "Cable_BG": "_Cable_BG",
    "Mesh_BG": "_Mesh_BG",
    "Low-light_CD": "_Low-light_CD",
    "Blurry_CD": "_Blurry_CD",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run anomaly detection pipeline with a selectable VLM backbone."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=SUPPORTED_MODELS,
        help="VLM model to use for sentence generation (qwen, llama, or llava).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        choices=ALL_DATASET_NAMES,
        help="Dataset(s) to process. If not specified, all datasets are used.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=None,
        choices=ALL_CONDITIONS,
        help="Condition(s) to process (Original, Cable_BG, Mesh_BG, Low-light_CD, Blurry_CD). If not specified, all conditions are used.",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./output",
        help="Base directory for output (results and models).",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./dataset/VID-AD_dataset",
        help="Base directory of the VID-AD dataset.",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default="./prompt",
        help="Directory containing prompt files.",
    )
    return parser.parse_args()


def build_dataset_configs(dataset_names, condition_names):
    """Build dataset config list from the given dataset names and conditions."""
    suffixes = [CONDITION_SUFFIX_MAP[c] for c in condition_names]
    return [
        {"DATASET_NAME": name, "CONDITION": cond}
        for name in dataset_names
        for cond in suffixes
    ]


def main(model_name, vlm_model, vlm_processor, base_dir, dataset_dir, prompt_dir, dataset_configs):
    """Main processing loop over all dataset/condition combinations."""
    model_dir = MODEL_DIR_MAP[model_name]

    for config in dataset_configs:
        DATASET_NAME = config['DATASET_NAME']
        CONDITION = config['CONDITION']

        print(f"\n=== Processing {DATASET_NAME}{CONDITION} ({model_name}) ===")

        # Path configuration
        PROMPT_PATH = f'{prompt_dir}/{DATASET_NAME}_prompt.txt'
        NEGATIVE_PROMPT_PATH = f'{prompt_dir}/negative_sentence_prompt.txt'
        TRAIN_PATH = f'{dataset_dir}/{DATASET_NAME}{CONDITION}/train/good'
        TEST_GOOD_PATH = f'{dataset_dir}/{DATASET_NAME}{CONDITION}/test/good'
        TEST_ANOMALY_PATH = f'{dataset_dir}/{DATASET_NAME}{CONDITION}/test/logical_anomalies'
        RESULT_PATH = f'{base_dir}/{model_dir}/results/{DATASET_NAME}{CONDITION}'
        SAVE_PATH = f'{base_dir}/{model_dir}/models/{DATASET_NAME}{CONDITION}'

        # Create SentenceGenerator instance for the selected model
        writting_instance = writting.get_sentence_generator(
            model_name,
            prompt_path=PROMPT_PATH,
            negative_prompt_path=NEGATIVE_PROMPT_PATH,
            train_path=TRAIN_PATH,
            test_good_path=TEST_GOOD_PATH,
            test_anomaly_path=TEST_ANOMALY_PATH,
            result_path=RESULT_PATH,
            model=vlm_model,
            processor=vlm_processor,
        )

        # Generate sentences
        train_sentences, train_negative_sentences, test_sentences, test_true = (
            writting_instance.generate_full_sentences()
        )

        # Train and evaluate
        train_and_evaluate_instance = train_and_evaluate.ComparativeLearning(
            train_sentences=train_sentences,
            train_negative_sentences=train_negative_sentences,
            test_sentences=test_sentences,
            test_true=test_true,
            save_path=SAVE_PATH,
            result_path=RESULT_PATH,
        )

        train_and_evaluate_instance.train_and_evaluate_model(save_model_path=SAVE_PATH)

        print(f"Completed {DATASET_NAME}{CONDITION}")

        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()

    print("\nAll datasets processed successfully!")


if __name__ == '__main__':
    args = parse_args()

    torch.cuda.empty_cache()

    # Build dataset configs from selected datasets and conditions (or all)
    selected_datasets = args.datasets if args.datasets else ALL_DATASET_NAMES
    selected_conditions = args.conditions if args.conditions else ALL_CONDITIONS
    dataset_configs = build_dataset_configs(selected_datasets, selected_conditions)
    print(f"Datasets: {selected_datasets}")
    print(f"Conditions: {selected_conditions}")

    print(f"Initializing {args.model} model...")
    vlm_model, vlm_processor = writting.get_model(args.model)
    print("Model loaded successfully!")

    main(args.model, vlm_model, vlm_processor, args.base_dir, args.dataset_dir, args.prompt_dir, dataset_configs)
