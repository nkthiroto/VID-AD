import torch
from PIL import Image, MpoImagePlugin
import pathlib
from tqdm import tqdm
import io
import re
import os

# ============================================================
# Model loading functions
# ============================================================

def get_llava_model():
    """Load LLaVA v1.5-13b model and processor."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    MODEL_ID = "llava-hf/llava-1.5-13b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def get_llama_model():
    """Load Llama 3.2 11B Vision Instruct model and processor."""
    from transformers import MllamaForConditionalGeneration, AutoProcessor

    MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.tie_weights()
    return model, processor


def get_qwen_model():
    """Load Qwen2-VL-7B-Instruct model and processor."""
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def get_model(model_name):
    """Factory function to load the specified VLM.

    Args:
        model_name: One of 'llava', 'llama', or 'qwen'.

    Returns:
        (model, processor) tuple.
    """
    loaders = {
        "llava": get_llava_model,
        "llama": get_llama_model,
        "qwen": get_qwen_model,
    }
    if model_name not in loaders:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(loaders.keys())}")
    return loaders[model_name]()


# ============================================================
# Base SentenceGenerator
# ============================================================

class SentenceGenerator:
    """Base class for generating image descriptions and negative sentences.

    Subclasses must implement ``llm_ans`` and ``generate_negative_sentences``.
    """

    def __init__(self, prompt_path, negative_prompt_path,
                 train_path, test_good_path, test_anomaly_path,
                 result_path, model, processor):
        self.prompt_path = prompt_path
        self.negative_prompt_path = negative_prompt_path
        self.train_path = train_path
        self.test_good_path = test_good_path
        self.test_anomaly_path = test_anomaly_path
        self.result_path = result_path
        self.model = model
        self.processor = processor

        os.makedirs(self.result_path, exist_ok=True)

    # ----------------------------------------------------------
    # Prompt I/O
    # ----------------------------------------------------------

    def read_prompt(self):
        """Read the image description prompt from file."""
        try:
            with open(self.prompt_path, 'r') as file:
                return file.read()
        except Exception as e:
            print(f'Error in reading prompt file: {e}')
            return None

    def read_negative_prompt(self):
        """Read the negative-sentence generation prompt from file."""
        try:
            with open(self.negative_prompt_path, 'r') as file:
                return file.read()
        except Exception as e:
            print(f'Error in reading prompt file: {e}')
            return None

    # ----------------------------------------------------------
    # Image utilities
    # ----------------------------------------------------------

    def ensure_jpeg_image(self, image):
        """Convert any PIL image to a clean JPEG-format RGB image."""
        if isinstance(image, MpoImagePlugin.MpoImageFile):
            image.seek(0)
        img = image.convert("RGB") if image.mode != "RGB" else image
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        img = Image.open(buf)
        img = img.copy()
        buf.close()
        return img

    # ----------------------------------------------------------
    # Abstract methods (must be overridden by subclasses)
    # ----------------------------------------------------------

    # Aspect priority for sorting anomaly subdirectories
    _ASPECT_ORDER = ["Quantity", "Length", "Type", "Placement", "Relation"]

    def _aspect_sort_key(self, folder_name):
        """Return a sort key so that subdirectories are ordered by aspect priority.

        Priority: Quantity → Length → Type → Placement → Relation → (unknown) → Dual-Aspects / Mix
        """
        name_lower = folder_name.lower()
        # Dual-Aspects or Mix always comes last
        if "mix" in name_lower:
            return (len(self._ASPECT_ORDER) + 1, folder_name)
        # Check for known aspect keywords
        for i, aspect in enumerate(self._ASPECT_ORDER):
            if aspect.lower() in name_lower:
                return (i, folder_name)
        # Unknown aspects go after known ones but before Mix
        return (len(self._ASPECT_ORDER), folder_name)

    def llm_ans(self, image):
        """Generate a sentence describing the given image."""
        raise NotImplementedError

    def generate_negative_sentences(self, text):
        """Generate a negative (anomalous) variant of the given sentence."""
        raise NotImplementedError

    # ----------------------------------------------------------
    # Shared generation pipeline
    # ----------------------------------------------------------

    def create_prompt(self, text):
        """Combine the negative prompt template with the input text."""
        base_prompt = self.read_negative_prompt()
        return base_prompt + "\n\n" + text

    def generate_train_sentences(self):
        """Generate description sentences for all training images."""
        input_list = sorted(list(pathlib.Path(self.train_path).glob('**/*.jpg')), key=lambda x: x.stem)
        text_list = []

        output_file_path = self.result_path + '/train_results.txt'
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        for i in tqdm(range(len(input_list)), desc="Creating train sentences"):
            image_file = input_list[i]
            generated_content = self.llm_ans(Image.open(image_file))
            if generated_content:
                with open(output_file_path, 'a', encoding='utf-8') as output_file:
                    output_file.write(str(i) + " : ")
                    output_file.write(generated_content)
                    output_file.write("\n")
                    text_list.append(generated_content)
            torch.cuda.empty_cache()
        print("train_sentence is over.")

        return text_list

    def generate_test_sentences(self):
        """Generate description sentences for all test images (good + anomaly)."""
        test_true = []
        text_list = []

        # Good test images
        input_list_good = sorted(list(pathlib.Path(self.test_good_path).glob('**/*.jpg')), key=lambda x: x.stem)
        output_file_path_good = self.result_path + '/test_good_results.txt'
        os.makedirs(os.path.dirname(output_file_path_good), exist_ok=True)

        for i in tqdm(range(len(input_list_good)), desc="Creating good test sentences"):
            test_true.append(1)
            image_file = input_list_good[i]
            generated_content = self.llm_ans(Image.open(image_file))
            if generated_content:
                with open(output_file_path_good, 'a', encoding='utf-8') as output_file:
                    output_file.write(str(i) + " : ")
                    output_file.write(generated_content)
                    output_file.write("\n")
                    text_list.append(generated_content)
            torch.cuda.empty_cache()
        print("test_good_sentence is over.")

        # Anomaly test images
        anomaly_base = pathlib.Path(self.test_anomaly_path)
        output_file_path_anomaly = self.result_path + '/test_anomaly_results.txt'
        os.makedirs(os.path.dirname(output_file_path_anomaly), exist_ok=True)

        # Check for aspect subdirectories
        subdirs = [d for d in anomaly_base.iterdir() if d.is_dir()] if anomaly_base.exists() else []

        if subdirs:
            # Sort subdirectories by aspect priority order
            subdirs = sorted(subdirs, key=lambda d: self._aspect_sort_key(d.name))

            for subdir in subdirs:
                # Write subfolder name as a header
                with open(output_file_path_anomaly, 'a', encoding='utf-8') as f:
                    f.write(f"[{subdir.name}]\n")

                input_list = sorted(list(subdir.glob('*.jpg')), key=lambda x: x.stem)
                for i in tqdm(range(len(input_list)), desc=f"Creating anomaly test sentences ({subdir.name})"):
                    test_true.append(0)
                    image_file = input_list[i]
                    generated_content = self.llm_ans(Image.open(image_file))
                    if generated_content:
                        with open(output_file_path_anomaly, 'a', encoding='utf-8') as f:
                            f.write(str(i) + " : ")
                            f.write(generated_content)
                            f.write("\n")
                            text_list.append(generated_content)
                    torch.cuda.empty_cache()
        else:
            # Flat structure (no subdirectories)
            input_list_anomaly = sorted(list(anomaly_base.glob('*.jpg')), key=lambda x: x.stem)
            for i in tqdm(range(len(input_list_anomaly)), desc="Creating anomaly test sentences"):
                test_true.append(0)
                image_file = input_list_anomaly[i]
                generated_content = self.llm_ans(Image.open(image_file))
                if generated_content:
                    with open(output_file_path_anomaly, 'a', encoding='utf-8') as output_file:
                        output_file.write(str(i) + " : ")
                        output_file.write(generated_content)
                        output_file.write("\n")
                        text_list.append(generated_content)
                torch.cuda.empty_cache()

        print("test_anomaly_sentence is over.")

        return test_true, text_list

    def generate_train_negative_sentences_llm(self, train_sentences):
        """Generate negative sentences for all training sentences."""
        text_list = []
        output_file_path = self.result_path + '/train_negative_llm_results.txt'
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        for i in tqdm(range(len(train_sentences)), desc="Creating train negative sentences"):
            train_negative_sentences = self.generate_negative_sentences(train_sentences[i])
            with open(output_file_path, 'a', encoding='utf-8') as output_file:
                output_file.write(str(i) + " : ")
                output_file.write(train_negative_sentences)
                output_file.write("\n")
                text_list.append(train_negative_sentences)
            torch.cuda.empty_cache()
        print("train_negative_sentence is over.")

        return text_list

    def generate_full_sentences(self):
        """Run the full sentence generation pipeline.

        Returns:
            (train_sentences, train_negative_sentences, test_sentences, test_true)
        """
        train_sentences = self.generate_train_sentences()
        train_negative_sentences = self.generate_train_negative_sentences_llm(train_sentences)
        test_true, test_sentences = self.generate_test_sentences()

        # Remove line breaks
        train_sentences = [s.replace('\n', '').replace('\r', '') for s in train_sentences]
        train_negative_sentences = [s.replace('\n', '').replace('\r', '') for s in train_negative_sentences]
        test_sentences = [s.replace('\n', '').replace('\r', '') for s in test_sentences]

        return train_sentences, train_negative_sentences, test_sentences, test_true


# ============================================================
# LLaVA v1.5 SentenceGenerator
# ============================================================

class LLaVASentenceGenerator(SentenceGenerator):
    """SentenceGenerator using LLaVA v1.5-13b."""

    def llm_ans(self, image):
        torch.cuda.empty_cache()
        image = self.ensure_jpeg_image(image)
        prompt = self.read_prompt()
        if prompt is None:
            raise ValueError(f"Prompt cannot be loaded: {self.prompt_path}")

        # LLaVA v1.5 format
        conversation = f"USER: <image>\n{prompt} ASSISTANT:"

        inputs = self.processor(
            text=conversation,
            images=image,
            return_tensors="pt"
        ).to(self.model.device, torch.float16)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.001,
                top_p=0.95,
            )

        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)

        if "ASSISTANT:" in generated_text:
            generated_text = generated_text.split("ASSISTANT:")[-1].strip()

        s = generated_text
        s = re.sub(r'\.(\s*\n\s*)', '. ', s)
        s = s.replace('\n', ' ').replace('\r', ' ')
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def generate_negative_sentences(self, text):
        torch.cuda.empty_cache()
        prompt = self.create_prompt(text)

        conversation = f"USER: {prompt} ASSISTANT:"

        inputs = self.processor(
            text=conversation,
            return_tensors="pt"
        ).to(self.model.device, torch.float16)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
            )

        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)

        if "ASSISTANT:" in generated_text:
            generated_text = generated_text.split("ASSISTANT:")[-1].strip()

        result = generated_text
        result = re.sub(r'\.(\s*\n\s*)', '. ', result)
        result = result.replace('\n', ' ').replace('\r', ' ')
        result = re.sub(r'\s+', ' ', result).strip()
        return result


# ============================================================
# Llama 3.2 Vision SentenceGenerator
# ============================================================

class LlamaSentenceGenerator(SentenceGenerator):
    """SentenceGenerator using Llama 3.2 Vision Instruct."""

    def llm_ans(self, image):
        torch.cuda.empty_cache()
        prompt = self.read_prompt()
        if prompt is None:
            raise ValueError(f"Prompt cannot be loaded: {self.prompt_path}")

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]

        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.001,
                top_p=0.95,
            )

        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        s = generated_text.replace(prompt, '')
        s = s.replace("assistant", '')
        s = s.replace("user", '')
        s = re.sub(r'\.(\s*\n\s*)', '. ', s)
        s = s.replace('\n', ' ').replace('\r', ' ')
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def generate_negative_sentences(self, text):
        torch.cuda.empty_cache()
        prompt = self.create_prompt(text)

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt}
            ]}
        ]

        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        inputs = self.processor(
            text=input_text,
            images=None,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
            )

        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.processor.decode(generated_tokens, skip_special_tokens=True)

        result = generated_text
        unwanted_patterns = [
            r"system.*?Today Date:.*?\d{1,2}\s+\w+\s+\d{4}",
            r"Cutting Knowledge Date:.*?\d{4}",
            r"assistant\s*",
            r"user\s*",
        ]
        for pattern in unwanted_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)

        result = re.sub(r'\.(\s*\n\s*)', '. ', result)
        result = result.replace('\n', ' ').replace('\r', ' ')
        result = re.sub(r'\s+', ' ', result).strip()
        return result


# ============================================================
# Qwen2-VL SentenceGenerator
# ============================================================

class QwenSentenceGenerator(SentenceGenerator):
    """SentenceGenerator using Qwen2-VL-7B-Instruct."""

    def llm_ans(self, image):
        torch.cuda.empty_cache()
        prompt = self.read_prompt()
        if prompt is None:
            raise ValueError(f"Prompt cannot be loaded: {self.prompt_path}")

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]

        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=[input_text],
            images=[image],
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.001,
                top_p=0.95,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.processor.decode(generated_tokens, skip_special_tokens=True)

        s = generated_text
        s = s.replace("assistant", '')
        s = s.replace("user", '')
        s = re.sub(r'\.(\s*\n\s*)', '. ', s)
        s = s.replace('\n', ' ').replace('\r', ' ')
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def generate_negative_sentences(self, text):
        torch.cuda.empty_cache()
        prompt = self.create_prompt(text)

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt}
            ]}
        ]

        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=[input_text],
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
            )

        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.processor.decode(generated_tokens, skip_special_tokens=True)

        result = generated_text
        unwanted_patterns = [
            r"system.*?Today Date:.*?\d{1,2}\s+\w+\s+\d{4}",
            r"Cutting Knowledge Date:.*?\d{4}",
            r"assistant\s*",
            r"user\s*",
        ]
        for pattern in unwanted_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)

        result = re.sub(r'\.(\s*\n\s*)', '. ', result)
        result = result.replace('\n', ' ').replace('\r', ' ')
        result = re.sub(r'\s+', ' ', result).strip()
        return result


# ============================================================
# Factory
# ============================================================

GENERATOR_CLASSES = {
    "llava": LLaVASentenceGenerator,
    "llama": LlamaSentenceGenerator,
    "qwen": QwenSentenceGenerator,
}


def get_sentence_generator(model_name, **kwargs):
    """Create a SentenceGenerator instance for the specified model.

    Args:
        model_name: One of 'llava', 'llama', or 'qwen'.
        **kwargs: Arguments forwarded to the SentenceGenerator constructor.

    Returns:
        A model-specific SentenceGenerator instance.
    """
    if model_name not in GENERATOR_CLASSES:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(GENERATOR_CLASSES.keys())}")
    return GENERATOR_CLASSES[model_name](**kwargs)
