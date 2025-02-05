from PIL import Image
import torch
import fire
from optimized_inference import OptimizedInference
from paligemma_processor import PaliGemmaProcessor
from gemma_model import KVCache, PaliGemmaForConditionalGeneration
from transformers import AutoProcessor, AutoModelForPreTraining
import os
from dotenv import load_dotenv

def move_inputs_to_device(model_inputs: dict, device: str):
    return {k: v.to(device) for k, v in model_inputs.items()}

def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str
):
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs

def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    n_predict: int
):
    # For testing, we use our optimized inference wrapper.
    inference_wrapper = OptimizedInference(model, processor, device)
    generated_text = inference_wrapper.generate(
        image_path=image_file_path,
        prompt=prompt,
        max_length=max_tokens_to_generate,
        temperature=temperature,
        top_p=top_p,
        n_predict=n_predict
    )
    print(f"Generated text: {generated_text}")

def main(
    prompt: str = None,
    image_file_path: str = None,
    max_tokens_to_generate: int = 300,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = True,
    only_cpu: bool = False,
    n_predict: int = 4  # number of tokens to predict in parallel
):
    """Main inference function with accelerated decoding."""
    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        raise ValueError(
            "Please set your Hugging Face token in the .env file or environment variables.\n"
            "You can get your token from https://huggingface.co/settings/tokens"
        )
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    device = "cpu" if only_cpu else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
    model = AutoModelForPreTraining.from_pretrained(
        "google/paligemma-3b-pt-224",
        torch_dtype=torch.float16 if device == "mps" else "auto"
    )
    
    # Test the accelerated decoding
    test_inference(model, processor, device, prompt, image_file_path, max_tokens_to_generate, temperature, top_p, do_sample, n_predict)

if __name__ == "__main__":
    fire.Fire(main)
