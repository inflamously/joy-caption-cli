import gradio as gr
from pathlib import Path
import torch.amp.autocast_mode
from PIL import Image
import os

from model import load_clip_model, load_tokenizer, load_llm, load_image_adapter, load_vision_model
from prompt_image import caption_image

CLIP_PATH = "google/siglip-so400m-patch14-384"
CHECKPOINT_PATH = Path(os.path.join("models", "cgrkzexw-599808"))
TITLE = "<h1><center>JoyCaption Alpha Two (2024-09-26a)</center></h1>"
CAPTION_TYPE_MAP = {
    "Descriptive": [
        "Write a descriptive caption for this image in a formal tone.",
        "Write a descriptive caption for this image in a formal tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a formal tone.",
    ],
    "Descriptive (Informal)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Training Prompt": [
        "Write a stable diffusion prompt for this image.",
        "Write a stable diffusion prompt for this image within {word_count} words.",
        "Write a {length} stable diffusion prompt for this image.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Booru tag list": [
        "Write a list of Booru tags for this image.",
        "Write a list of Booru tags for this image within {word_count} words.",
        "Write a {length} list of Booru tags for this image.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}
HF_TOKEN = os.environ.get("HF_TOKEN", None)

clip_model = load_clip_model(CLIP_PATH)
load_vision_model(CHECKPOINT_PATH, clip_model)
tokenizer = load_tokenizer(CHECKPOINT_PATH)
text_model = load_llm(CHECKPOINT_PATH)
image_adapter = load_image_adapter(CHECKPOINT_PATH, clip_model, text_model)


@torch.no_grad()
def stream_chat(input_image: Image.Image, caption_type: str, caption_length: str | int, extra_options: list[str],
                name_input: str, custom_prompt: str) -> tuple[str, str]:
    return caption_image(
        tokenizer, text_model, clip_model, image_adapter, [input_image], caption_type, caption_length, extra_options,
        name_input, custom_prompt, CAPTION_TYPE_MAP)


with gr.Blocks() as demo:
    gr.HTML(TITLE)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")

            caption_type = gr.Dropdown(
                choices=["Descriptive", "Descriptive (Informal)", "Training Prompt", "MidJourney", "Booru tag list",
                         "Booru-like tag list", "Art Critic", "Product Listing", "Social Media Post"],
                label="Caption Type",
                value="Descriptive",
            )

            caption_length = gr.Dropdown(
                choices=["any", "very short", "short", "medium-length", "long", "very long"] +
                        [str(i) for i in range(20, 261, 10)],
                label="Caption Length",
                value="long",
            )

            extra_options = gr.CheckboxGroup(
                choices=[
                    "If there is a person/character in the image you must refer to them as {name}.",
                    "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
                    "Include information about lighting.",
                    "Include information about camera angle.",
                    "Include information about whether there is a watermark or not.",
                    "Include information about whether there are JPEG artifacts or not.",
                    "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
                    "Do NOT include anything sexual; keep it PG.",
                    "Do NOT mention the image's resolution.",
                    "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
                    "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
                    "Do NOT mention any text that is in the image.",
                    "Specify the depth of field and whether the background is in focus or blurred.",
                    "If applicable, mention the likely use of artificial or natural lighting sources.",
                    "Do NOT use any ambiguous language.",
                    "Include whether the image is sfw, suggestive, or nsfw.",
                    "ONLY describe the most important elements of the image."
                    "Describe the quality of the image as details as possible, including artifacts, bad anatomy and or unusual patterns that differ from your average training images.",
                    "Provide a detailed analysis of the image quality, highlighting any artifacts, anatomical inaccuracies, or unusual patterns that deviate from typical training data. Include observations about lighting, texture, proportions, and other visual elements that stand out as abnormal or inconsistent."
                ],
                label="Extra Options"
            )

            name_input = gr.Textbox(label="Person/Character Name (if applicable)")
            gr.Markdown("**Note:** Name input is only used if an Extra Option is selected that requires it.")

            custom_prompt = gr.Textbox(label="Custom Prompt (optional, will override all other settings)")
            gr.Markdown(
                "**Note:** Alpha Two is not a general instruction follower and will not follow prompts outside its training data well. Use this feature with caution.")

            run_button = gr.Button("Caption")

        with gr.Column():
            output_prompt = gr.Textbox(label="Prompt that was used")
            output_caption = gr.Textbox(label="Caption")

    run_button.click(fn=stream_chat,
                     inputs=[input_image, caption_type, caption_length, extra_options, name_input, custom_prompt],
                     outputs=[output_prompt, output_caption])

if __name__ == "__main__":
    demo.launch()
