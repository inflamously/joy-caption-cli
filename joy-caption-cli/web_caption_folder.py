import easygui
import gradio as gr
import os
import json

import model_selection
from captions.joy.folder import process_caption_folder
from initialization import setup_config


# File: web_caption_folder.py
# Author: nflamously
# Original License: Apache License 2.0

def generate_captions(
        folder,
        type,
        caption_length,
        extra_instruction,
        custom_prompt,
        batch_size
):
    process_caption_folder(folder, "text", "", type, caption_length, extra_instruction, custom_prompt, batch_size)


if not "PYTHONUNBUFFERED" in os.environ:
    raise RuntimeError("Must start with unbuffered for easygui to work.")


def select_folder():
    folder = easygui.diropenbox("Folder of images:")
    return folder


def load_model(model_type: str):
    setup_config(model_type)
    model_selection.load_model()


with open("./config/config.json", "r") as f:
    config = json.load(f)

caption_types = config["captions"]["map"].keys()

with gr.Blocks(title="Image Caption Generator") as app:
    gr.Markdown("## Image Caption Generator")
    with gr.Column():
        model_type = gr.Radio(["alpha", "beta"], value="alpha")
        load_model_type = gr.Button()
        load_model_type.click(fn=load_model, inputs=model_type, outputs=[])
        folder_selection = gr.Textbox(value=r"", label="Caption Folder")
        folder = gr.Button("Select Folder")
        folder.click(fn=select_folder, outputs=folder_selection)
    caption_type = gr.Radio(choices=caption_types, value="Training Prompt", label="Caption Type")
    caption_length = gr.Radio(choices=["short", "medium", "long"], value="short", label="Caption Length")
    extra_instruction = gr.Textbox(value="Describe the quality of the image as details as possible.",
                                   label="Extra Instruction")
    custom_prompt = gr.Textbox(value="", label="Custom Prompt")
    batch_size = gr.Slider(minimum=1, maximum=40, step=1, value=2, label="Batch Size")
    generator = gr.Button("Generate Captions")
    generator.click(fn=generate_captions,
                    inputs=[folder_selection, caption_type, caption_length, extra_instruction,
                            custom_prompt,
                            batch_size])

app.launch()
