import easygui
import gradio as gr
import os
import json

from captions.joy.folder import process_caption_folder
from initialization import setup_config
from model import load_models
from state import APP_STATE

def generate_captions(
        folder,
        type,
        caption_length,
        extra_instruction,
        custom_prompt,
        batch_size
):
    process_caption_folder(folder, "", "", type, caption_length, extra_instruction, custom_prompt, batch_size)


setup_config()
load_models(APP_STATE['clip_model_name'], APP_STATE['checkpoint_path'])

if not "PYTHONUNBUFFERED" in os.environ:
    raise RuntimeError("Must start with unbuffered for easygui to work.")


def select_folder():
    folder = easygui.diropenbox("Folder of images:")
    return folder


with open("./config/config.json", "r") as f:
    config = json.load(f)

caption_types = config["captions"]["map"].keys()

with gr.Blocks(title="Image Caption Generator") as app:
    with gr.Row() as row:
        folder_selection = gr.Textbox(value=r"", label="Caption Folder")
        folder = gr.Button("Select Folder")
        folder.click(fn=select_folder, outputs=folder_selection)
    caption_type = gr.Radio(choices=caption_types, value="Training Prompt", label="Caption Type")
    caption_length = gr.Radio(choices=["short", "medium", "long"], value="short", label="Caption Length")
    extra_instruction = gr.Textbox(value="Describe the quality of the image as details as possible.",
                                   label="Extra Instruction")
    custom_prompt = gr.Textbox(value="", label="Custom Prompt")
    batch_size = gr.Slider(minimum=1, maximum=16, step=1, value=4, label="Batch Size")
    generator = gr.Button("Generate Captions")
    generator.click(fn=generate_captions,
                    inputs=[folder_selection, caption_type, caption_length, extra_instruction, custom_prompt,
                            batch_size])

app.launch()
