import sys

import gradio as gr
import os
import subprocess


def process_images(folder_path, batch_size):
    # Construct the CLI command
    command = [
        sys.executable,
        "cli.py",
        "caption",
        "folder",
        folder_path,
        "--caption_type", "Describe",
        "--caption_length", "short",
        "--batch_size", str(batch_size)
    ]

    result_stdout = None

    try:
        # Run the command
        process_result = subprocess.run(command, capture_output=True, text=True)
        if process_result.returncode != 0:
            result_stdout = process_result.stderr
    except Exception as error:
        result_stdout = str(error)

    if result_stdout is None:
        result_stdout = process_result.stdout

    # Return the output
    return result_stdout


# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Image Captioning with Gradio")

    with gr.Row():
        with gr.Column():
            folder_path = gr.Textbox(label="Folder Path", placeholder="Enter or select the folder path")

    with gr.Row():
        batch_size_input = gr.Number(label="Batch Size", value=2, precision=0)

    process_button = gr.Button("Process Images")

    stdout = gr.Textbox(label="Output", lines=10, interactive=False)

    # Function to handle processing
    def process(folder_path, batch_size):
        if not os.path.isdir(folder_path):
            return "Invalid folder path. Please select a valid folder."
        return process_images(folder_path, batch_size)


    process_button.click(process, inputs=[folder_path, batch_size_input], outputs=stdout)

# Launch the Gradio interface
demo.launch()