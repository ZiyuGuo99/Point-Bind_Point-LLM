import argparse

import gradio as gr
import plotly.graph_objects as go
import torch
import torch.cuda

import ImageBind.data as data
from diffusers import StableUnCLIPImg2ImgPipeline
from image_generate import image_generate

import llama

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="7B", type=str,
    help="Name of or path to pretrained checkpoint",
)
parser.add_argument(
    "--llama_type", default="7B", type=str,
    help="Type of llama original weight",
)
parser.add_argument(
    "--llama_dir", default="/path/to/LLaMA", type=str,
    help="Path to LLaMA pretrained checkpoint",
)
args = parser.parse_args()

model = llama.load(args.model, args.llama_dir, knn=True, llama_type=args.llama_type)
model.eval()

pipe = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", cache_dir="./ckpts")
pipe = pipe.to("cuda")


def multimodal_generate(
        modality,
        img_path,
        img_weight,
        text_path,
        text_weight,
        video_path,
        video_weight,
        audio_path,
        audio_weight,
        point_path,
        point_weight,
        prompt,
        question_input,
        cache_size,
        cache_t,
        cache_weight,
        max_gen_len,
        gen_t, top_p, output_type
):
    if len(modality) == 0:
        raise gr.Error('Please select at least one modality!')

    inputs = {}
    if 'Image' in modality:
        if img_path is None:
            raise gr.Error('Please select an image')
        if img_weight == 0:
            raise gr.Error('Please set the weight')
        image = data.load_and_transform_vision_data([img_path], device='cuda')
        inputs['Image'] = [image, img_weight]
    if 'Text' in modality:
        if text_path == '':
            raise gr.Error('Please input the text')
        if text_weight == 0:
            raise gr.Error('Please set the weight')
        text = data.load_and_transform_text([text_path], device='cuda')
        inputs['Text'] = [text, text_weight]
    if 'Video' in modality:
        if video_path is None:
            raise gr.Error('Please select a video')
        if video_weight == 0:
            raise gr.Error('Please set the weight')
        video = data.load_and_transform_video_data([video_path], device='cuda')
        inputs['Video'] = [video, video_weight]
    if 'Audio' in modality:
        if audio_path is None:
            raise gr.Error('Please select an audio')
        if audio_weight == 0:
            raise gr.Error('Please set the weight')
        audio = data.load_and_transform_audio_data([audio_path], device='cuda')
        inputs['Audio'] = [audio, audio_weight]
    if 'Point Cloud' in modality:
        if point_path is None:
            raise gr.Error('Please select a point cloud')
        if point_weight == 0:
            raise gr.Error('Please set the weight')
        point = data.load_and_transform_point_cloud_data([point_path], device='cuda')
        inputs['Point'] = [point, point_weight]

    image_prompt = prompt # image use original prompt

    text_output = None
    image_output = None
    if output_type == "Text":
        # text output
        prompts = [llama.format_prompt(prompt, question_input)]

        prompts = [model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        with torch.cuda.amp.autocast():
            results = model.generate(inputs, prompts, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p,
                                         cache_size=cache_size, cache_t=cache_t, cache_weight=cache_weight)
        text_output = results[0].strip()
        print(text_output)

    else:
        # image output
        image_output = image_generate(inputs, model, pipe, image_prompt, cache_size, cache_t, cache_weight)

    return text_output, image_output

def show_point_cloud(file):
    point = torch.load(file.name).numpy()
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=point[:,0], y=point[:,1], z=point[:,2], 
                mode='markers',
                marker=dict(
                size=1.2,
                color='gray'
            )
            )
        ],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            )
        ),
    )
    return fig


def create_imagebind_llm_demo():
    with gr.Blocks() as imagebind_llm_demo:
        modality = gr.CheckboxGroup(choices=['Image', 'Text', 'Video', 'Audio', 'Point Cloud'], value='Image', interactive=True,
                                    label='Input Modalities')
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column() as image_input:
                        img_path = gr.Image(label='Image Input', type='filepath')
                        img_weight = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, interactive=True, label='Weight')
                    with gr.Column() as text_input:
                        text_path = gr.Textbox(label='Text Input', lines=9)
                        text_weight = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, interactive=True, label='Weight')
                with gr.Row():
                    with gr.Column() as video_input:
                        video_path = gr.Video(label='Video Input')
                        video_weight = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, interactive=True, label='Weight')
                    with gr.Column() as audio_input:
                        audio_path = gr.Audio(label='Audio Input', type='filepath')
                        audio_weight = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, interactive=True, label='Weight')
                with gr.Row():
                    with gr.Column(scale=1) as point_input:
                        point_path = gr.File(label='Point Cloud Input', elem_id="pointpath", elem_classes="")
                        output = gr.Plot()
                        btn = gr.Button(value="Show Point Cloud")
                        btn.click(show_point_cloud, point_path, output)
                        point_weight = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, interactive=True, label='Weight')
            with gr.Column():
                with gr.Row():
                    output_dropdown = gr.Dropdown(['Text', 'Image'], value='Text', label='Output type')
                with gr.Row():
                    prompt = gr.Textbox(lines=4, label="Question")
                with gr.Row():
                    question_input = gr.Textbox(lines=4, label="Question Input (Optional)")
                with gr.Row():
                    cache_size = gr.Slider(minimum=1, maximum=100, value=10, interactive=True, label="Cache Size")
                    cache_t = gr.Slider(minimum=0.0, maximum=100, value=20, interactive=True, label="Cache Temperature")
                    cache_weight = gr.Slider(minimum=0.0, maximum=1, value=0.5, interactive=True, label="Cache Weight")
                with gr.Row() as text_config_row:
                    max_gen_len = gr.Slider(minimum=1, maximum=512, value=128, interactive=True, label="Max Length")
                    # with gr.Accordion(label='Advanced options', open=False):
                    gen_t = gr.Slider(minimum=0, maximum=1, value=0.1, interactive=True, label="Temperature")
                    top_p = gr.Slider(minimum=0, maximum=1, value=0.75, interactive=True, label="Top p")
                with gr.Row():
                    # clear_botton = gr.Button("Clear")
                    run_botton = gr.Button("Run", variant='primary')

                with gr.Row():
                    gr.Markdown("Output")
                with gr.Row():
                    text_output = gr.Textbox(lines=11, label='Text Out')
                    image_output = gr.Image(label='Image Out', visible=False)

    def modality_select(modality, img, text, video, audio, point):
        modality = []
        if img is not None:
            modality.append('Image')
        if len(text) > 0:
            modality.append('Text')
        if video is not None:
            modality.append('Video')
        if audio is not None:
            modality.append('Audio')
        if point is not None:
            modality.append('Point Cloud')
        return modality

    def change_output_type(output_type):
        if output_type == 'Text':
            result = [gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(label='Question'),
            gr.update(visible=True)]
        elif output_type == 'Image':
            result = [gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(label='Prompt'),
            gr.update(visible=False)]

        return result

    output_dropdown.change(change_output_type, output_dropdown,
                           [image_output, text_output, question_input, prompt, text_config_row])


    img_path.change(modality_select, inputs=[modality, img_path, text_path, video_path, audio_path, point_path], outputs=[modality])
    text_path.blur(modality_select, inputs=[modality, img_path, text_path, video_path, audio_path, point_path], outputs=[modality])
    video_path.change(modality_select, inputs=[modality, img_path, text_path, video_path, audio_path, point_path],
                      outputs=[modality])
    audio_path.change(modality_select, inputs=[modality, img_path, text_path, video_path, audio_path, point_path],
                      outputs=[modality])
    point_path.change(modality_select, inputs=[modality, img_path, text_path, video_path, audio_path, point_path],
                      outputs=[modality])

    inputs = [
        modality,
        img_path, img_weight,
        text_path, text_weight,
        video_path, video_weight,
        audio_path, audio_weight,
        point_path, point_weight,
        prompt, question_input,
        cache_size, cache_t, cache_weight,
        max_gen_len, gen_t, top_p, output_dropdown
    ]
    outputs = [text_output, image_output]
    run_botton.click(fn=multimodal_generate, inputs=inputs, outputs=outputs)

    # gr.Examples(
    #     examples=examples,
    #     inputs=inputs,
    #     outputs=outputs,
    #     fn=multimodal_generate,
    #     cache_examples=False)

    return imagebind_llm_demo


description = """
# Point-LLM🚀
"""

with gr.Blocks(theme=gr.themes.Default(), css="#pointpath {height: 10em} .label {height: 3em}") as demo:
    gr.Markdown(description)
    create_imagebind_llm_demo()


demo.queue(api_open=True, concurrency_count=1).launch(share=True)
