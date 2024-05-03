import os 
import shutil    
import argparse
import logging
import gradio as gr
import pandas as pd

from interface.file_dataframe import data_selected, uploaded_data_selected, load_json_file_data
from interface.summarization import (
    get_summary, 
    set_summary_otpion, 
)

from config.llm_option import SMMARIZATION_MAX_TOKEN_NUM
from config.path import EXAMPLE_FILE_PATH



def save_text(text, file_type):
    directory = f"{EXAMPLE_FILE_PATH}"
    file_path = os.path.join(directory, "custom_text.txt")
    with open(file_path, "w") as file:
        file.write(text)
    
    data = load_json_file_data(file_type)
    
    return data
    


def file_type_layout():
    gr.Markdown("# <br/> 파일 입력")
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("#### 파일 종류 선택")
            file_type = gr.Radio(choices=["articles", "meetingnotes", "reports", "etc"], label="파일 종류", value="articles")

    return file_type


def file_list_layout(file_type) : 
    with gr.Column(scale=2):                      
        gr.Markdown("#### 파일 선택")
        file_input = gr.File(label="파일을 선택해주세요", type="filepath")
        submit_button = gr.Button("Submit")
        data = gr.Dataframe(wrap=True, show_label=False, interactive=False, height=524)
            
        file_name = gr.Textbox(label="선택된 파일", interactive=False, autoscroll=False)
        document = gr.Textbox(label="", container=False, interactive=True, lines=16, autoscroll=True)
    
    document.change(fn=save_text, inputs=[document, file_type], outputs=[data])
    gr.Markdown("# <br/>")  

    return data, file_name, document, submit_button, file_input


def summary_layout(file_name, file_type, file_input) :    
    gr.Markdown("# <br/> 요약")
    gr.Markdown("#### 요약문 길이")
    with gr.Row():
        summary_token_num = gr.Textbox(value="", container=False, interactive=False, lines=1, max_lines=1, autoscroll=True)
    gr.Markdown("#### 요약문")
    with gr.Row():
        with gr.Column(scale=1):                                                
            summary_document = gr.Textbox(value="", container=False, interactive=False, lines=16, max_lines=16, autoscroll=True)
            get_summary_btn = gr.Button(value="Run", interactive=True)
    gr.Markdown("# <br/>")        
    get_summary_btn.click(fn=get_summary,
            inputs=[file_name, file_type, file_input],
            outputs=[summary_document, summary_token_num])

        
    return summary_document


def build_demo() :    
    with gr.Blocks() as demo :    
        file_type = file_type_layout()
        data, file_name, document, submit_button, file_input = file_list_layout(file_type)
        
        summary_document = summary_layout(file_name, file_type, file_input)
        
        file_type.change(fn=load_json_file_data, inputs=[file_type], outputs=[data])
        demo.load(fn=load_json_file_data, inputs=[file_type], outputs=[data])
        data.select(fn=data_selected, inputs=[data, file_type], outputs=[file_name, document])
        data.change(fn=save_text, inputs=[document, file_type], outputs=[data])
        submit_button.click(fn=uploaded_data_selected, inputs=[file_input], outputs=[file_name, document])
                                        
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11181)    
    args = parser.parse_args()    
    demo = build_demo()
    demo.queue().launch(server_name=args.host,
                        server_port=args.port,
                        share=False)