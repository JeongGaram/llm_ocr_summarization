import os 
import shutil    
import argparse
import logging
import gradio as gr
import pandas as pd

from interface.file_dataframe import data_selected, load_json_file_data
from interface.chunk_frame import set_splitter, get_chunk, clear_chunk
from interface.summarization import (
    get_summary, 
    summary_instruction_change, 
    set_summary_otpion, 
    set_samplingparams
)
from interface.draft import (
    body_instruction_change,    
    get_data,    
    form_change
)
from config.llm_option import (
    SMMARIZATION_INSTRUCTION, 
    SMMARIZATION_MAX_TOKEN_NUM, 
    SMMARIZATION_TOTAL_STOP, 
    GENERATE_INSTRUCTION,
    GENERATE_FORM, 
    CHUNK_SIZE,
    CHUNK_OVERLAP_SIZE,
    TEMPERATURE,
    TOP_P,
    GENERATE_MAX_TOKEN,
)


def file_list_layout() : 
    with gr.Row() :
        with gr.Column(scale=2):                            
            gr.Markdown("#### 파일 리스트")
            data = gr.Dataframe(wrap=True, show_label=False, interactive=False, height=524)            
        with gr.Column(scale=2):
            file_name = gr.Textbox(label="선택된 파일", interactive=False, autoscroll=False)
            gr.Markdown("#### Char num / Token num")
            with gr.Row():                    
                char_num = gr.Textbox(container=False, interactive=False, autoscroll=False)
                token_num = gr.Textbox(container=False, interactive=False, autoscroll=False)
            gr.Markdown("#### 원본 문서(특수문자 제거)")
            document = gr.Textbox(label="", container=False, interactive=False, lines=16, autoscroll=True)

    return data, file_name, char_num, token_num, document


def chunk_layout(document) : 
    gr.Markdown("## Set Chunk Splitter(base : langchain)")
    gr.Markdown("#### default : chunk_size 5000, chunk_overlap 500 <br> TODO : chunk 인코딩 문제 해결")                                                       
    with gr.Row() :
        chunk_size = gr.Textbox(value=CHUNK_SIZE, container=False, interactive=True, autoscroll=False)
        chunk_overlap = gr.Textbox(value=CHUNK_OVERLAP_SIZE, container=False, interactive=True, autoscroll=False)                
        set_splitter_btn = gr.Button(value="Set Chunks", interactive=True)
                
    gr.Markdown("## Make Chunk")            
    with gr.Row():
        with gr.Column(scale=1):                
            chunk_info_text = gr.Textbox(value="", container=False, interactive=False, lines=16, max_lines=16, autoscroll=True)                 
        with gr.Column(scale=3):                
            chunk_text = gr.Textbox(value="", container=False, interactive=False, lines=16, max_lines=16, autoscroll=True) 
        
    with gr.Row():            
        get_chunk_btn = gr.Button(value="Run", interactive=True)
        chunk_clear_btn = gr.Button(value="Clear", interactive=True)

        #splitter option
        set_splitter_btn.click(fn=set_splitter, inputs=[chunk_size, chunk_overlap], outputs=[chunk_size, chunk_overlap])
        
        #chunk
        get_chunk_btn.click(fn=get_chunk, inputs=[document], outputs=[chunk_info_text, chunk_text])
        chunk_clear_btn.click(fn=clear_chunk, inputs=None, outputs=[chunk_info_text, chunk_text])
        
    return chunk_text


def summary_layout(document, chunk_text) :    
    gr.Markdown("#### model max token = 입력 토큰 + 생성 토큰 개수 <br> 요약 모듈 호출 순서 : summary_module -> make_chunk_summary -> make_prompt -> llm_inference_api <br> summary_module : 원하는 토큰 수 만큼 요약 반복 loop 실행 <br> summary_max_token_num : 요악하길 원하는 토큰 수 <br> summary_total_step : 최대 반복 실행할 루프 횟수 <br> 최대 반복 실행하여도 토큰 수가 줄지 않는 문제가 있음.")
    gr.Markdown("#### TODO : 최대 반복 실행하여도 토큰 수가 줄지 않는 문제가 있음. -> prompt로 해결가능 <br> 입력 chunk 수가 많으면 오래걸림 -> 개선방안 고민중..")
    
    gr.Markdown("#### temperature / top_p / max_token <br>max_token : Maximum number of tokens to generate per output sequence <br>temperature: Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling")    
    with gr.Row() :            
        with gr.Column(scale=2):
            with gr.Row() :
                summary_temperature = gr.Textbox(label="temperature", value=TEMPERATURE, lines=1, container=False, interactive=True)
                summary_top_p = gr.Textbox(value=TOP_P, lines=1, container=False, interactive=True)
                summary_gen_max_token = gr.Textbox(value=GENERATE_MAX_TOKEN, lines=1, container=False, interactive=True)
        with gr.Column(scale=1):
            set_SamplingParams_btn = gr.Button(value="Set SamplingParams", interactive=True)
    gr.Markdown("#### Prompt")        
    with gr.Row():
        with gr.Column(scale=2):
            summary_instruction_text = gr.Textbox(value=SMMARIZATION_INSTRUCTION, container=False, interactive=True, lines=1, max_lines=1, autoscroll=True)
        with gr.Column(scale=1):
            set_summary_instruction_btn = gr.Button(value="요약 생성 지시문 변경", interactive=True)
    
    gr.Markdown("### summary_max_token_num / summary_total_step")    
    with gr.Row():            
        with gr.Column(scale=2):                
            with gr.Row():
                summary_max_token_num = gr.Textbox(value=SMMARIZATION_MAX_TOKEN_NUM, container=False, interactive=True, lines=1, max_lines=1, autoscroll=True)
                summary_total_step = gr.Textbox(value=SMMARIZATION_TOTAL_STOP, container=False, interactive=True, lines=1, max_lines=1, autoscroll=True)
        with gr.Column(scale=1):
            set_summary_option_btn = gr.Button(value="요약 모듈 옵션 변경", interactive=True)
    
    gr.Markdown("### Result")
    gr.Markdown("#### Summary Token num")
    with gr.Row():
        summary_token_num = gr.Textbox(value="", container=False, interactive=False, lines=1, max_lines=1, autoscroll=True)
    gr.Markdown("#### Summary")
    with gr.Row():
        with gr.Column(scale=1):                                                
            summary_document = gr.Textbox(value="", container=False, interactive=False, lines=16, max_lines=16, autoscroll=True)
            get_summary_btn = gr.Button(value="Run", interactive=True)
    gr.Markdown("# <br/>")        
    get_summary_btn.click(fn=get_summary,
            inputs=[
                document, chunk_text, summary_instruction_text, summary_max_token_num, summary_total_step,
                summary_temperature, summary_top_p, summary_gen_max_token
                ],
            outputs=[summary_document, summary_token_num])
    
        
    set_summary_instruction_btn.click(fn=summary_instruction_change, inputs=[summary_instruction_text], outputs=[summary_instruction_text])
    set_summary_option_btn.click(fn=set_summary_otpion, inputs=[summary_max_token_num, summary_total_step], outputs=[summary_max_token_num, summary_total_step])
    
    set_SamplingParams_btn.click(
        fn=set_samplingparams, 
        inputs=[summary_temperature, summary_top_p, summary_gen_max_token],
        outputs=[summary_temperature, summary_top_p, summary_gen_max_token]
        )
        
    return summary_document


def prompt_test_layout(summary_document, chunk_text) :    
    gr.Markdown("#### temperature / top_p / max_token")
    with gr.Row() :            
        with gr.Column(scale=2):
            with gr.Row() :                    
                body_temperature = gr.Textbox(label="temperature", value=TEMPERATURE, lines=1, container=False, interactive=True)
                body_top_p = gr.Textbox(value=TOP_P, lines=1, container=False, interactive=True)
                body_gen_max_token = gr.Textbox(value=GENERATE_MAX_TOKEN, lines=1, container=False, interactive=True)
        with gr.Column(scale=1):
            set_body_SamplingParams_btn = gr.Button(value="Set SamplingParams", interactive=True)
    gr.Markdown("#### Prompt")        
    with gr.Row():
        with gr.Column(scale=2):
            body_instruction_text = gr.Textbox(value=GENERATE_INSTRUCTION, container=False, interactive=True, lines=1, max_lines=1, autoscroll=True)
        with gr.Column(scale=1):
            set_body_instruction_btn = gr.Button(value="prompt 변경", interactive=True)
            
    gr.Markdown("#### One-Shot Set")
    with gr.Row():        
        body_form_text = gr.Textbox(value=GENERATE_FORM, container=False, interactive=True, lines=5, max_lines=5, autoscroll=True)        
    with gr.Row():
        set_body_form_btn = gr.Button(value="one-shot", interactive=True)   
               
    gr.Markdown("#### 생성 결과")
    with gr.Row():
        with gr.Column(scale=1):            
            body_document = gr.Textbox(value="", container=False, interactive=False, lines=10, max_lines=10, autoscroll=True)            
            get_body_btn = gr.Button(value="Run", interactive=True)                      
    
    set_body_form_btn.click(
        fn=form_change, 
        inputs=[body_form_text],
        outputs=[body_form_text]
        )
     
    set_body_SamplingParams_btn.click(
        fn=set_samplingparams,
        inputs=[body_temperature, body_top_p, body_gen_max_token],
        outputs=[body_temperature, body_top_p, body_gen_max_token]
        )
    
    set_body_instruction_btn.click(
        fn=body_instruction_change, 
        inputs=[body_instruction_text], 
        outputs=[body_instruction_text]
        )
    
    get_body_btn.click(
        fn=get_data, 
        inputs=[
            body_instruction_text, body_form_text, summary_document, chunk_text,
            body_temperature, body_top_p, body_gen_max_token
            ], 
        outputs=[body_document]
        )
        
    return body_document


def build_demo() :    
    with gr.Blocks() as demo :    
        data, file_name, char_num, token_num, document = file_list_layout()
        
        chunk_text = chunk_layout(document)
        
        gr.Markdown("## <br/> Summarization")
        summary_document = summary_layout(document, chunk_text)
                                                
        gr.Markdown("## <br/> Prompt Test") 
        body_document = prompt_test_layout(summary_document, chunk_text)
                                                                                                                                                                
        data.select(fn=data_selected, inputs=[data], outputs=[file_name, char_num, token_num, document])
        demo.load(fn=load_json_file_data, inputs=None, outputs=[data])
                                        
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