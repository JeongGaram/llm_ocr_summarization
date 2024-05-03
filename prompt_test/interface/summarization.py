import requests
import gradio as gr
from module.llm_module import LLM_Module
from interface.chunk_frame import get_chunk
import importlib

def load_llm_options(file_type):
    module_name = f"config.llm_option_{file_type}"
    llm_options = importlib.import_module(module_name)
    return llm_options


def get_summary(
    file_name,
    file_type,
    file_input
    ) :
    
    llm_options = load_llm_options(file_type)
    CHUNK_SIZE = llm_options.CHUNK_SIZE
    CHUNK_OVERLAP_SIZE = llm_options.CHUNK_OVERLAP_SIZE
    TEMPERATURE = llm_options.TEMPERATURE
    TOP_P = llm_options.TOP_P
    GENERATE_MAX_TOKEN = llm_options.GENERATE_MAX_TOKEN
    MAP_TEMPLATE = llm_options.MAP_TEMPLATE
    REDUCE_TEMPLATE = llm_options.REDUCE_TEMPLATE
    SMMARIZATION_MAX_TOKEN_NUM = llm_options.SMMARIZATION_MAX_TOKEN_NUM
    
    params_dict = {
        "temperature" : TEMPERATURE,
        "top_p" : TOP_P,
        "max_tokens" : GENERATE_MAX_TOKEN
    }    
                 
    summary_document = LLM_Module.summary_module(
        map_template=MAP_TEMPLATE,
        reduce_template=REDUCE_TEMPLATE,
        file_name=file_name, 
        file_type=file_type,
        summary_max_token_num=int(SMMARIZATION_MAX_TOKEN_NUM), 
        params_dict=params_dict,
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP_SIZE,
        file_input=file_input
        )
    summary_token_num = LLM_Module.get_token_num(input_text=summary_document)["token_num"]
    
    return summary_document, summary_token_num


def set_summary_otpion(summary_max_token_num) : 
    return summary_max_token_num
    
    
def summary_instruction_change(summary_instruction_text) :
    return summary_instruction_text


def set_samplingparams(summary_temperature, summary_top_p, gen_max_token) :
    return summary_temperature, summary_top_p, gen_max_token