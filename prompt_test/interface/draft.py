import requests
import gradio as gr
from module.llm_module import LLM_Module

def title_instruction_change(title_instruction_text) :
    return title_instruction_text


def intro_instruction_change(intro_instruction_text) :
    return intro_instruction_text


def form_change(form_text) :
    return form_text


def body_instruction_change(body_instruction_text) :
    return body_instruction_text


def conclusion_instruction_change(conclusion__instruction_text) :
    return conclusion__instruction_text


def get_data(
    instruction_text, 
    form_text,
    summary_document, 
    chunk_text, 
    temperature, 
    top_p, 
    gen_max_token) : 
        
    instruction_text += "\n" + form_text    
    params_dict = {
        "temperature" : float(temperature),
        "top_p" : float(top_p),
        "max_tokens" : int(gen_max_token)
    }
        
    result = LLM_Module.test_module(instruction_text, summary_document, params_dict)
    
    return result

