import requests
import gradio as gr
from module.llm_module import LLM_Module

def get_summary(
    remove_document,
    chunk_text, 
    summary_instruction_text, 
    summary_max_token_num, 
    summary_total_step, 
    summary_temperature, 
    summary_top_p, 
    summary_gen_max_token
    ) :
    params_dict = {
        "temperature" : float(summary_temperature),
        "top_p" : float(summary_top_p),
        "max_tokens" : int(summary_gen_max_token)
    }    
                       
    summary_document = LLM_Module.summary_module(
        summary_instruction=summary_instruction_text,
        ori_doc=remove_document, 
        chunks=chunk_text, 
        summary_max_token_num=int(summary_max_token_num), 
        summary_total_step=int(summary_total_step),
        params_dict=params_dict
        )
    summary_token_num = LLM_Module.get_token_num(input_text=summary_document)["token_num"]
    
    return summary_document, summary_token_num


def set_summary_otpion(summary_max_token_num, summary_total_step) : 
    return summary_max_token_num, summary_total_step
    
    
def summary_instruction_change(summary_instruction_text) :
    return summary_instruction_text


def set_samplingparams(summary_temperature, summary_top_p, gen_max_token) :
    return summary_temperature, summary_top_p, gen_max_token