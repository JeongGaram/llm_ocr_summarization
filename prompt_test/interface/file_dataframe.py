import os
from langchain_community.document_loaders import UnstructuredPDFLoader
import gradio as gr
import pandas as pd
from config.path import EXAMPLE_FILE_PATH
from module.llm_module import LLM_Module

def load_json_file_data(file_type):
    file_list = os.listdir(os.path.join(EXAMPLE_FILE_PATH, file_type))
    example_list = []
    for i, file_name in enumerate(file_list) :
        file_name = file_name
        example_list.append([i, file_name])    
    examples = pd.DataFrame(example_list, columns=["id", "file_name"])
    return examples


def data_selected(
        selected_index: gr.SelectData,
        data,
        file_type
    ) :
    
    index = selected_index.index[0]    
    file_name = data.iloc[index]["file_name"]
    document_path = os.path.join(EXAMPLE_FILE_PATH, file_type, file_name) 
    
    if file_name[-3:] == "pdf":
        loader = UnstructuredPDFLoader(document_path)
        doc = loader.load() 
        token_dict = LLM_Module.get_token_num(doc[0].page_content)
        
    elif file_name[-3:] == "txt":
        with open(document_path) as f :
            doc = f.read()
        token_dict = LLM_Module.get_token_num(doc)
                
    return  gr.Textbox(value=file_name), gr.Textbox(value=token_dict["text"], max_lines=16)

def uploaded_data_selected(
        file_input
    ) :
    print(file_input)
    file_name = file_input.split("/")[-1]
    document_path = file_input 
    
    if file_name[-3:] == "pdf":
        loader = UnstructuredPDFLoader(document_path)
        doc = loader.load() 
        token_dict = LLM_Module.get_token_num(doc[0].page_content)
        
    elif file_name[-3:] == "txt":
        with open(document_path) as f :
            doc = f.read()
        token_dict = LLM_Module.get_token_num(doc)
                
    return  gr.Textbox(value=file_name), gr.Textbox(value=token_dict["text"], max_lines=16)