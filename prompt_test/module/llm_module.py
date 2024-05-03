#-*- coding:utf-8 -*-
import re
import requests
import json
import unicodedata
from typing import Union
from config.llm_url import (
    MODEL_INFERENCE_URL, 
    TOKEN_URL, 
    CHUNK_URL, 
    SET_SPLITTER_URL
)
from config.path import EXAMPLE_FILE_PATH

from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint 
import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


class LLM_Module :
    @classmethod
    def get_token_num(cls, input_text) : 
        input_text = cls.remove_special_char(input_text)                               
        response = requests.post(
                url=TOKEN_URL,
                data=json.dumps({"input_text" : input_text})
                )
        data_dict = eval(response.text)
        data_dict["text"] = input_text
        
        return data_dict


    @classmethod
    def remove_special_char(cls, text) :
        text = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", str(text))
        return ' '.join(text.split())
    
    
    @classmethod
    def set_splitter_option(cls, chunk_size, chunk_overlap) :
        response = requests.post(
                url=SET_SPLITTER_URL,
                data=json.dumps({"chunk_size" : chunk_size, "chunk_overlap" : chunk_overlap})
                )
        return response.text
    
    
    @classmethod
    def get_chunk_text(cls, text) :
        response = requests.post(
                url=CHUNK_URL,
                data=json.dumps({"input_text" : text})
                )
        data_dict = eval(response.text)
        temp = []
        for chunk in data_dict["chunk_text"] :
            remove_pront_num = 0
            remove_back_num = 0
            for i in range(0, 3) :
                if unicodedata.name(chunk[i]) == "REPLACEMENT CHARACTER" :
                    remove_pront_num += 1                                
            for i in range(3, 0, -1) :
                if unicodedata.name(chunk[-i]) == "REPLACEMENT CHARACTER" :
                    remove_back_num += 1
            
            if remove_back_num != 0 :
                temp.append(chunk[remove_pront_num:-remove_back_num])
            else :
                temp.append(chunk[remove_pront_num:])
            
        data_dict["chunk_text"] = temp                
        return data_dict
        
    
    
    @classmethod
    def llm_inference_api(cls, text, input_dict) :
        input_dict["prompt"] = text        
        response = requests.post(
                url=MODEL_INFERENCE_URL,
                data=json.dumps(input_dict)
                )
        data_dict = eval(response.text)        
        return data_dict["text"]   
    
    
    @classmethod
    def make_prompt(cls, instruction, chunk, params_dict) :   
        input_text = instruction +"\n <문서>"+ chunk +"\n</문서>"  
        re_gen_num = 0             
                
        while True :            
            generated_text = cls.llm_inference_api(input_text, params_dict)[0]        
            re_gen_num += 1                            
            if re_gen_num == 2 or generated_text != "" :
                break
                                    
        return ' '.join(generated_text.split()) 


    @classmethod
    def make_chunk_summary(cls, summary_instruction : str, chunks : list, params_dict : dict) :
        #generate chunk summary        
        summary_list = []
        summary_document = """"""
        for i, chunk in enumerate(chunks) :
            if len(chunk) < 5 :
                continue                        
            generated_text = cls.make_prompt(summary_instruction, chunk, params_dict) 
            summary_list.append([chunk, generated_text])            
            summary_document += generated_text + "\n\n"            
            
        summary_document = re.sub("\*\*요약\*\*", "", summary_document)      
        summary_document = re.sub("\*\*요약:\*\*", "", summary_document)
        return summary_list, summary_document
    
    
    @classmethod
    def summary_module(        
            cls,
            map_template : str,
            reduce_template : str,
            file_name : str,
            file_type : str,
            summary_max_token_num : int,
            params_dict : dict,
            chunk_size : int,
            chunk_overlap : int,
            file_input: str
        ) :
        def txt_to_pdf(txt_file_path, pdf_file_path):
            c = canvas.Canvas(pdf_file_path, pagesize=letter)
            width, height = letter 
            
            with open(txt_file_path, 'r') as file:
                lines = file.readlines()

            y_position = height - 72  
            for line in lines:
                c.drawString(72, y_position, line.strip())  
                y_position -= 12  

                if y_position < 72:
                    c.showPage()  
                    y_position = height - 72  

            c.save()
        
        # map
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
        repo_id = "google/gemma-1.1-7b-it"
        llm = HuggingFaceEndpoint(
            repo_id=repo_id, max_length=params_dict["max_tokens"], temperature=params_dict["temperature"]
        ) 
        
        
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=llm, prompt=map_prompt)
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)


        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )

        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=summary_max_token_num,
        )


        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="docs",
            return_intermediate_steps=False,
        )

        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        if file_input==None:
            if file_name[-3:] == "pdf":
                document_path = os.path.join(EXAMPLE_FILE_PATH, file_type, file_name)
                loader = UnstructuredPDFLoader(document_path)
                doc = loader.load() 
            elif file_name[-3:] == "txt":
                document_path = os.path.join(EXAMPLE_FILE_PATH, "custom_text.txt")
                txt_to_pdf(document_path, document_path.replace("txt","pdf"))
                loader = UnstructuredPDFLoader(document_path.replace("txt","pdf"))
                doc = loader.load() 
        
        else:
            file_name = file_input.split("/")[-1]
            document_path = file_input 
            
            if file_name[-3:] == "pdf":
                loader = UnstructuredPDFLoader(document_path)
                doc = loader.load() 
                
            elif file_name[-3:] == "txt":
                txt_to_pdf(document_path, document_path.replace("txt","pdf"))
                loader = UnstructuredPDFLoader(document_path.replace("txt","pdf"))
                doc = loader.load() 
                        
        split_texts = text_splitter.split_documents(doc)
        summary = map_reduce_chain.run(split_texts)
        
        return summary
    
    
    @classmethod
    def test_module(cls, instruction : str, summary_document : str, params_dict : dict) :
        max_gen_num = 4
        gen_num = 0
        while True :            
            gen_data = cls.make_prompt(instruction, summary_document, params_dict)
            if max_gen_num == gen_num or len(gen_data) >=3 :
                break
            gen_num += 1
            
        gen_data = gen_data.replace("<start_of_turn>model", "")
        gen_data = gen_data.replace("<eos>", "")
        
        return gen_data