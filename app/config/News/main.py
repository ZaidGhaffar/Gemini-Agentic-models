import google.generativeai as genai
import os
from dotenv import load_dotenv
from colorama import Fore
from pydantic import BaseModel
import json
from typing import Optional
from RAG_documents import RAG
load_dotenv()



class NewpaperSummarizer(BaseModel):
    title :str
    summary : list[str]
    keywords : list[str]
    source:str


class Recipe(BaseModel):
    recipe_name : str
    ingredients: list[str]
    
    
    
    
    
    
    
    
    


class GeminiModle:
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        # Create the model
        generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
        
        }

        model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=generation_config,
        )

        self.chat_session = model.start_chat(
        history=[
        ]
        )
        file_path = r"C:\python\AI-Agents\app\utils\Dark_web.pdf"
        api_key = os.getenv("GEMINI_API_KEY")

        self.obj = RAG(file_path, api_key)
        
    def RAG_Caller(self,text):
        self.obj.load_embeddings()
        if self.obj.faiss_index is None:
            docs = self.obj.document_loader()
            chunks = self.obj.text_splitter(docs)
            self.obj.storing_embeddings(chunks)
            
        result = self.obj.query_ans(text)
        data = f"here are some of the information that might help you to answer {result}"
        return data
        
        
        
        
        
    
    def model_response(self,text):
        response = self.chat_session.send_message(text)
        return response.text
        # try:
        #     # Parse the JSON response
        #     data = json.loads(response.text)
        #     # Map the JSON data to the Pydantic model
        #     summaries = [NewpaperSummarizer(**item) for item in data]
        #     return summaries
        # except Exception as e:
        #     print(f"Error parsing response: {e}")
        #     return None
        # return response.text

        
    
    
    
    
if __name__ == "__main__":
    obj = GeminiModle()
    while True:
        user_input = input("Please Enter your Prompt Here:        ").strip()
        if user_input:
            result = obj.model_response(user_input)
            # if result:
            #     for summary in result:
            #         print(summary.title+ "\n")    
            #         print(summary.summary)
                    
            print(f"Assistant:  {Fore.LIGHTGREEN_EX+ str(result)}")
            

        else:
            print("Please Enetr user input 🥪")
        
    