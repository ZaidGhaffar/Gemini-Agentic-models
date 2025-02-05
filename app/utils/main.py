import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel
import json


class Recipe(BaseModel):
    recipe_name : str
    ingredients: list[str]




class Gemini_model:
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINNI_API_KEY"))

        # Create the model
        generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
        "response_schema": list[Recipe]
        }

        model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=generation_config,
        
        )

        self.chat_session = model.start_chat(
        history=[
        ]
        )
        
    def model_reponse(self,text="List a few popular cookie recipes."):
        response = self.chat_session.send_message(text)
        try:
            data = json.loads(response.text)
            recipes = [Recipe(**item) for item in data]
            return recipes
        except Exception as e:
            print("Sorry an Error occured ")
        
        return None
    
        
    
    
    
if __name__ == "__main__":
    obj = Gemini_model()
    while True:
        user_input = input("please Enter your prompt here :               ")
        result = obj.model_reponse()
        if result :
            for response in result:
                print(f"Recipe Names are  : {response.recipe_name}")
        print(result)
        
        

        



