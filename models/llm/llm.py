import os
import json
from dotenv import load_dotenv
from llama_index.llms.ibm import WatsonxLLM
from llama_index.llms.google_genai import GoogleGenAI
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames


# Load environment variables from .env file
load_dotenv()

# Get Watsonx credentials
# api_key = os.getenv("WATSONX_APIKEY")
api_key = os.getenv("GOOGLE_API_KEY")

url = os.getenv("WATSONX_URL")
project_id = os.getenv("WATSONX_PROJECT_ID")
params_str = os.getenv("PARAMS", '{"decoding_method": "greedy", "max_new_tokens": 1000}')

# Validate required credentials
if not api_key:
    raise ValueError("WATSONX_APIKEY environment variable is not set")
if not url:
    raise ValueError("WATSONX_URL environment variable is not set")
if not project_id:
    raise ValueError("WATSONX_PROJECT_ID environment variable is not set")



# Initialize Watsonx LLM
watsonx_llm = WatsonxLLM(
    model_id="ibm/granite-4-h-small",
    ibm_cloud_api_key=api_key,
    ibm_cloud_url=url,
    project_id=project_id,
    temperature = 0.3,
    max_new_tokens = 2000,
    additional_prams = {
        GenTextParamsMetaNames.DECODING_METHOD: "sample"
    }
)

google_llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=api_key,
    temperature= 0.3,
    context_window=1000000,
    max_new_tokens=3000,
    top_k=10
)

# res = watsonx_llm.complete("Give me hi reply")
# print("Watsonx llm call", res)



