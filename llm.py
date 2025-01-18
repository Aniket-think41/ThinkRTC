# llm.py
import os
from dotenv import load_dotenv
from openai import OpenAI
import logging
import asyncio

logging.basicConfig(level=logging.INFO)

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

async def get_llm_response(prompt: str):
    logging.info(f"LLM called with prompt: {prompt}")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        print("\nStreaming LLM response:")
        for chunk in response:  # Changed to async for
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)  # Print each chunk as it comes
                yield content
                
    except Exception as e:
        error_msg = f"Error in streaming response: {e}"
        logging.error(error_msg)
        yield error_msg