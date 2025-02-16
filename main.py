from fastapi import FastAPI, Query, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from fastembed import TextEmbedding, ImageEmbedding
from groq import Groq
import requests
import tempfile
import os
import base64
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
import requests
import os
from fastapi.middleware.cors import CORSMiddleware
import json


# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific frontend URL in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Ensure OPTIONS is allowed
    allow_headers=["*"],
)

# Load API keys securely (consider environment variables for production use)
GROQ_API_KEY = "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"  
AZURE_API_KEY = "EOflF5TbnuIB1zgZ7w6iYIS06X4uteo1imZ3fLx5r3p0P8DDwiWGJQQJ99BBACYeBjFXJ3w3AAAAACOG2q4I"  
# Configuration
AZURE_WHISPER_ENDPOINT = "https://didhi-m76j4vs0-eastus2.cognitiveservices.azure.com/openai/deployments/whisper/audio/translations?api-version=2024-06-01"
AZURE_EMBEDDING_ENDPOINT = "https://aragen-demo.cognitiveservices.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15"
AZURE_API_KEY = "5sxH5vswXOrIkPqkulpiJwIEfn68RabBIjLqXu1vrqYhB5WbCp3qJQQJ99BBACHYHv6XJ3w3AAAAACOGUKK6"

class TranslationEmbeddingResponse(BaseModel):
    text: str
    text_embedding: List[float]

# Initialize AI models
text_embedding_model = TextEmbedding()
image_embedding_model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
groq_client = Groq(api_key=GROQ_API_KEY)

print("Models initialized successfully!")

# Pydantic model for text input
class QueryRequest(BaseModel):
    text: str

# 1️⃣ **On-Prem Text Embedding Endpoint**
@app.get("/get-on-prem-text_embedding")
def get_text_embedding(text: str = Query(..., description="Input text to embed")):
    embeddings_generator = text_embedding_model.embed([text])
    embedding_vector = list(embeddings_generator)[0]
    return {"embedding": embedding_vector.tolist()}

# 2️⃣ **On-Prem Image Embedding Endpoint**
@app.post("/image_embedding")
async def get_image_embedding(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(await file.read())
        image_path = tmp.name

    embeddings = list(image_embedding_model.embed([image_path]))[0]
    os.remove(image_path)  # Clean up temp file
    return {"embedding": embeddings.tolist()}


# 4️⃣ **Azure Text Embedding Endpoint**
# Define the request body schema for the user query
class Queryy(BaseModel):
    text: str

# Endpoint to get embedding for a user query from azure
@app.post("/get-azure-embedding/")
async def get_embedding(query: Queryy):
    url = "https://aragen-demo.cognitiveservices.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15"
    headers = {
        "Content-Type": "application/json",
        "api-key": "EOflF5TbnuIB1zgZ7w6iYIS06X4uteo1imZ3fLx5r3p0P8DDwiWGJQQJ99BBACYeBjFXJ3w3AAAAACOG2q4I"
    }
    
    # Prepare the payload to send
    payload = {
        "input": query.text
    }

    try:
        # Make the API request to get embeddings
        response = requests.post(url, headers=headers, json=payload)

        # Check if the response is successful
        if response.status_code == 200:
            # Parse the embedding from the response
            embedding = response.json().get("data", [{}])[0].get("embedding")
            if embedding:
                return {"embedding": embedding}
            else:
                raise HTTPException(status_code=400, detail="Embedding not found in response")
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to get embedding from API")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request to API failed: {str(e)}")

# 5️⃣ **Image Description with LLaMA Vision + Embedding**
@app.post("/advanced_image_embedding")
async def advanced_image_embedding(file: UploadFile = File(...), text_prompt: str = Form(...)):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(await file.read())
        image_path = tmp.name

    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    messages = [{"role": "user", "content": [
        {"type": "text", "text": text_prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ]}]

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=messages,
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        output_text = completion.choices[0].message.content
        image_vector = list(image_embedding_model.embed([image_path]))[0].tolist()
        text_vector = list(text_embedding_model.embed([output_text]))[0].tolist()

        os.remove(image_path)  # Cleanup temp file
        return {"image_metadata": output_text, "image_vector_embedding": image_vector, "image_text_embedding": text_vector}

    except Exception as e:
        os.remove(image_path)  # Cleanup temp file
        raise HTTPException(500, str(e))

# 6️⃣ **Image QNA* with chat feature*
@app.post("/Image_QNA")
async def describe_image(
    file: UploadFile = None,
    text_prompt: str = Form(...),
    chat_history: str = Form(default="[]")
):
    """Process an image and a text prompt using Groq's LLaMA Vision model with chat history."""
    
    try:
        # Parse chat history from JSON string
        chat_messages = json.loads(chat_history)
        
        # Initialize messages list (without system message)
        messages = []

        # Only process new image if provided
        if file:
            try:
                # Save the uploaded image to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    content = await file.read()
                    tmp.write(content)
                    image_path = tmp.name

                # Convert the image to base64 format
                with open(image_path, "rb") as img_file:
                    base64_image = base64.b64encode(img_file.read()).decode("utf-8")

                # Add the image context in the first user message
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                })

                # Cleanup the temporary file
                os.remove(image_path)
            except Exception as e:
                if 'image_path' in locals():
                    try:
                        os.remove(image_path)
                    except:
                        pass
                raise e
        else:
            # Add previous chat history
            for msg in chat_messages:
                if msg["role"] == "user":
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": msg["content"]}]
                    })
                else:
                    messages.append({
                        "role": "assistant",
                        "content": msg["content"]
                    })
            
            # Add the current user prompt
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": text_prompt}]
            })

        # Call Groq API
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=messages,
            temperature=0.7,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        # Extract response text
        output_text = completion.choices[0].message.content

        return {
            "response": output_text,
            "role": "assistant"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




async def get_embedding(text: str) -> List[float]:
    """Get embedding from Azure's text embedding model"""
    headers = {
        "Content-Type": "application/json",
        "api-key": "EOflF5TbnuIB1zgZ7w6iYIS06X4uteo1imZ3fLx5r3p0P8DDwiWGJQQJ99BBACYeBjFXJ3w3AAAAACOG2q4I"
    }
    payload = {"input": text}
    response = requests.post(AZURE_EMBEDDING_ENDPOINT, headers=headers, json=payload)
    
    if response.status_code == 200:
        embedding = response.json().get("data", [{}])[0].get("embedding")
        if embedding:
            return embedding
    raise HTTPException(response.status_code, "Failed to get embedding")

@app.post("/translate_and_embed/", response_model=TranslationEmbeddingResponse)
async def translate_and_embed(audio_file: UploadFile = File(...)):
    """
    Endpoint to translate audio and get embeddings.
    Returns translated text, language, and text embedding.
    """
    if not audio_file.content_type.startswith('audio/'):
        raise HTTPException(400, "File must be an audio file")

    try:
        # Read file content
        file_content = await audio_file.read()
        
        # First get the translation
        headers = {"api-key": AZURE_API_KEY}
        files = {'file': (audio_file.filename, file_content, audio_file.content_type)}
        
        translation_response = requests.post(
            AZURE_WHISPER_ENDPOINT, 
            headers=headers, 
            files=files
        )
        
        if translation_response.status_code != 200:
            raise HTTPException(
                status_code=translation_response.status_code,
                detail=f"Azure translation API error: {translation_response.text}"
            )
            
        translation_result = translation_response.json()
        translated_text = translation_result.get("text", "")
       
        
        # Then get the embedding for the translated text
        text_embedding = await get_embedding(translated_text)
        
        return TranslationEmbeddingResponse(
            text=translated_text,
            
            text_embedding=text_embedding
        )
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(500, f"Failed to communicate with Azure API: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Internal server error: {str(e)}")




# Run the FastAPI app
# uvicorn main:app --reload