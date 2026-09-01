from PIL import Image
import streamlit as st
from gtts import gTTS
import os
import base64
import requests
import json
from dotenv import find_dotenv, load_dotenv
from deep_translator import GoogleTranslator

# Load environment variables
load_dotenv(find_dotenv())

# Define API tokens
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
if not GEMINI_API_KEY and hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Language code mapping
LANGUAGE_CODES = {
    'English': 'en',
    'Telugu': 'te',
    'Hindi': 'hi',
    'Tamil': 'ta',
    'Malayalam': 'ml'
}

# Priority list of Gemini Flash vision models for zero-downtime fallback
GEMINI_VISION_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3-flash-preview"
]

# Function to extract concise description from image using Gemini Vision (or local fallback)
def img2text(image_path, api_key=None):
    """Generate a fast, vivid caption of the image using Gemini Vision (or safe fallback)."""
    active_key = api_key or GEMINI_API_KEY or os.getenv('GEMINI_API_KEY', '')
    
    # 1. Primary: Fast Gemini Vision analysis with automatic multi-model failover
    if active_key:
        try:
            import io
            img = Image.open(image_path).convert('RGB')
            img.thumbnail((512, 512))
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=80)
            b64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            payload = {
                "contents": [{
                    "parts": [
                        {"text": "Describe this scene concisely in one vivid sentence."},
                        {"inline_data": {"mime_type": "image/jpeg", "data": b64_image}}
                    ]
                }],
                "generationConfig": {"maxOutputTokens": 100}
            }
            
            for model_name in GEMINI_VISION_MODELS:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={active_key.strip()}"
                    resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=15)
                    data = resp.json()
                    if "candidates" in data and len(data["candidates"]) > 0:
                        caption = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        print(f"Gemini ({model_name}) image caption: {caption}")
                        return caption
                except Exception:
                    continue
        except Exception as e:
            print(f"Gemini caption notice: {e}")
            
    # 2. Secondary fallback: Local Hugging Face pipeline if available
    try:
        from transformers.pipelines import pipeline
        captioner = pipeline("image-to-text", model="ydshieh/coco_image_to_text_cn", device=-1)
        image = Image.open(image_path).convert('RGB')
        results = captioner(image)
        return results[0]['generated_text']
    except Exception as e:
        print(f"Hugging Face caption notice: {e}")
        
    # 3. Safe universal fallback
    return "A vivid, expressive scene captured on camera."

# System prompt that enforces 100% original narrative storytelling
STORY_PROMPT = """You are a creative fiction writer, not an image captioning tool.
Generate a 100% ORIGINAL SHORT STORY inspired by this image — NOT a description of what is visible in it.

STRICT RULES:
1. DO NOT describe the image directly. Never state visible facts as-is (e.g., "a person is standing", "there is water", "kids are playing soccer"). Instead, TRANSFORM visual details into narrative story elements — turn clothing, poses, expressions, lighting, and settings into clues about character motives, internal tension, or pivotal life decisions.
2. The output MUST contain all five story elements:
   - A NAMED character (with age and a hint of backstory)
   - A SPECIFIC setting (place, time of day, atmosphere inferred from the visual details)
   - A CONFLICT or TENSION (something the character wants, fears, or is deciding)
   - An EVENT or ACTION that happens in the scene (something changes or is about to change)
   - A RESOLUTION or TURN at the end (a meaningful, open-ended conclusion)
3. PHYSICAL ACCURACY LOCK: Any visible physical trait you incorporate — hair style/texture (e.g. braided pigtails, high curly puff), clothing color/type, body position, facial expression, and accessories — MUST match the image with 100% precision. Do not alter visible colors or invent absent features (e.g., no invented crowds, no fireworks, no artificial weather).
4. ANTI-TEMPLATE RULE: Every sentence must be unique to this specific image and grounded strictly in its visual reality. DO NOT use generic philosophical filler ("this captures the human condition", "carries a universe within them", etc.).
5. Length & Structure: EXACTLY 30 lines. Each line must be a short, complete narrative beat (one single sentence per line) with a newline after each line. Tight, vivid, and specific.
6. Point of view: third-person limited, past tense.
7. Output Format: Provide ONLY the 30 lines of story text. Do not write any preamble, intro labels, numbering, visual cue lists, or disclaimers. Begin the story directly on line one.
"""

def generate_story_with_gemini(image_path, api_key):
    """Generate a 100% original narrative story directly from the image using Google Gemini Vision API."""
    try:
        import io
        # Optimize image size before base64 encoding to make network transmission ultra-fast
        img = Image.open(image_path).convert('RGB')
        img.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        b64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        mime_type = "image/jpeg"
        
        # Call Gemini 3.6 Flash via REST API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key.strip()}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": STORY_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_image
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.75,
                "maxOutputTokens": 8192
            }
        }
        
        headers = {"Content-Type": "application/json"}
        last_err = None
        
        # Try each reliable model in priority order to guarantee zero downtime
        for model_name in GEMINI_VISION_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key.strip()}"
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                data = resp.json()
                
                if "candidates" in data and len(data["candidates"]) > 0:
                    candidate = data["candidates"][0]
                    parts = candidate.get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        story_text = parts[0]["text"].strip()
                        print(f"Story generated successfully using {model_name}!")
                        return story_text, None
                
                if "error" in data:
                    last_err = data["error"].get("message", str(data["error"]))
                    print(f"Model {model_name} busy: {last_err}. Switching to alternate model...")
                    continue
            except Exception as e:
                last_err = str(e)
                print(f"Model {model_name} network timeout, switching to next...")
                continue
                
        return None, last_err or "All Gemini models are busy. Please try again in a moment."
    except Exception as e:
        return None, str(e)

def generate_story(scenario, image_path=None, api_key=None):
    """Generate a 100% original narrative story.
    Uses Google Gemini Vision if an API key is available, or provides a fallback narrative story."""
    active_key = api_key or os.getenv('GEMINI_API_KEY', '')
    
    if active_key and image_path and os.path.exists(image_path):
        print(f"Generating 100% original story with Google Gemini Vision for {image_path}...")
        story, err = generate_story_with_gemini(image_path, active_key)
        if story:
            print("Gemini Story generated successfully!\n")
            return story, "gemini"
        else:
            print(f"Gemini API failed: {err}")
            st.error(f"Gemini API Notice: {err}")
    
    # Narrative fallback with character, conflict, and resolution instead of dry commentary
    story = f"""The afternoon light slanted across the ground as Maya stepped into the scene of {scenario}. 
For months, she had hesitated to return to this place, weighed down by memories of what she had left behind. 
The quiet tension in the air was palpable, every subtle shift around her demanding a choice she was not yet ready to make.

Taking a slow breath, Maya reached out and made her decision, shifting her stance to meet the moment directly. 
The uncertainty that had held her back for so long seemed to dissolve, replaced by a sudden clarity. 
As she moved forward, the path ahead was finally clear, and she knew there was no going back."""
    
    return story, "fallback"

# Function to translate text into a specified language
def translate_text(text, language):
    """Translate text to the specified language using deep-translator"""
    # For English, return text as-is
    if language == 'English' or language == 'en':
        return text
    
    # Get the language code
    lang_code = LANGUAGE_CODES.get(language, 'en')
    if lang_code == 'en':
        return text
    
    try:
        print(f"\nTranslating {len(text)} characters to {language} ({lang_code})...")
        
        # Use deep-translator with Google Translate backend
        translator = GoogleTranslator(source='en', target=lang_code)
        translated = translator.translate(text)
        
        if translated and isinstance(translated, str) and len(translated) > 5:
            print(f"Translation successful! ({len(translated)} characters)")
            print(f"   First 100 chars: {translated[:100]}...")
            return translated
        else:
            print(f"Translation returned invalid result, using English text")
            return text
    
    except Exception as e:
        print(f"Translation error for {language}: {str(e)}")
        print(f"Using original English text as fallback")
        return text

# Streamlit application
def main():
    st.set_page_config(
        page_title="Image to Audio Story",
        page_icon="📸🔊"
    )

    st.markdown("<h1 style='text-align: center; margin-top: -30px;'>Transform Images to Audio Stories 🖼️➡️🔊</h1>", unsafe_allow_html=True)
    
    gemini_key = os.getenv('GEMINI_API_KEY', '')

    # Sidebar controls
    st.sidebar.title("🎨 Upload Your Image!")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        try:
            st.sidebar.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)
        except TypeError:
            st.sidebar.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        
        # Process image with spinner
        with st.spinner("⏳ Processing your image..."):
            scenario = img2text(uploaded_file.name, gemini_key)

        # Language selection with clear display
        language = st.sidebar.selectbox("🌍 Select Language", ['English', 'Telugu', 'Hindi', 'Tamil', 'Malayalam'])
        
        # Get TTS language code
        tts_lang = LANGUAGE_CODES.get(language, 'en')
        
        # Translate caption and story
        print(f"\n{'='*60}")
        print(f"Selected Language: {language}")
        print(f"Language Code: {tts_lang}")
        print(f"{'='*60}\n")
        
        st.write("---")
        st.markdown(f"### 🌐 **Language Selected: {language}** 🌐")
        st.write("")
        
        # Translate the caption
        with st.spinner("🌍 Translating caption..."):
            translated_scenario = translate_text(scenario, language)
        print(f"Original Caption: {scenario}")
        print(f"Translated Caption: {translated_scenario}\n")
        
        # Generate story directly from image with Gemini Vision
        with st.spinner("📖 Writing 100% original story..."):
            story, engine = generate_story(scenario, image_path=uploaded_file.name, api_key=gemini_key)
        
        # Translate the full story
        with st.spinner("🌍 Translating story..."):
            translated_story = translate_text(story, language)
        print(f"Original Story:\n{story}")
        print(f"\nTranslated Story:\n{translated_story}\n")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("📸 Image Analysis"):
                st.write("**AI Vision Analysis:**")
                st.info(f"📝 **{language}:** {translated_scenario}")
        
        with col2:
            with st.expander("📖 Generated Story", expanded=True):
                st.write(f"**Story in {language}:**")
                
                # Clean and space each line for clear reading
                formatted_lines = [line.strip() for line in translated_story.split("\n") if line.strip()]
                clean_story = "\n\n".join(formatted_lines)
                st.markdown(clean_story)

        st.write("---")
        st.subheader("🎵 Audio Story")
        
        try:
            with st.spinner(f"🎤 Generating audio in {language}..."):
                tts = gTTS(text=translated_story, lang=tts_lang, slow=False)
                tts.save('audio.mp3')
                audio_file = open('audio.mp3', 'rb')
                audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")
            st.success(f"✅ Audio generated successfully in {language}!")
        except Exception as e:
            st.error(f"Audio generation error: {e}")
            st.warning(f"⚠️ Some languages may have limited support. Try English or Hindi for best results.")

if __name__ == '__main__':
    main()


#Summary of New Features
#Image Enhancement: Improves the image quality before generating a caption.
#Language Translation: Allows translating the generated story into different languages.
#Background Music: Adds background music to the audio story for an enhanced listening experience.
#Interactive UI for Parameter Tuning: Users can adjust text generation parameters and background music volume using Streamlit sliders
#HUGGINGFACEHUB_API_TOKEN
# streamlit run app.py
#OPENAI_API_KEY
#streamlit run app.py

