from transformers import pipeline
from PIL import Image
import streamlit as st
from gtts import gTTS
import os
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from translate import Translator

# Load environment variables
load_dotenv(find_dotenv())

# Define the Hugging Face Hub API Token and OpenAI API Key
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Create OpenAI client (new API format)
client = OpenAI(api_key=OPENAI_API_KEY)

# Language code mapping
LANGUAGE_CODES = {
    'English': 'en',
    'Telugu': 'te',
    'Hindi': 'hi',
    'Tamil': 'ta',
    'Malayalam': 'ml'
}

# Function to extract text from an image using Hugging Face model
def img2text(image_path):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(image_path)[0]['generated_text']
    print(text)
    return text

# Function to generate a story based on a scenario using OpenAI's GPT-3.5-turbo
# MODIFIED: Using free version - caption becomes the story
def generate_story(scenario):
    # ℹ️ FREE VERSION: Uses caption as story (no OpenAI payment needed)
    # To enable AI story generation, add payment to OpenAI account
    
    # Expand the caption into a simple 3-paragraph format
    story = f"""Scene One: {scenario}

In this moment, we observe: {scenario}. The details are vivid and clear, 
captured in time.

The scene continues to unfold naturally, with all elements working in harmony. 
Each detail contributes to the overall beauty of what we witness here.

As time moves forward, this scene will be remembered. It stands as a testament 
to the power of observation and appreciation for the world around us."""
    
    print(story)
    return story

# Function to translate text into a specified language
def translate_text(text, language):
    """Translate text to the specified language using LibreTranslate API"""
    # For English, return text as-is
    if language == 'English' or language == 'en':
        return text
    
    try:
        # Get the language code
        lang_code = LANGUAGE_CODES.get(language, 'en')
        if lang_code == 'en':
            return text
        
        print(f"🌐 Translating to {language} ({lang_code})...")
        
        # Create translator instance for the target language
        translator = Translator(to_lang=lang_code)
        
        # Translate the text
        translated = translator.translate(text)
        
        if translated and isinstance(translated, str):
            print(f"✅ Translation successful!")
            return translated
        else:
            print(f"⚠️ Translation failed, returning original text")
            return text
    
    except Exception as e:
        print(f"❌ Translation error for language {language}: {str(e)}")
        return text

# Streamlit application
def main():
    st.set_page_config(
        page_title="Image to Audio Story",
        page_icon="📸🔊"
    )

    st.markdown("<h1 style='text-align: center; margin-top: -30px;'>Transform Images to Audio Stories 🖼️➡️🔊</h1>", unsafe_allow_html=True)
    
    # Info banner
    st.info("✨ **FREE VERSION**: Uses AI vision + text-to-speech. Upgrade to unlock GPT story generation!")

    st.sidebar.title("🎨 Upload Your Image!")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.sidebar.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        scenario = img2text(uploaded_file.name)

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
        translated_scenario = translate_text(scenario, language)
        print(f"Original Caption: {scenario}")
        print(f"Translated Caption: {translated_scenario}\n")
        
        # Generate story from translated caption
        story = generate_story(translated_scenario)
        
        # Translate the full story
        translated_story = translate_text(story, language)
        print(f"Original Story:\n{story}")
        print(f"\nTranslated Story:\n{translated_story}\n")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("📸 Image Analysis"):
                st.write("**AI Vision Analysis:**")
                st.info(f"📝 **{language}:** {translated_scenario}")
        
        with col2:
            with st.expander("📖 Generated Story"):
                st.write(f"**Story in {language}:**")
                st.info(translated_story)

        st.write("---")
        st.subheader("🎵 Audio Story")
        st.markdown(f"**🎤 Generating audio in {language}...**")
        
        try:
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

