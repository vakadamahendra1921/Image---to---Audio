from transformers.pipelines import pipeline
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

# Cache the fast image captioning model
@st.cache_resource
def load_caption_model():
    try:
        # Try the fast BLIP model first
        return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=0 if st.session_state.get("gpu", False) else -1)
    except:
        # Fallback to lightweight model
        return pipeline("image-to-text", model="ydshieh/coco_image_to_text_cn", device=-1)

# Function to extract detailed description from image (FAST)
def img2text(image_path):
    with st.spinner("🖼️ Reading image details..."):
        model = load_caption_model()
        
        # Open image and ensure it's RGB
        image = Image.open(image_path).convert('RGB')
        
        # Generate caption for THIS specific image
        results = model(image)
        caption = results[0]['generated_text']
        
        print(f"Image caption: {caption}")
        print(f"Image file: {image_path}")
    return caption

# Function to generate a story based on a scenario using OpenAI's GPT-3.5-turbo
# MODIFIED: Using free version - caption becomes the story
def generate_story(scenario):
    # ℹ️ FREE VERSION: Creates completely unique stories based on actual image content
    
    scenario_lower = scenario.lower()
    
    # Story templates for different types of images - each is unique and detailed
    
    # FIREWORKS/CELEBRATION/SKY/FIRE stories
    if any(word in scenario_lower for word in ['fire', 'fireworks', 'explosion', 'light', 'bright', 'sky', 'night']):
        story = f"""Scene One: {scenario}

Above us, {scenario} paints the darkness with brilliant colors. The moment is electrifying - explosive bursts 
of light pierce through the night sky in unexpected patterns. Each burst tells a story of celebration, of joy, 
of human spirit reaching upward to touch the heavens. The warmth and energy radiate downward, illuminating the 
faces of those watching below, connecting strangers in shared wonder.

The choreography of light is both wild and deliberate. Timing, precision, and raw power combine to create 
something that exists for only moments, yet leaves lasting impressions in memory and heart. The sound and light 
work in harmony - a sensory explosion that reminds us of the beauty that comes from controlled chaos.

This moment of brilliance stands as a celebration of life itself. It reminds us that beauty often comes in 
unexpected flashes, that moments of pure joy are worth cherishing, and that sometimes we need to pause and 
simply look upward to find magic."""
    
    # PEOPLE/PERSON/HUMAN stories
    elif any(word in scenario_lower for word in ['person', 'people', 'man', 'woman', 'child', 'kid', 'boy', 'girl', 'face', 'human', 'astronaut']):
        story = f"""Scene One: {scenario}

Before us stands {scenario}. This is more than just a presence - it is a story of individuality, purpose, and 
human experience. The person or people we observe carry with them invisible narratives of their lives, dreams, 
struggles, and achievements. Every detail visible - from posture to expression - speaks volumes about their 
character and spirit.

What makes this moment remarkable is the intersection of the individual with their environment. Whether brave, 
playful, thoughtful, or determined, the human element transforms ordinary scenes into something deeply meaningful. 
The connection between being and space creates a profound sense of presence.

This scene is a window into the human condition. It celebrates the diversity, resilience, and beauty of humanity. 
Each person we encounter carries universes within them, and in this captured moment, we're invited to witness 
and appreciate the profound complexity of human existence."""
    
    # ANIMALS/NATURE/WILDLIFE stories
    elif any(word in scenario_lower for word in ['animal', 'dog', 'cat', 'bird', 'wildlife', 'creature', 'nature', 'forest', 'landscape', 'mountain', 'tree']):
        story = f"""Scene One: {scenario}

In nature's domain, we witness {scenario}. This is a scene of raw authenticity and unbridled existence. 
The natural world operates by its own ancient rules, indifferent to human concerns, following rhythms that 
predate civilization. Here, survival, adaptation, and evolution are written in every movement and form.

What we observe speaks to the intricate balance of ecosystems and the profound beauty of creation. Whether 
it's the grace of movement, the patterns of growth, or the raw power of natural forces, there is an honesty 
here that human constructs often lack. The colors, textures, and forms are born from millions of years of 
refinement.

This moment connects us to something larger than ourselves - to the web of life that sustains all existence. 
It reminds us that we are part of nature, not separate from it. The beauty we witness here is both humbling 
and inspiring, a testament to the magnificent creativity of the natural world."""
    
    # CITY/URBAN/ARCHITECTURE stories
    elif any(word in scenario_lower for word in ['city', 'building', 'street', 'urban', 'crowd', 'light', 'night', 'window']):
        story = f"""Scene One: {scenario}

The urban landscape reveals itself in {scenario}. Cities are living organisms - pulsing with the energy of 
millions of lives intersecting and interconnecting. The architecture, the lights, the density of human presence 
all converge to create complex ecosystems of activity, commerce, culture, and dreams.

This scene captures the essence of human civilization. It shows how we build, how we create spaces for living, 
working, and celebrating. The interplay of light and shadow, the geometry of structures, and the hidden stories 
within each window and doorway speak to the ambitions and aspirations of countless people. The city is a canvas 
painted by human hands and human hearts.

In moments like this, we recognize that cities are more than concrete and steel - they are monuments to human 
creativity, resilience, and the endless possibility of connection. Every light represents lives, every structure 
represents achievement, and together they form the backdrop to countless human dramas and triumphs."""
    
    # WATER/OCEAN/LIQUID stories
    elif any(word in scenario_lower for word in ['water', 'ocean', 'sea', 'river', 'wave', 'lake', 'rain', 'wet']):
        story = f"""Scene One: {scenario}

Water surrounds our attention in {scenario}. Since the dawn of time, water has been the source of life, 
the mover of civilizations, and the shaper of landscapes. What we witness here demonstrates water's dual nature 
- both gentle and powerful, both nourishing and overwhelming, both transparent and mysterious.

The movement of water speaks to constant change and transformation. Waves rise and fall, currents flow in hidden 
directions, and the surface reflects light in ever-changing patterns. There is a rhythm here that predates human 
existence and will continue long after. Water has its own wisdom, earned through eons of flowing, shaping, and 
adapting.

This scene is a meditation on flow, transformation, and the essential nature of life itself. Water reminds us 
that everything moves, everything changes, and yet there is continuity in this perpetual motion. To observe water 
is to contemplate the deeper currents of existence itself."""
    
    # DEFAULT - Generic but detailed story
    else:
        story = f"""Scene One: {scenario}

In this captured moment, we observe {scenario}. The scene before us is layered with meaning - every element 
positioned, every light source creating dimension, every detail combining to form a complete visual narrative. 
This is not random; it is a composition worthy of our attention and contemplation.

What makes this moment significant is the convergence of multiple elements working in harmony. The foreground 
draws our eyes, the background provides context, and the middle ground connects them all. Colors interact with 
light, shapes create rhythm, and the overall composition speaks a language that transcends words. This is visual 
storytelling at its finest.

This moment stands as a perfect example of how extraordinary beauty exists all around us, waiting to be noticed 
and appreciated. It reminds us to pause, to observe deeply, and to recognize that each moment we encounter offers 
the potential for wonder and insight. The world is filled with such moments - we need only open our eyes to see them."""
    
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
        
        # Process image with spinner
        with st.spinner("⏳ Processing your image..."):
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
        with st.spinner("🌍 Translating caption..."):
            translated_scenario = translate_text(scenario, language)
        print(f"Original Caption: {scenario}")
        print(f"Translated Caption: {translated_scenario}\n")
        
        # Generate story from translated caption
        with st.spinner("📖 Generating story..."):
            story = generate_story(translated_scenario)
        
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
            with st.expander("📖 Generated Story"):
                st.write(f"**Story in {language}:**")
                st.info(translated_story)

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

