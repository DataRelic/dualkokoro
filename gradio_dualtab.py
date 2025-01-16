import gradio as gr
import os
import uuid
import wave
import json
import time
import shutil
import torch
import numpy as np
import requests
import base64
import cv2
from typing import List, Tuple, Dict, Optional, Union

# Environment setup for Windows
if os.name == 'nt':
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"

# Local imports (after environment setup)
from models import build_model
from kokoro import generate

# Model Configuration
SAMPLE_RATE = 24000
MAX_CHUNK_LENGTH = 300
DEFAULT_DURATION = 5.0
WORDS_PER_MINUTE = 130

# API Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
CHAT_MODEL = "stablelm-zephyr"
VISION_MODEL = "llama3.2-vision"

# Vision Analysis Prompt
VISION_PROMPT = """Describe this image in natural, conversational language. Focus on the following aspects:

1. Main Subject: What is the primary focus of the image? Describe its appearance, position, and notable features.

2. Visual Details:
   - Colors and color schemes present
   - Lighting conditions and atmosphere
   - Textures and patterns
   - Spatial arrangement and composition

3. Context and Setting:
   - Location or environment
   - Time of day or season if apparent
   - Any relevant background elements
   - Relationship between different elements

4. Notable Elements:
   - Secondary subjects or objects
   - Interesting details that stand out
   - Any text or signage visible
   - Actions or movements captured

Provide your description in clear, natural sentences. Avoid using any special characters, symbols, or formatting marks. Focus on creating a flowing, detailed narrative that someone could easily listen to."""

# Initialize model and device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
MODEL = build_model('kokoro-v0_19.pth', device)

# Voice loading and caching
voice_cache = {}

def load_voice(voice_name):
    """Load voicepack if not already in cache."""
    if voice_name not in voice_cache:
        voice_cache[voice_name] = torch.load(
            f'voices/{voice_name}.pt',
            weights_only=True
        ).to(device)
    return voice_cache[voice_name]

def chunk_text(text: str, max_length: int = 250) -> List[str]:
    """
    Split text into smaller chunks at sentence boundaries.
    Args:
        text: The text to split
        max_length: Maximum length of each chunk
    Returns:
        List of text chunks
    """
    import re
    sentences = re.split(r'([.!?])', text)
    chunks = []
    current_chunk = ""

    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i] + sentences[i + 1]
        if len(current_chunk) + len(sentence) > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += (" " + sentence) if current_chunk else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # If there's an odd leftover piece
    if len(sentences) % 2 != 0:
        leftover = sentences[-1].strip()
        if leftover:
            chunks.append(leftover)

    return [c for c in chunks if c]

class PodcastSession:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.script_dir = os.path.join('temp_scripts', self.session_id)
        self.audio_dir = os.path.join('temp_audio', self.session_id)
        os.makedirs(self.script_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        self.segments: List[Dict] = []
        self.debug_logs: List[str] = []
        self.script: str = ""  # Store the generated script

    def log(self, message: str):
        """Add a timestamped log message."""
        timestamp = time.strftime("%H:%M:%S")
        self.debug_logs.append(f"[{timestamp}] {message}")

    def save_segment(self, speaker: str, text: str, index: int) -> str:
        """Save a script segment to file and return its path."""
        # Strip any remaining markdown formatting from the text
        text = text.replace('*', '').replace('[', '').replace(']', '')
        
        filename = f"{index:03d}_{speaker}.txt"
        filepath = os.path.join(self.script_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)

        self.segments.append({
            'index': index,
            'speaker': speaker,
            'text_file': filepath,
            'audio_file': os.path.join(self.audio_dir, f"{index:03d}_{speaker}.wav")
        })
        self.log(f"Saved segment {index} for {speaker}")
        return filepath

    def save_audio_segment(self, audio: np.ndarray, index: int):
        """Save an audio segment as WAV file."""
        segment = next(s for s in self.segments if s['index'] == index)
        with wave.open(segment['audio_file'], 'wb') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes((audio * 32767).astype(np.int16).tobytes())
        self.log(f"Saved audio for segment {index}")

    def get_debug_log(self) -> str:
        """Get all debug logs as a single string."""
        return "\n".join(self.debug_logs)

    def cleanup(self):
        """Remove temporary files."""
        shutil.rmtree(self.script_dir, ignore_errors=True)
        shutil.rmtree(self.audio_dir, ignore_errors=True)
        self.log("Cleaned up temporary files")

# Global session storage
current_session = None

def generate_podcast_script(topic: str, speaker1: str, speaker2: str, duration: float):
    """Generate a podcast script using Ollama."""
    global current_session
    current_session = PodcastSession()
    current_session.log(f"Generating script for topic: {topic}")
    current_session.log(f"Speakers: {speaker1} and {speaker2}")
    current_session.log(f"Target duration: {duration} minutes")

    # Calculate target word count based on duration and speaking rate
    total_words = int(duration * WORDS_PER_MINUTE)
    current_session.log(f"Target word count: {total_words} words")

    prompt = f"""Generate a natural conversation between two hosts discussing the following topic: {topic}

Important Specifications:
- Target Duration: {duration:.1f} minutes
- Target Word Count: {total_words} words (based on {WORDS_PER_MINUTE} words per minute)
- Format each exchange as:
  {speaker1}: [dialogue]
  {speaker2}: [dialogue]

Guidelines:
1. Aim for roughly {total_words} total words to fill {duration:.1f} minutes
2. Keep each speaking segment relatively short (1-5 sentences)
3. Balance speaking time between both hosts
4. Make the conversation natural and engaging
5. Separate each exchange with a blank line
6. Do not use any markdown formatting in the dialogue

Please generate a discussion that fits these specifications while maintaining a natural flow.
"""

    try:
        current_session.log("Sending request to Ollama API")
        current_session.log(f"API URL: {OLLAMA_API_URL}")
        current_session.log(f"Model: {CHAT_MODEL}")
        yield "", current_session.get_debug_log()

        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": CHAT_MODEL,
                "prompt": prompt,
                "stream": True
            },
            stream=True
        )
        response.raise_for_status()
        current_session.log("Connected to Ollama API, receiving response...")
        yield "", current_session.get_debug_log()

        script = ""
        last_log_time = time.time()
        log_interval = 2.0  # Log every 2 seconds
        
        for line in response.iter_lines():
            if line:
                json_response = json.loads(line.decode('utf-8'))
                if 'response' in json_response:
                    script += json_response['response']
                    
                    # Only log periodically
                    current_time = time.time()
                    if current_time - last_log_time >= log_interval:
                        current_session.log("Receiving text stream from Ollama...")
                        last_log_time = current_time
                        
                    yield script, current_session.get_debug_log()
                if json_response.get('done', False):
                    break

        current_session.log("Text generation complete, processing segments...")
        yield script, current_session.get_debug_log()

        # Process and save script segments
        blocks = script.split('\n\n')
        segment_count = 0
        word_count = 0

        for i, block in enumerate(blocks):
            block = block.strip()
            if not block:
                continue

            if block.startswith(f'{speaker1}:'):
                speaker = speaker1
                text = block[len(f'{speaker1}:'):].strip()
                word_count += len(text.split())
                current_session.save_segment(speaker, text, segment_count)
                current_session.log(f"Processed segment {segment_count}: {speaker} ({len(text.split())} words)")
                segment_count += 1
                yield script, current_session.get_debug_log()
            elif block.startswith(f'{speaker2}:'):
                speaker = speaker2
                text = block[len(f'{speaker2}:'):].strip()
                word_count += len(text.split())
                current_session.save_segment(speaker, text, segment_count)
                current_session.log(f"Processed segment {segment_count}: {speaker} ({len(text.split())} words)")
                segment_count += 1
                yield script, current_session.get_debug_log()

        current_session.log(f"Generated script with {word_count} words in {segment_count} segments")
        current_session.log(f"Estimated duration: {word_count/WORDS_PER_MINUTE:.1f} minutes")
        current_session.log("Successfully generated and processed script")
        yield script, current_session.get_debug_log()

    except Exception as e:
        error_msg = f"Error generating script: {str(e)}"
        current_session.log(error_msg)
        raise gr.Error(error_msg)

def generate_podcast_audio(script: str, speaker1: str, speaker2: str):
    """Generate audio for the podcast script."""
    global current_session
    if not current_session or not current_session.segments:
        raise gr.Error("No active session. Please generate a script first.")

    current_session.log("Starting audio generation")
    try:
        all_audio = []
        
        # Process segments in order
        total_segments = len(current_session.segments)
        current_session.log(f"Processing {total_segments} total segments")
        
        for segment in sorted(current_session.segments, key=lambda x: x['index']):
            speaker = segment['speaker']
            current_session.log(f"Processing segment {segment['index']+1}/{total_segments} for {speaker}")
            yield None
            
            # Read the text from saved file
            with open(segment['text_file'], 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Generate audio for segment
            try:
                voicepack = load_voice(speaker)
                chunks = chunk_text(text)
                segment_audio = []

                for i, chunk in enumerate(chunks):
                    current_session.log(f"Generating audio for chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
                    yield None
                    
                    audio, _ = generate(MODEL, chunk, voicepack, lang=speaker[0])
                    if torch.is_tensor(audio):
                        audio = audio.cpu().numpy()
                    segment_audio.append(audio)
                    current_session.log(f"Generated {len(audio)} samples for chunk {i+1}")

                segment_audio = np.concatenate(segment_audio)
                all_audio.append(segment_audio)
                current_session.save_audio_segment(segment_audio, segment['index'])
                current_session.log(f"Completed segment {segment['index']+1}/{total_segments} with {len(segment_audio)} samples")
                yield None

            except Exception as e:
                error_msg = f"Error processing segment {segment['index']}: {str(e)}"
                current_session.log(error_msg)
                raise gr.Error(error_msg)

        final_audio = np.concatenate(all_audio)
        current_session.log(f"Successfully concatenated all segments into final audio: {len(final_audio)} samples")
        yield (SAMPLE_RATE, final_audio), current_session.get_debug_log()

    except Exception as e:
        error_msg = f"Error generating podcast audio: {str(e)}"
        current_session.log(error_msg)
        raise gr.Error(error_msg)

# Available voices for the podcast
ALL_VOICES = [
    'af',       # Default (50-50 mix of Bella & Sarah)
    'af_bella', 'af_sarah',
    'am_adam', 'am_michael',
    'bf_emma', 'bf_isabella',
    'bm_george', 'bm_lewis',
    'af_nicole', 'af_sky',
]

def create_interface():
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.Tab("Podcast Generator"):
                gr.Markdown("# Podcast Generator")
                
                # Topic input
                topic_input = gr.Textbox(
                    label="Podcast Topic",
                    placeholder="Enter the topic for your podcast...",
                    lines=2
                )
                
                # Voice selection and duration
                with gr.Row():
                    with gr.Column():
                        speaker1 = gr.Dropdown(
                            choices=ALL_VOICES,
                            label="Speaker 1 Voice",
                            value='af_bella',
                            interactive=True
                        )
                    with gr.Column():
                        speaker2 = gr.Dropdown(
                            choices=ALL_VOICES,
                            label="Speaker 2 Voice",
                            value='bm_george',
                            interactive=True
                        )
                    with gr.Column():
                        duration = gr.Number(
                            label="Approx. Podcast Duration (minutes)",
                            value=5.0,
                            minimum=0.5,  # Allow shorter durations
                            maximum=60,
                            step=0.5,
                            interactive=True
                        )
                
                # Generate button (centered)
                with gr.Row():
                    with gr.Column(scale=1):
                        pass
                    with gr.Column(scale=2):
                        generate_btn = gr.Button("Generate Script", variant="primary")
                    with gr.Column(scale=1):
                        pass
                
                # Generated script output (editable)
                generated_script = gr.TextArea(
                    label="Generated Podcast Script",
                    placeholder="Your generated podcast script will appear here. Feel free to edit it...",
                    lines=10,
                    interactive=True
                )
                
                # Generate podcast button and audio player
                with gr.Row():
                    with gr.Column(scale=1):
                        pass
                    with gr.Column(scale=2):
                        generate_podcast_btn = gr.Button("Generate Podcast", variant="primary")
                    with gr.Column(scale=1):
                        pass
                
                # Audio output
                audio_output = gr.Audio(
                    label="Generated Podcast",
                    type="numpy",
                    interactive=False
                )
                
                # Debug output for podcast tab
                podcast_debug = gr.TextArea(
                    label="Debug Output",
                    placeholder="Debug messages and errors will appear here...",
                    interactive=False,
                    lines=5
                )
            
            with gr.Tab("VisionTTS"):
                gr.Markdown("# Vision TTS")
                
                # Image input methods
                gr.Markdown("### Input Image")
                with gr.Tabs():
                    with gr.Tab("Upload"):
                        image_input = gr.Image(
                            label="Upload Image",
                            type="filepath",
                            sources=["upload"]
                        )
                    
                    with gr.Tab("Webcam"):
                        webcam_input = gr.Image(
                            label="Take Photo",
                            type="filepath",
                            sources=["webcam"],
                            mirror_webcam=True
                        )

                gr.Markdown("### Preview")
                image_preview = gr.Image(
                    label="Selected Image",
                    interactive=False
                )
                
                # Add voice selection
                vision_voice_select = gr.Dropdown(
                    label="Narrator Voice",
                    choices=ALL_VOICES,
                    value='am_michael',
                    info="Select a voice for the image description"
                )
                
                # Generate button
                with gr.Row():
                    with gr.Column(scale=1):
                        pass
                    with gr.Column(scale=2):
                        vision_generate_btn = gr.Button("Generate Audio from Image", variant="primary")
                    with gr.Column(scale=1):
                        pass
                
                # Add text output for the analysis
                vision_text_output = gr.Textbox(
                    label="Image Analysis",
                    interactive=False,
                    lines=10,
                    show_copy_button=True
                )
                
                # Audio output
                vision_audio_output = gr.Audio(
                    label="Generated Audio",
                    type="numpy",
                    interactive=False
                )
                
                # Debug output for vision tab
                vision_debug = gr.TextArea(
                    label="Debug Output",
                    placeholder="Debug messages and errors will appear here...",
                    interactive=False,
                    lines=5
                )

                # Event handlers for VisionTTS
                # Update preview when image is uploaded or captured from webcam
                image_input.change(
                    fn=lambda x: x,
                    inputs=[image_input],
                    outputs=[image_preview],
                    queue=False
                )
                
                webcam_input.change(
                    fn=lambda x: x,
                    inputs=[webcam_input],
                    outputs=[image_preview],
                    queue=False
                )

                def encode_image_to_base64(image):
                    """Convert image to base64 string."""
                    try:
                        if isinstance(image, str):  # File path
                            with open(image, 'rb') as f:
                                return base64.b64encode(f.read()).decode('utf-8')
                        elif isinstance(image, np.ndarray):  # Numpy array
                            success, buffer = cv2.imencode('.jpg', image)
                            if not success:
                                raise ValueError("Failed to encode image")
                            return base64.b64encode(buffer).decode('utf-8')
                        else:
                            raise ValueError(f"Unsupported image type: {type(image)}")
                    except Exception as e:
                        raise gr.Error(f"Error encoding image: {str(e)}")

                def analyze_image(image, voice_name):
                    """Process the image and generate analysis with audio."""
                    try:
                        if image is None:
                            raise gr.Error("Please provide an image first")
                            
                        # Convert image to base64
                        base64_image = encode_image_to_base64(image)
                        
                        # Call vision API
                        try:
                            response = requests.post(
                                OLLAMA_API_URL,
                                json={
                                    "model": VISION_MODEL,
                                    "prompt": VISION_PROMPT,
                                    "stream": True,
                                    "images": [base64_image]
                                },
                                stream=True
                            )
                            response.raise_for_status()
                            
                            # Process streaming response
                            analysis_text = ""
                            for line in response.iter_lines():
                                if line:
                                    json_response = json.loads(line.decode('utf-8'))
                                    if 'response' in json_response:
                                        analysis_text += json_response['response']
                                    if json_response.get('done', False):
                                        break
                            
                            if not analysis_text.strip():
                                raise gr.Error("No analysis generated from the image")
                                
                        except requests.exceptions.RequestException as e:
                            raise gr.Error(f"Failed to connect to vision API: {str(e)}")
                            
                        # Generate audio from analysis
                        try:
                            voicepack = load_voice(voice_name)
                            chunks = chunk_text(analysis_text)
                            audio_chunks = []

                            for chunk in chunks:
                                audio, _ = generate(MODEL, chunk, voicepack, lang=voice_name[0])
                                if torch.is_tensor(audio):
                                    audio = audio.cpu().numpy()
                                audio_chunks.append(audio)

                            final_audio = np.concatenate(audio_chunks)
                            return (SAMPLE_RATE, final_audio), analysis_text, "Analysis completed successfully"

                        except Exception as e:
                            raise gr.Error(f"Error generating audio: {str(e)}")

                    except Exception as e:
                        error_msg = f"Error in image analysis: {str(e)}"
                        print(error_msg)  # For debugging
                        return None, None, error_msg

                # Connect vision generation button
                vision_generate_btn.click(
                    fn=analyze_image,
                    inputs=[image_preview, vision_voice_select],
                    outputs=[vision_audio_output, vision_text_output, vision_debug],
                    queue=True
                )

        # Event handlers for Podcast Generator
        def generate_script_wrapper(topic, speaker1, speaker2, duration):
            try:
                # Initialize progress
                progress_generator = generate_podcast_script(topic, speaker1, speaker2, float(duration))
                
                # Stream updates
                script = ""
                debug_log = ""
                for new_script, new_debug in progress_generator:
                    script = new_script if new_script else script
                    debug_log = new_debug
                    yield script, debug_log
                    time.sleep(0.1)  # Small delay to prevent UI freezing
                    
            except Exception as e:
                raise gr.Error(str(e))

        def generate_audio_wrapper(script, speaker1, speaker2):
            try:
                # Initialize progress
                progress_generator = generate_podcast_audio(script, speaker1, speaker2)
                
                # Stream updates
                for result in progress_generator:
                    if result is None:  # Progress update
                        yield None, current_session.get_debug_log()
                    else:  # Final result with audio
                        sample_rate, audio_data = result[0]  # Unpack the tuple
                        current_session.log(f"Final audio generated: {len(audio_data)} samples at {sample_rate}Hz")
                        yield result[0], result[1]  # Pass through the complete audio tuple and debug log
                        
            except Exception as e:
                error_msg = f"Error in audio wrapper: {str(e)}"
                if current_session:
                    current_session.log(error_msg)
                raise gr.Error(error_msg)

        # Connect event handlers with streaming
        generate_btn.click(
            fn=generate_script_wrapper,
            inputs=[topic_input, speaker1, speaker2, duration],
            outputs=[generated_script, podcast_debug],
            queue=True  # Enable queuing for stability
        )

        generate_podcast_btn.click(
            fn=generate_audio_wrapper,
            inputs=[generated_script, speaker1, speaker2],
            outputs=[audio_output, podcast_debug],
            queue=True  # Enable queuing for stability
        )

    return demo

if __name__ == "__main__":
    # Create temp directories if they don't exist
    for dir_name in ['temp_scripts', 'temp_audio', 'temp_images']:
        os.makedirs(dir_name, exist_ok=True)
        
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        ssl_certfile="localhost.crt",
        ssl_keyfile="localhost.key",
        ssl_verify=False
    )
