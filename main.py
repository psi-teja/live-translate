import whisper
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import pyaudio
import numpy as np
import threading
import queue
import traceback

class RealTimeTranslator:
    def __init__(self, source_lang="en", target_lang="es"):
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # Load Whisper for speech recognition
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")
        
        # Load NLLB for translation
        print("Loading translation model...")
        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-distilled-600M",
            dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-600M",
            src_lang=source_lang
        )
        
        # Move to GPU if available (M4 should handle this well)
        if torch.backends.mps.is_available():
            self.translation_model = self.translation_model.to('mps')
        
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def record_audio(self):
        """Record audio in real-time"""
        p = pyaudio.PyAudio()
        
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1600,  # 100ms chunks
            stream_callback=self.audio_callback
        )
        
        stream.start_stream()
        self.is_recording = True
        
        try:
            while self.is_recording:
                pass
        except KeyboardInterrupt:
            pass
        
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def process_audio(self):
        """Process audio and translate"""
        audio_buffer = []
        buffer_duration = 3  # seconds
        
        while self.is_recording:
            try:
                # Collect audio chunks
                while len(audio_buffer) < buffer_duration * 10:  # 10 chunks per second
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    audio_buffer.extend(audio_chunk)
                
                # Process audio
                audio_array = np.array(audio_buffer)
                
                # Transcribe
                result = self.whisper_model.transcribe(
                    audio_array,
                    language=self.source_lang,
                    fp16=torch.backends.mps.is_available()
                )
                
                text = result['text'].strip()
                if text:
                    # Translate
                    inputs = self.tokenizer(text, return_tensors="pt", padding=True)
                    
                    if torch.backends.mps.is_available():
                        inputs = {k: v.to('mps') for k, v in inputs.items()}
                    
                    translated_tokens = self.translation_model.generate(
                        **inputs,
                        forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target_lang],
                        max_length=500
                    )
                    
                    translation = self.tokenizer.batch_decode(
                        translated_tokens, 
                        skip_special_tokens=True
                    )[0]
                    
                    print(f"\nOriginal: {text}")
                    print(f"Translation: {translation}")
                    print("-" * 50)
                
                # Keep last 1 second for context
                audio_buffer = audio_buffer[-16000:]  # Keep last second
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                continue
    
    def start(self):
        """Start real-time translation"""
        print("Starting real-time translation...")
        print("Press Ctrl+C to stop")
        
        # Start recording thread
        record_thread = threading.Thread(target=self.record_audio)
        record_thread.daemon = True
        record_thread.start()
        
        # Start processing thread
        process_thread = threading.Thread(target=self.process_audio)
        process_thread.daemon = True
        process_thread.start()
        
        try:
            while True:
                pass
        except KeyboardInterrupt:
            self.is_recording = False
            print("\nStopping translation...")

# Usage
if __name__ == "__main__":
    translator = RealTimeTranslator(source_lang="eng_Latn", target_lang="spa_Latn")
    translator.start()