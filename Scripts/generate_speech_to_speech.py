import argparse
import tqdm
import soundfile as sf
import sys
import shutil
import json
import pandas as pd
import os
import re
import torch
import numpy as np

from pathlib import Path
from tqdm import tqdm
from inference import StyleTTS2Synthesizer, SAMPLING_RATE

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_metadata(dataset_path):
    return pd.read_csv(dataset_path)

def s2s(text: str, 
        language: str,
        source_style, 
        target_style, 
        output_filepath: str, 
        synthesizer: StyleTTS2Synthesizer, 
        speaker_config, 
        max_duration: float = 20.0,
        fadeout: float = 0.1,
        silence: float = 0.1,
        ):
    """Helper function to generate a single audio file"""
    try:
        wav = synthesizer.s2s(
            text,
            source_style,
            target_style,
            language,
            alpha=speaker_config["alpha"],
            beta=speaker_config["beta"],
            diffusion_steps=speaker_config.get("diffusion_steps", 10),
            phonemes=False
        )
        duration = len(wav) / SAMPLING_RATE
        if max_duration and duration > max_duration:
            print(f"Skipping audio with {speaker_config['name']} - Duration {duration:.2f}s exceeds limit of {max_duration}s")
            return False
        
        # Apply fadeout with configurable duration
        fadeout_samples = int(fadeout * SAMPLING_RATE)
        fadeout_window = np.linspace(1, 0, fadeout_samples)
        wav[-fadeout_samples:] *= fadeout_window
        
        # Add configurable silence duration
        silence_length = int(SAMPLING_RATE * silence)
        silence = np.zeros(silence_length)
        wav = np.concatenate([wav, silence])
        
        sf.write(str(output_filepath), wav, SAMPLING_RATE)
        # Speed up audio
        from audiostretchy.stretch import stretch_audio
        stretch_audio(output_filepath, output_filepath, ratio=0.85)
        return True
    except Exception as e:
        print(f"Error processing with {output_filepath}: {e}")
        return False

def generate_speech_to_speech(model_config_path, 
                              checkpoint_path, 
                              output_path, 
                              speaker_config_path, 
                              max_duration, 
                              limit,
                              fadeout,
                              silence):
    
    config = load_config(speaker_config_path)
    settings = config['speakers']
    synthesizer = StyleTTS2Synthesizer(config_path=model_config_path, checkpoint_path=checkpoint_path)
    output_path = Path(output_path)
    if output_path.exists():
        shutil.rmtree(output_path)
        
    wavs_dir = output_path / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    for speaker_config in settings:
        text_column = speaker_config["text_column"]
        audio_column = speaker_config["audio_column"]
        df = load_metadata(speaker_config["dataset_path"])
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {speaker_config['name']}"):
            if limit and idx >= limit:
                break
            try:
                text_column = speaker_config["text_column"]
                audio_column = speaker_config["audio_column"]
                text = row[text_column]
                
                output_filename = text
                if speaker_config["name"]:
                    output_filename = f"{speaker_config['name']}_{output_filename}"
                output_filepath = wavs_dir / f"{output_filename}.wav"
                
                path_to_source_audio = Path(speaker_config["root_path"]) / row[audio_column]
                path_to_reference_audio =  speaker_config["reference_path"]
                
                source_style = synthesizer.compute_style(path_to_source_audio)
                target_style = synthesizer.compute_style(path_to_reference_audio)
                
                language = speaker_config["language"]
                if language == "en-au":
                    language = "en-gb"

                if s2s(text, 
                    language,
                    source_style, 
                    target_style, 
                    output_filepath, 
                    synthesizer, 
                    speaker_config,
                    max_duration=max_duration,
                    fadeout=fadeout,
                    silence=silence):
                    metadata.append(f"{output_filepath}|{text}|{speaker_config['language']}")
            except Exception as e:
                print(f"Error processing with {path_to_source_audio}: {e}")
                continue
    
    with open(output_path / "train.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(metadata))

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic audio dataset using StyleTTS2")
    parser.add_argument("--model_config_path", type=str, required=True, help="Path to model config file")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output directory")
    parser.add_argument("--speaker_config_path", type=str, required=True, help="Path to speaker config JSON file")
    parser.add_argument("--max_duration", type=float, default=None, help="Maximum duration in seconds for generated audio")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples generated per speaker")
    parser.add_argument("--fadeout", type=float, default=0.1, help="Fadeout duration in seconds (default: 0.1)")
    parser.add_argument("--silence", type=float, default=0.1, help="Silence duration in seconds to append (default: 0.1)")
    args = parser.parse_args()
    generate_speech_to_speech(
        model_config_path=args.model_config_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        speaker_config_path=args.speaker_config_path,
        max_duration=args.max_duration,
        limit=args.limit,
        fadeout=args.fadeout,
        silence=args.silence
    )

if __name__ == "__main__":
    main()