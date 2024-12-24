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
    """Load metadata.csv from dataset directory"""
    metadata_path = os.path.join(dataset_path, "metadata.csv")
    return pd.read_csv(metadata_path)

def generate_single_audio(text: str, source_style: torch.Tensor, target_style: torch.Tensor, language: str, output_filepath: str, synthesizer: StyleTTS2Synthesizer, **kwargs):
    """Helper function to generate a single audio file"""
    try:
        wav = synthesizer.s2s(
            text,
            source_style,
            target_style,
            language,
            alpha=kwargs.get("alpha", 0.1),
            beta=kwargs.get("beta", 0.3),
            diffusion_steps=kwargs.get("diffusion_steps", 10),
            phonemes=kwargs.get("phonemes", False)
        )
        duration = len(wav) / SAMPLING_RATE
        if kwargs.get("max_duration", 0) and duration > kwargs.get("max_duration", 0):
            print(f"Skipping audio with {kwargs['speaker_config']['name']} - Duration {duration:.2f}s exceeds limit of {kwargs.get('max_duration', 0)}s")
            return False
        fadeout_samples = int(0.030 * SAMPLING_RATE)  # 25ms
        fadeout_window = np.linspace(1, 0, fadeout_samples)
        wav[-fadeout_samples:] *= fadeout_window
        sf.write(str(output_filepath), wav, SAMPLING_RATE)
        return True
    except Exception as e:
        print(f"Error processing with {kwargs['speaker_config']['name']}: {e}")
        return False

def generate_speech_to_speech(model_config_path, 
                              checkpoint_path, 
                              output_path, 
                              speaker_config_path, 
                              max_duration, 
                              limit):
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
            output_filename = row[audio_column]
            output_filename = output_filename.split("/")[-1]
            if speaker_config["name"]:
                output_filename = f"{speaker_config['name']}_{output_filename}"
            output_filepath = wavs_dir /output_filename
            try:
                text_column = speaker_config["text_column"]
                audio_column = speaker_config["audio_column"]
                reference_path = speaker_config["reference_path"]
                speaker_id = row["speaker_id"]
                text = row[text_column]
                source_audio_path = row[audio_column]
                source_style = synthesizer.compute_style(source_audio_path)
                target_style = synthesizer.compute_style(reference_path)
                language = row["language"]
                alpha=speaker_config["alpha"]
                beta=speaker_config["beta"]
                embedding_scale=speaker_config["embedding_scale"]
                if generate_single_audio(text, 
                                        source_style, 
                                        target_style, 
                                        language, 
                                        output_filepath, 
                                        synthesizer, 
                                        max_duration=max_duration,
                                        alpha=alpha,
                                        beta=beta,
                                        embedding_scale=embedding_scale,
                                        speaker_config=speaker_config):
                    metadata.append(f"{output_filename}|{text}|{speaker_id}")
            except Exception as e:
                print(f"Error processing with {speaker_config['name']}: {e}")
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
    args = parser.parse_args()
    generate_speech_to_speech(
        model_config_path=args.model_config_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        speaker_config_path=args.speaker_config_path,
        max_duration=args.max_duration,
        limit=args.limit
    )

if __name__ == "__main__":
    main()