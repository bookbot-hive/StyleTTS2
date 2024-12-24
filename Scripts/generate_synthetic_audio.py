import argparse
import tqdm
import soundfile as sf
import sys
import shutil
import json
import pandas as pd
import os
import re
import sys

from pathlib import Path
from tqdm import tqdm
from inference import StyleTTS2Synthesizer, SAMPLING_RATE

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_metadata(dataset_path):
    """Load metadata.csv from dataset directory"""
    return pd.read_csv(dataset_path)

def generate_single_audio(text, reference_path, output_path, speaker, synthesizer, max_duration):
    """Helper function to generate a single audio file"""
    try:
        wav, _ = synthesizer.synthesize_speech(
            text=text,
            reference_path=reference_path,
            alpha=speaker["alpha"],
            beta=speaker["beta"],
            embedding_scale=speaker["embedding_scale"],
            language=speaker["language"]
        )
        duration = len(wav) / SAMPLING_RATE
        if max_duration and duration > max_duration:
            print(f"Skipping audio with {speaker['name']} - Duration {duration:.2f}s exceeds limit of {max_duration}s")
            return False
        sf.write(str(output_path), wav, SAMPLING_RATE)
        return True
    except Exception as e:
        print(f"Error processing with {speaker['name']}: {e}")
        return False

def generate_synthetic_dataset(model_config_path, 
                               checkpoint_path, 
                               dataset_path, 
                               mode,output_path, 
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
    
    for speaker in settings:
        if mode == "metadata":
            text_column = speaker["text_column"]
            audio_column = speaker["audio_column"]
            
            df = load_metadata(dataset_path)
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {speaker['name']}"):
                if limit and idx >= limit:
                    break
                dataset_dir = '/'.join(dataset_path.split('/')[:-1])
                print(dataset_dir)
                ref_audio = os.path.join(dataset_dir, row[audio_column])
                if speaker["name"]:
                    output_filename = f"{speaker['name']}_{row[audio_column].replace('wavs/', '')}"
                else:
                    output_filename = row[audio_column]
                output_filename = output_filename.split("/")[-1]
                output_filepath = wavs_dir/output_filename
                text = row[text_column]
                text = re.sub(r'<break time="1\.0s" /> ', '', text)
                if generate_single_audio(text, ref_audio, output_filepath, speaker, synthesizer, max_duration):
                    metadata.append(f"{output_filename}|{text}|{speaker['speaker_id']}")
                
        elif mode == "single":  # single reference mode
            with open(dataset_path, 'r', encoding='utf-8') as f:
                transcripts = [line.strip() for line in f if line.strip()]
                reference_path = speaker["reference_path"]
                for idx, transcript in enumerate(tqdm(transcripts, desc=f"Processing {speaker['name']}")):
                    if limit and idx >= limit:
                        break
                    output_filename = f"{speaker['name']}_{idx+1:04d}.wav"
                    output_filepath = wavs_dir/output_filename
                    if generate_single_audio(transcript, reference_path, output_filepath, speaker, synthesizer, max_duration):
                        metadata.append(f"{output_filename}|{transcript}|{speaker['speaker_id']}")
    
    # Save metadata file
    with open(output_path / "train.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(metadata))

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic audio dataset using StyleTTS2")
    parser.add_argument("--model_config_path", type=str, required=True, help="Path to model config file")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset_path",  default="", type=str, help="Path to transcript text file (only needed for single reference mode)")
    parser.add_argument("--mode", type=str, choices=["single", "metadata"], help="Mode of operation: 'single' for single reference mode, 'metadata' for metadata mode")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output directory")
    parser.add_argument("--speaker_config_path", type=str, required=True, help="Path to speaker config JSON file")
    parser.add_argument("--max_duration", type=float, default=None, help="Maximum duration in seconds for generated audio")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples generated per speaker")
    
    args = parser.parse_args()
    
    generate_synthetic_dataset(
        model_config_path=args.model_config_path,
        checkpoint_path=args.checkpoint_path,
        dataset_path=args.dataset_path,
        mode=args.mode,
        output_path=args.output_path,
        speaker_config_path=args.speaker_config_path,
        max_duration=args.max_duration,
        limit=args.limit
    )
if __name__ == "__main__":
    main()