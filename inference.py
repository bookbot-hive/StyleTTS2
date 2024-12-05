import torch
import numpy as np
import yaml
import torchaudio
import librosa
import phonemizer
import soundfile as sf
import time
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from models import *
from utils import *
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert
from nltk.tokenize import word_tokenize
import json
import os
from pathlib import Path
import tqdm
import random
import spacy
import string
random.seed(time.time())


# Update the device selection in the __init__ method
SAMPLING_RATE = 24000

class StyleTTS2Synthesizer:
    def __init__(self, config_path, checkpoint_path):
        self.device = torch.device(
            "cuda:1"
            if torch.cuda.is_available() and torch.cuda.device_count() > 1
            else "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.config = yaml.safe_load(open(config_path))
        self.model_params = recursive_munch(self.config["model_params"])
        self.load_models(checkpoint_path)
        self.setup_mel_spectrogram()
        self.setup_sampler()
        
        # Initialize phonemizers dictionary for different languages
        self.phonemizers = {}
        self.textcleaner = TextCleaner()
        
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")

    def load_models(self, checkpoint_path):
        params_whole = torch.load(checkpoint_path, map_location="cpu")
        params = params_whole["net"]

        ASR_config = self.config.get("ASR_config", False)
        ASR_path = self.config.get("ASR_path", False)
        F0_path = self.config.get("F0_path", False)
        BERT_path = self.config.get("PLBERT_dir", False)

        text_aligner = load_ASR_models(ASR_path, ASR_config)
        pitch_extractor = load_F0_models(F0_path)
        plbert = load_plbert(BERT_path)

        self.model = build_model(
            self.model_params, text_aligner, pitch_extractor, plbert
        )

        for key in self.model:
            if key in params:
                try:
                    self.model[key].load_state_dict(params[key])
                except:
                    from collections import OrderedDict

                    state_dict = params[key]
                    new_state_dict = OrderedDict(
                        (k[7:], v) for k, v in state_dict.items()
                    )
                    self.model[key].load_state_dict(new_state_dict, strict=False)

            self.model[key].eval()
            self.model[key].to(self.device)

    def setup_phonemizer(self, language):
        """Setup phonemizer for a specific language if not already initialized"""
        if language not in self.phonemizers:
            self.phonemizers[language] = phonemizer.backend.EspeakBackend(
                language=language, preserve_punctuation=True, with_stress=True
            )
        return self.phonemizers[language]

    def setup_mel_spectrogram(self):
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300
        )
        self.mean, self.std = -4, 4

    def setup_sampler(self):
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )

    def preprocess(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def compute_style(self, path):
        wave, sr = librosa.load(path, sr=SAMPLING_RATE)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        if sr != SAMPLING_RATE:
            audio = librosa.resample(audio, sr, SAMPLING_RATE)
        mel_tensor = self.preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)

    def length_to_mask(self, lengths):
        mask = (
            torch.arange(lengths.max())
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
            .type_as(lengths)
        )
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask

    def _inference(
        self, text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1, phonemes=False, language="en-us"
    ):
        text = text.strip()
        if phonemes:
            ps = text
        else:
            # Get or create phonemizer for the specified language
            current_phonemizer = self.setup_phonemizer(language)
            ps = current_phonemizer.phonemize([text])[0]
            
        print(f"Phonemes: {ps}")
            
        tokens = self.textcleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = self.length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s,
                num_steps=diffusion_steps,
            ).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            
            if not text[-1].isalnum():
                pred_dur[-1] = 1

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device)
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = t_en @ pred_aln_trg.unsqueeze(0).to(self.device)
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()[..., :-50], ps

    def synthesize_speech(
        self,
        text="",
        reference_path="",
        diffusion_steps=10,
        embedding_scale=1,
        alpha=0.3,
        beta=0.7,
        phonemes=False,
        language="en-us" 
    ):
        start = time.time()
        try:
            ref_s = self.compute_style(reference_path)

            wav, ps = self._inference(
                text,
                ref_s,
                alpha=alpha,
                beta=beta,
                diffusion_steps=diffusion_steps,
                embedding_scale=embedding_scale,
                phonemes=phonemes,
                language=language  # Pass language to _inference
            )
        except Exception as e:
            print(f"Error during synthesis: {e}")

        return wav, ps