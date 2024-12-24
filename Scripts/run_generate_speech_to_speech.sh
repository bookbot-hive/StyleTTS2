# s2s with sw syllables
# python Scripts/generate_speech_to_speech.py \
#     --model_config_path ./Models/SW-Bible-Victoria-20-epochs/config_ft_sw_bible_victoria.yml \
#     --checkpoint_path ./Models/SW-Bible-Victoria-20-epochs/epoch_2nd_00019.pth \
#     --reference_mode
#     --output_path ../sw-Victoria-Althaf-24kHz \
#     --speaker_config_path ./Scripts/Configs/Swahili/sentence.json \
#     --max_duration 20.0 \
#     --limit 10

# EN-Althaf-S2S
python Scripts/generate_speech_to_speech.py \
    --model_config_path ./Models/EN-Multi-ID-Althaf-emphasis/config_ft_en_multi_id_althaf_sw_victoria.yml \
    --checkpoint_path /home/s44504/3b01c699-3670-469b-801f-13880b9cac56/StyleTTS2/Models/EN-Multi-ID-Althaf-emphasis/epoch_2nd_00029.pth \
    --output_path ../en-Multi-Althaf-Short-Sentence-24kHz \
    --speaker_config_path ./Scripts/Configs/English/speech2speech/short_sentences.json \
    --max_duration 20.0