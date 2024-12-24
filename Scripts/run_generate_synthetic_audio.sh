# Swahili
# python generate_synthetic_audio.py \
#     --model_config_path ./Models/SW-Bible-Victoria-20-epochs/config_ft_sw_bible_victoria.yml \
#     --checkpoint_path ./Models/SW-Bible-Victoria-20-epochs/epoch_2nd_00019.pth \
#     --dataset_path /home/s44504/3b01c699-3670-469b-801f-13880b9cac56/OpenBible_Swahili_Althaf/metadata.csv \
#     --mode metadata \
#     --output_path ../sw-Victoria-synthetic-24kHz \
#     --speaker_config_path ./Scripts/Configs/Swahili/sentence.json \
#     --max_duration 30.0 

# python generate_synthetic_audio.py \
#     --model_config_path ./Models/SW-Bible-Victoria-20-epochs/config_ft_sw_bible_victoria.yml \
#     --checkpoint_path ./Models/SW-Bible-Victoria-20-epochs/epoch_2nd_00019.pth \
#     --dataset_path /home/s44504/3b01c699-3670-469b-801f-13880b9cac56/sw-TZ-Althaf-Multi-Syllables/metadata.csv \
#     --mode metadata \
#     --output_path ../sw-Victoria-Multi-Syllables-24kHz \
#     --speaker_config_path ./Scripts/Configs/Swahili/sentence.json \
#     --max_duration 20.0 

# Indonesia
# python generate_synthetic_audio.py \
#     --model_config_path /home/s44504/3b01c699-3670-469b-801f-13880b9cac56/StyleTTS2/Models/EN-Multi-ID-Althaf-SW-Victoria/config_ft_en_multi_id_althaf_sw_victoria.yml \
#     --checkpoint_path /home/s44504/3b01c699-3670-469b-801f-13880b9cac56/StyleTTS2/Models/EN-Multi-ID-Althaf-SW-Victoria/epoch_2nd_00019.pth \
#     --dataset_path /home/s44504/3b01c699-3670-469b-801f-13880b9cac56/dataset_creation/data/indonesia_book_transcripts_with_f.txt \
#     --reference_path /home/s44504/3b01c699-3670-469b-801f-13880b9cac56/StyleTTS2/Demo/id-ID-Althaf/concatenated_audio.wav \
#     --mode single \
#     --output_path ../id-Althaf-Synthetic-24kHz \
#     --speaker_config_path ./Scripts/Configs/Indonesia/sentence.json \
#     --max_duration 20.0 

# EN-MULTI
python Scripts/generate_synthetic_audio.py \
    --model_config_path /home/s44504/3b01c699-3670-469b-801f-13880b9cac56/StyleTTS2/Models/EN-Multi-ID-Althaf-emphasis/config_ft_en_multi_id_althaf_sw_victoria.yml \
    --checkpoint_path /home/s44504/3b01c699-3670-469b-801f-13880b9cac56/StyleTTS2/Models/EN-Multi-ID-Althaf-emphasis/epoch_2nd_00029.pth \
    --dataset_path /home/s44504/3b01c699-3670-469b-801f-13880b9cac56/dataset_creation/data/english_transcript_short_sentences.txt \
    --mode single \
    --output_path ../en-Multi-Short-Sentences-24kHz \
    --speaker_config_path ./Scripts/Configs/English/short_sentences.json \
    --max_duration 20.0 