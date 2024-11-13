conda init
conda activate vad

# python script/generate_synthetic_audio.py \
#     --model_config_path /home/s44504/3b01c699-3670-469b-801f-13880b9cac56/StyleTTS2/Models/EN-Multi-ID-Althaf-emphasis/config_ft_en_multi_id_althaf_sw_victoria.yml \
#     --checkpoint_path /home/s44504/3b01c699-3670-469b-801f-13880b9cac56/StyleTTS2/Models/EN-Multi-ID-Althaf-emphasis/epoch_2nd_00029.pth \
#     --dataset_path /home/s44504/3b01c699-3670-469b-801f-13880b9cac56/dataset_creation/data/all_book_english_words_2.txt \
#     --output_path ../en-Multi-Word-24kHz \
#     --speaker_config_path ./configs/exclamation_config.json \
#     --max_duration 10.0


python Scripts/generate_synthetic_audio.py \
    --model_config_path /home/s44504/3b01c699-3670-469b-801f-13880b9cac56/StyleTTS2/Models/EN-Multi-ID-Althaf-emphasis/config_ft_en_multi_id_althaf_sw_victoria.yml \
    --checkpoint_path /home/s44504/3b01c699-3670-469b-801f-13880b9cac56/StyleTTS2/Models/EN-Multi-ID-Althaf-emphasis/epoch_2nd_00029.pth \
    --output_path ../en-Multi-Exclamation-24kHz-2 \
    --speaker_config_path ./Scripts/Configs/exclamation_config.json \
    --max_duration 10.0