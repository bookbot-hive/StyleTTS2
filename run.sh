# accelerate launch train_first.py --config_path ./Configs/config_librittsr_first.yml
# python train_second.py --config_path ./Configs/config_vctk.yml

accelerate launch train_second_accelerate.py --config_path ./Configs/config_vctk_second.yml

# python train_finetune.py --config_path ./Configs/config_ft_en_multi_id_althaf_sw_victoria.yml