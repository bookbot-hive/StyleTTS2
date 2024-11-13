find ./wavs -name "*.wav" > audio_files.lst

audiosr -il audio_files.lst -s ../en-Multi-Exclamation-48kHz -d cuda:0