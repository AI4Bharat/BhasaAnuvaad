langs=("te" "mr" "hi")

# Start new screen sessions in detached mode for each language
for i in "${!langs[@]}"; do
  screen -dmS "${langs[$i]}"
  screen -S "${langs[$i]}" -X stuff "cd /data-4/sparsh/shrutilipi-pipeline/NPTEL/spoken-translation-pipeline\n"
  screen -S "${langs[$i]}" -X stuff "conda activate spokent\n"
  screen -S "${langs[$i]}" -X stuff "CUDA_VISIBLE_DEVICES=$i python3 main.py -c config_${langs[$i]}.yaml\n"
  echo "Started session for ${langs[$i]} with CUDA device $i (index $i)"
done