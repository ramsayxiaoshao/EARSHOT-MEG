export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

LEXICON="MALD-1000-train"
SEED=0

earshot_options=( --target OneHot --lexicon $LEXICON --n_bands 64 --seed $SEED --batch_size 32 --mechanism LSTM --steps_per_epoch 25 )
#earshot_options=( --lexicon $LEXICON --n_bands 64 --seed $SEED --batch_size 32 --mechanism LSTM --steps_per_epoch 50 --patience 250 )

# Base
for hidden in 512 320x320 256x256x256 192x192x192x192 # 512 320x320 256x256x256 192x192x192x192
do
  for loss in  16 64 256 1024 4096  # 16 64 256 1024 4096
  do
    earshot-train    "${earshot_options[@]}" --hidden $hidden --loss dw${loss}to10 "$@"
    earshot-evaluate "${earshot_options[@]}" --hidden $hidden --loss dw${loss}to10 "$@"
    earshot-evaluate "${earshot_options[@]}" --hidden $hidden --loss dw${loss}to10 --test-stimulus Sil "$@"
  done
done
