export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

LEXICON="MALD-1000-train"
#LEXICON="MALD-1000-test"

earshot_options=( --lexicon $LEXICON --n_bands 64 --batch_size 32 --steps_per_epoch 25 --mechanism LSTM )
#earshot_options=( --lexicon $LEXICON --mechanism LSTM --n_bands 64 --batch_size 32 --steps_per_epoch 50 --patience 250 )

# Plain deep
for hidden in 320x320 256x256x256 192x192x192x192; do  # 512 320x320 256x256x256 192x192x192x192
  for target in OneHot Glove-300c Glove-50c; do  # OneHot Glove-300c Glove-50c
    for seed in 0; do  # 1 2
      earshot-train    "${earshot_options[@]}" --target $target --seed $seed --hidden $hidden "$@"
      earshot-evaluate "${earshot_options[@]}" --target $target --seed $seed --hidden $hidden "$@"
      earshot-evaluate "${earshot_options[@]}" --target $target --seed $seed --hidden $hidden --test-stimulus Sil "$@"
    done
  done
done
