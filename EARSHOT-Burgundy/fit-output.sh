export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/


LEXICON="MALD-1000-train"
# LEXICON="MALD-1000-test"

earshot_options=( --lexicon $LEXICON --n_bands 64 --batch_size 32 --steps_per_epoch 25 --mechanism LSTM )
#earshot_options=( --lexicon $LEXICON --mechanism LSTM --n_bands 64 --batch_size 32 --steps_per_epoch 50 --patience 250 )

for hidden in 512 1024 2048  # 512 1024 2048
do
  for target in OneHot Glove-300c Glove-50c Sparse-10of300 Sparse-10of900
  do
    for seed in 0 # 1 2
      do
#      earshot-train    "${earshot_options[@]}" --target $target --seed $seed --hidden $hidden "$@"
#      earshot-evaluate "${earshot_options[@]}" --target $target --seed $seed --hidden $hidden "$@"
      earshot-evaluate "${earshot_options[@]}" --target $target --seed $seed --hidden $hidden --test-stimulus Sil "$@"
    done
  done
done
