export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# Skip alternating (like resnet):
for hidden in 640x640x0s640x640x2s640-256
do
  for mechanism in LSTM
  do
    for target in OneHot # Glove-50c Glove-300c # Sparse-10of300 Sparse-10of900  
    do
      for seed in 0 # 1 2
        do
        earshot-train    --lexicon MALD-1000-train --n_bands 64 --batch_size 32 --steps_per_epoch 50 --patience 250 --target $target --seed $seed --hidden $hidden --mechanism $mechanism "$@"
        earshot-evaluate --lexicon MALD-1000-train --n_bands 64 --batch_size 32 --steps_per_epoch 50 --patience 250 --target $target --seed $seed --hidden $hidden --mechanism $mechanism "$@"
      done
    done
  done
done

#systemctl suspend
