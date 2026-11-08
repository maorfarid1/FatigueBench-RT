set -e
for s in $@; do
  for k in 0 20 10 5 3; do
    python3 finetune_real.py --pre S4 --seed $s --k $k --crack --methods "fine-tuned" > /home/claude/paper/results/ftc_k${k}_s${s}.log 2>&1
  done
done
echo done > /home/claude/paper/results/ftc_$1.done
