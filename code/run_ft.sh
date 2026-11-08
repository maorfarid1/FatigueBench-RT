set -e
PRE=$1
for s in 0 1 2 3 4; do
  for k in 0 20 10 5 3; do
    python3 finetune_real.py --pre $PRE --seed $s --k $k > /home/claude/paper/results/ft_${PRE}_k${k}_s${s}.log 2>&1
  done
done
echo FTDONE > /home/claude/paper/results/ft_${PRE}.done
