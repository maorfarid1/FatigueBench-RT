# two parallel queues, one torch thread each
export NT=1
q1() {
  for s in 0 1 2; do
    sp=""; [ $s = 0 ] && sp="--save_pred"
    python3 run_bench.py S1 --seed $s $sp
    while ! grep -q DONE ../data/gen.log; do sleep 30; done
    python3 run_bench.py S3 --seed $s --test_variants base,sae,ood_kurt $sp
    python3 run_bench.py S3 --seed $s --split extrap --methods miner,dirlik,farid,mlp,lstm,pibnn,pibnn_noprior,pibnn_fullfeat,pibnn_noshrink,pibnn_noaux,pibnn_single
    python3 run_bench.py S2 --seed $s --split extrap --methods miner,dirlik,farid,mlp,lstm,pibnn
  done
  for n in 25 50 100 200 400; do for s in 0 1 2; do
    python3 run_bench.py S3 --ntrain $n --seed $s --methods miner,farid,mlp,pibnn
  done; done
}
q2() {
  for s in 0 1 2; do
    sp=""; [ $s = 0 ] && sp="--save_pred"
    python3 run_bench.py S4 --seed $s $sp
    python3 run_bench.py S2 --seed $s $sp
    python3 run_bench.py S4 --seed $s --split extrap --methods miner,dirlik,farid,mlp,lstm,pibnn
    python3 run_bench.py S3 --seed $s --methods pibnn_noprior,pibnn_fullfeat,pibnn_noshrink,pibnn_noaux,pibnn_single
  done
  for n in 25 50 100 200 400; do for s in 0 1 2; do
    python3 run_bench.py S4 --ntrain $n --seed $s --methods miner,farid,mlp,pibnn
  done; done
}
q1 > ../results/q1.log 2>&1 &
while ! grep -q DONE ../data/gen.log; do sleep 30; done
q2 > ../results/q2.log 2>&1 &
wait
echo ALLDONE >> ../results/q1.log
