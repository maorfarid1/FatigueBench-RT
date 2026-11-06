set -e
i=1
for S in S1 S2 S3 S4; do
  python3 gen.py $S train 600 $((i*10+1)); python3 gen.py $S val 150 $((i*10+2)); python3 gen.py $S test 250 $((i*10+3)); i=$((i+1))
done
python3 gen.py S3 test 150 91 sae
python3 gen.py S3 test 150 92 ood_kurt
echo DONE
