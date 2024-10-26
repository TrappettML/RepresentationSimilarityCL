for n in {0..9}
do
    python launch.py --exp-name "Br${n}SomaFunc" \
                    --command "python ../experiments/LongLearning.py ${n}" \
                    --num-nodes 1 \
                    --partition "computelong" \
                    --days 14
done