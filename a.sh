#for s in 1
#do
#for l in 14
#do
#  python main_BAS.py --layer $l --seed $s --num_timesteps 30 --ansatz_connection "star"
#done
#done


for s in 1 2 3 4 5
do
for l in 8 10
do
  python main_BAS.py --N 4 --layer $l --seed $s --num_timesteps 30 --ansatz_connection "star" --sigma 10000 --epochs 6000 --decay_steps 3000 --batch_size 16 --num_samples 10000 --min_lr 0.0001
done
done

#for s in 1 2 3 4 5
#do
#for l in 6 8 10 12
#do
#  python main_BAS.py --N 4 --num_dataset 5000 --layer $l --seed $s --num_timesteps 30 --ansatz_connection "chain" --sigma 10000 --epochs 6000 --decay_steps 3000 --batch_size 16 --num_samples 10000 --min_lr 0.0001
#done
#done


