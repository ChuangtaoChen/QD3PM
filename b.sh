

#python gaussian_10bit.py --layer 8 --lr 0.0005

for s in 1 2 3 4 5
do
for l in 6 12
do
  python main_BAS.py --N 4 --layer $l --seed $s --num_timesteps 30 --ansatz_connection "star" --sigma 10000 --epochs 6000 --decay_steps 3000 --batch_size 16 --num_samples 10000 --min_lr 0.0001
done
done






#for decay_steps in 2000 4000 5000 6000 8000
#do
#for ilr in 0.01 0.008 0.005 0.002 0.001 0.0005
#do
#python main_BAS.py --initial_lr  $ilr --decay_steps $decay_steps
#done
#done

#
#
#for ds in 4000 6000 8000 10000
#do
#for l in 4 6 8 10 12 14
#do
#python gaussian_8bit.py --layer $l --lr 0.0008 --final_lr 0.00001 --decay_steps $ds --num_timesteps 30 --lr_decay True
#done
#done

#for ds in 6000
#do
#for l in 40 35
#do
#python gaussian_8bit.py --layer $l --lr 0.0008 --final_lr 0.00005 --decay_steps $ds --num_timesteps 30 --lr_decay True
#done
#done
#
#for ds in 4000 6000 8000 10000
#do
#for l in 4 6 8 10 12 14
#do
#python gaussian_8bit.py --layer $l --lr 0.0005 --final_lr 0.00001 --decay_steps $ds --num_timesteps 30 --lr_decay True
#done
#done



# /usr/bin/shutdown