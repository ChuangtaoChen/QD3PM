#for s in 1
#do
#for l in 14
#do
#  python main_BAS.py --layer $l --seed $s --num_timesteps 30 --ansatz_connection "star"
#done
#done

for s in 1 2 3 4 5
do
for l in 6 8 10 12
do
  python main_BAS.py --layer $l --seed $s --num_timesteps 30 --ansatz_connection "all_to_all" --sigma 0.1
done
done



