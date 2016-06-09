t_task=book_order

for t_gru in 0 1; do
for t_n_embed in 64; do
for t_h in 128 256; do
for t_h2 in 0 64; do
for t_b in 32 64; do
for t_bptt in 3 5 7 9 11; do

qsub -v task=$t_task,gru=$t_gru,n_embed=$t_n_embed,h=$t_h,h2=$t_h2,b=$t_b,bptt=$t_bptt pbs_gm.pbs

done
done
done
done
done
done
