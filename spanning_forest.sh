k=0
#CSR

for xx in CSR SAMPLE IHOOK BFS
do

NAME[$k]="SYM ${xx} ASYNC_NAIVE"
k=$k+1
NAME[$k]="SYM ${xx} ASYNC_COMPRESS"
k=$k+1
NAME[$k]="SYM ${xx} ASYNC_A_HALVE"
k=$k+1
NAME[$k]="SYM ${xx} ASYNC_A_SPLIT"
k=$k+1

NAME[$k]="SYM ${xx} STOPT_NAIVE"
k=$k+1
NAME[$k]="SYM ${xx} STOPT_COMPRESS"
k=$k+1
NAME[$k]="SYM ${xx} STOPT_A_HALVE"
k=$k+1
NAME[$k]="SYM ${xx} STOPT_A_SPLIT"
k=$k+1

NAME[$k]="SYM ${xx} EARLY_NAIVE"
k=$k+1
NAME[$k]="SYM ${xx} EARLY_COMPRESS"
k=$k+1
NAME[$k]="SYM ${xx} EARLY_A_HALVE"
k=$k+1
NAME[$k]="SYM ${xx} EARLY_A_SPLIT"
k=$k+1

NAME[$k]="SYM ${xx} RAND_NAIVE" 
k=$k+1
NAME[$k]="SYM ${xx} RAND_SPLIT_2" 
k=$k+1


NAME[$k]="SYM ${xx} ECL_NAIVE"
k=$k+1
NAME[$k]="SYM ${xx} ECL_COMPRESS"
k=$k+1
NAME[$k]="SYM ${xx} ECL_A_HALVE" 
k=$k+1
NAME[$k]="SYM ${xx} ECL_A_SPLIT" 
k=$k+1
 
NAME[$k]="SYM ${xx} AFFOREST_NAIVE" 
k=$k+1
NAME[$k]="SYM ${xx} AFFOREST_COMPRESS" 
k=$k+1
NAME[$k]="SYM ${xx} AFFOREST_A_HALVE" 
k=$k+1
NAME[$k]="SYM ${xx} AFFOREST_A_SPLIT" 
k=$k+1

done

for xx in COO COO_SAMPLE
do

NAME[$k]="SYM ${xx} ASYNC_NAIVE"
k=$k+1
NAME[$k]="SYM ${xx} ASYNC_COMPRESS"
k=$k+1
NAME[$k]="SYM ${xx} ASYNC_A_HALVE"
k=$k+1
NAME[$k]="SYM ${xx} ASYNC_A_SPLIT"
k=$k+1

NAME[$k]="SYM ${xx} STOPT_NAIVE"
k=$k+1
NAME[$k]="SYM ${xx} STOPT_COMPRESS"
k=$k+1
NAME[$k]="SYM ${xx} STOPT_A_HALVE"
k=$k+1
NAME[$k]="SYM ${xx} STOPT_A_SPLIT"
k=$k+1

NAME[$k]="SYM ${xx} EARLY_NAIVE"
k=$k+1
NAME[$k]="SYM ${xx} EARLY_COMPRESS"
k=$k+1
NAME[$k]="SYM ${xx} EARLY_A_HALVE"
k=$k+1
NAME[$k]="SYM ${xx} EARLY_A_SPLIT"
k=$k+1

NAME[$k]="SYM ${xx} RAND_NAIVE" 
k=$k+1
NAME[$k]="SYM ${xx} RAND_SPLIT_2" 
k=$k+1


NAME[$k]="SYM ${xx} ECL_NAIVE"
k=$k+1
NAME[$k]="SYM ${xx} ECL_COMPRESS"
k=$k+1
NAME[$k]="SYM ${xx} ECL_A_HALVE" 
k=$k+1
NAME[$k]="SYM ${xx} ECL_A_SPLIT" 
k=$k+1
 
NAME[$k]="SYM ${xx} AFFOREST_NAIVE" 
k=$k+1
NAME[$k]="SYM ${xx} AFFOREST_COMPRESS" 
k=$k+1
NAME[$k]="SYM ${xx} AFFOREST_A_HALVE" 
k=$k+1
NAME[$k]="SYM ${xx} AFFOREST_A_SPLIT" 
k=$k+1

done

DIR="file_location"
EXEC="file_name"

######for each file, you can elaborate as follows.

#for ((j=0;j<$k;j++))
#do
#echo -n ${NAME[$j]}
#echo -n ","
#${DIR}/${EXEC} mtx_file_name ${NAME[$j]}
#done


