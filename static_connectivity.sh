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

NAME[$k]="SYM ${xx} REMCAS_NAIVE_HALVE_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMCAS_COMPRESS_HALVE_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMCAS_A_HALVE_HALVE_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMCAS_A_SPLIT_HALVE_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMCAS_NAIVE_SPLIT_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMCAS_COMPRESS_SPLIT_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMCAS_A_HALVE_SPLIT_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMCAS_A_SPLIT_SPLIT_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMCAS_NAIVE_SPLICE_CAS"
k=$k+1
NAME[$k]="SYM ${xx} REMCAS_A_HALVE_SPLICE_CAS"
k=$k+1
NAME[$k]="SYM ${xx} REMCAS_A_SPLIT_SPLICE_CAS"
k=$k+1

NAME[$k]="SYM ${xx} REMLOCK_NAIVE_HALVE_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMLOCK_COMPRESS_HALVE_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMLOCK_A_HALVE_HALVE_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMLOCK_A_SPLIT_HALVE_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMLOCK_NAIVE_SPLIT_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMLOCK_COMPRESS_SPLIT_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMLOCK_A_HALVE_SPLIT_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMLOCK_A_SPLIT_SPLIT_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMLOCK_NAIVE_SPLICE_CAS"
k=$k+1
NAME[$k]="SYM ${xx} REMLOCK_A_HALVE_SPLICE_CAS"
k=$k+1
NAME[$k]="SYM ${xx} REMLOCK_A_SPLIT_SPLICE_CAS"
k=$k+1

NAME[$k]="SYM ${xx} SIM_P_U_S"
k=$k+1
NAME[$k]="SYM ${xx} SIM_P_R_S"
k=$k+1
NAME[$k]="SYM ${xx} SIM_E_U_S"
k=$k+1
NAME[$k]="SYM ${xx} SIM_P_U_SS"
k=$k+1
NAME[$k]="SYM ${xx} SIM_P_R_SS"
k=$k+1
NAME[$k]="SYM ${xx} SIM_E_U_SS"
k=$k+1

NAME[$k]="SYM ${xx} STERGIOS"
k=$k+1
NAME[$k]="SYM ${xx} SVA"
k=$k+1
NAME[$k]="SYM ${xx} LPA"
k=$k+1

done

#COO

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

NAME[$k]="SYM ${xx} REMCAS_NAIVE_HALVE_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMCAS_COMPRESS_HALVE_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMCAS_A_HALVE_HALVE_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMCAS_A_SPLIT_HALVE_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMCAS_NAIVE_SPLIT_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMCAS_COMPRESS_SPLIT_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMCAS_A_HALVE_SPLIT_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMCAS_A_SPLIT_SPLIT_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMCAS_NAIVE_SPLICE_CAS"
k=$k+1
NAME[$k]="SYM ${xx} REMCAS_A_HALVE_SPLICE_CAS"
k=$k+1
NAME[$k]="SYM ${xx} REMCAS_A_SPLIT_SPLICE_CAS"
k=$k+1

NAME[$k]="SYM ${xx} REMLOCK_NAIVE_HALVE_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMLOCK_COMPRESS_HALVE_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMLOCK_A_HALVE_HALVE_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMLOCK_A_SPLIT_HALVE_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMLOCK_NAIVE_SPLIT_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMLOCK_COMPRESS_SPLIT_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMLOCK_A_HALVE_SPLIT_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMLOCK_A_SPLIT_SPLIT_ONE"
k=$k+1
NAME[$k]="SYM ${xx} REMLOCK_NAIVE_SPLICE_CAS"
k=$k+1
NAME[$k]="SYM ${xx} REMLOCK_A_HALVE_SPLICE_CAS"
k=$k+1
NAME[$k]="SYM ${xx} REMLOCK_A_SPLIT_SPLICE_CAS"
k=$k+1

NAME[$k]="SYM ${xx} SIM_C_U_S_A"
k=$k+1
NAME[$k]="SYM ${xx} SIM_C_R_S_A"
k=$k+1
NAME[$k]="SYM ${xx} SIM_P_U_S_A"
k=$k+1
NAME[$k]="SYM ${xx} SIM_P_R_S_A"
k=$k+1
NAME[$k]="SYM ${xx} SIM_P_U_S"
k=$k+1
NAME[$k]="SYM ${xx} SIM_P_R_S"
k=$k+1
NAME[$k]="SYM ${xx} SIM_E_U_S_A"
k=$k+1
NAME[$k]="SYM ${xx} SIM_E_U_S"
k=$k+1

NAME[$k]="SYM ${xx} SIM_C_U_SS_A"
k=$k+1
NAME[$k]="SYM ${xx} SIM_C_R_SS_A"
k=$k+1
NAME[$k]="SYM ${xx} SIM_P_U_SS_A"
k=$k+1
NAME[$k]="SYM ${xx} SIM_P_R_SS_A"
k=$k+1
NAME[$k]="SYM ${xx} SIM_P_U_SS"
k=$k+1
NAME[$k]="SYM ${xx} SIM_P_R_SS"
k=$k+1
NAME[$k]="SYM ${xx} SIM_E_U_SS_A"
k=$k+1
NAME[$k]="SYM ${xx} SIM_E_U_SS"
k=$k+1

NAME[$k]="SYM ${xx} STERGIOS"
k=$k+1
NAME[$k]="SYM ${xx} SVA"
k=$k+1
NAME[$k]="SYM ${xx} LPA"
k=$k+1

done

#BFS-CC (ori.enable)

NAME[$k]="SYM BFS BFS_CC"
k=$k+1


DIR="file_location"
EXEC="file_name"

######for each file, you can elaborate as follows.

#for ((j=0;j<$k;j++))
#do
#echo -n ${NAME[$j]}
#echo -n ","
#${DIR}/${EXEC} mtx_file_name ${NAME[$j]}
#done



