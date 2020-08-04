ARCH="-DVOLTA"

#SCC
nvcc --disable-warnings  -DALPHA=4 ${ARCH} -O3 -std=c++11 -gencode arch=compute_70,code=sm_70 -DSSFACT=4 -DSCC -DFUS_SAMPLE --use_fast_math -Xptxas "-v -dlcm=ca" gen_graph.cu union_find.cu main.cu -o scc

#SPTREE
nvcc --disable-warnings  ${ARCH} -O3 -std=c++11 -gencode arch=compute_70,code=sm_70 -DSSFACT=4 -DSP_TREE -DFUS_SAMPLE --use_fast_math -Xptxas "-v -dlcm=ca" gen_graph.cu union_find.cu main.cu -o sp_tree

#DCC
nvcc --disable-warnings  ${ARCH} -O3 -std=c++11 -gencode arch=compute_70,code=sm_70 -DSSFACT=4 -DSTREAMING_SIMPLE -DFUS_SAMPLE --use_fast_math -Xptxas "-v -dlcm=ca" gen_graph.cu union_find.cu main.cu -o dcc

#CHUNK
nvcc --disable-warnings  ${ARCH} -O3 -std=c++11 -gencode arch=compute_70,code=sm_70 -DSSFACT=4 -DCHUNK_STREAMING -DSTREAMING_SIMPLE -DFUS_SAMPLE --use_fast_math -Xptxas "-v -dlcm=ca" gen_graph.cu union_find.cu main.cu -o chunk

#PL
nvcc --disable-warnings  ${ARCH} -O3 -std=c++11 -gencode arch=compute_70,code=sm_70 -DSSFACT=4 -DSCC -DFUS_SAMPLE -DPATH_LENGTH --use_fast_math -Xptxas "-v -dlcm=ca" gen_graph.cu union_find.cu main.cu -o PL






