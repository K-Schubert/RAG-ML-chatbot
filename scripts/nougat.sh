export CUDA_VISIBLE_DEVICES=$1
for file in ./input/cs_CL_$1/*.pdf
do
	nougat $file -o output/cs_CL_$1 -m 0.1.0-base --batchsize 8
	echo $file
done
