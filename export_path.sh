# https://www.youtube.com/watch?v=LHtNv-dq8I4&t=861s

#cd $(dirname $(python -c 'print(__import__("tensorflow").__file__)'))
#ln -svf ../nvidia/*/lib/*.so* .
#cd -
#ln -sf $VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas $VIRTUAL_ENV/bin/ptxas

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib/python3.12/site-packages/tensorflow