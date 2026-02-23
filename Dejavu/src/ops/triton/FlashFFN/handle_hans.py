

handle_hans是处理hans_taskQ的triton Kernel

输入task_hans={hans0, hans1, xxx}

其中hansi={0, 9, 11, xxx}代表神经元的索引，对于输入的X, M, 

采用TensorCore的m16n8k16处理，

设置持久化Block，也即就那么多Block不断取任务处理

参考/home/yswang/tvm_learn/cuda/flash-attention/flash_attn/flash_attn_triton.py写成FlashMMM的格式，来了X和A_i以及B_i，算一部分输出一部分
    
    

