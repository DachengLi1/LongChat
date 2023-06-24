from torch.distributed import get_rank, is_initialized

def rank0_print(*args):
    if is_initialized():
        if get_rank() == 0:
            print(*args)
    else:
        print(*args)
