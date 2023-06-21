from torch.distributed import get_rank

def rank0_print(*args):
    if get_rank() == 0:
        print(*args)
