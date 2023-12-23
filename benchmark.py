import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func

from lightseq.lightseq_async_attn import attention
from lightseq.async_communication import initialize_distributed,\
    get_sequence_parallel_rank, get_sequence_parallel_size, \
        get_sequence_parallel_group


def benchmark_flash_attn(batch_size, seq_length, num_head, head_dim):
    # batch_size = [1]
    # seq_len = [8*1024, 16*1024, 32*1024, 64*1024, 128*1024]
    # # corresponds to 30B, 7B, 1.3B
    # head_and_dim = [(16, 96), (8, 128), (4, 128)]
    
    dtype = torch.float16
    device = torch.cuda.current_device()
    warm_up, iters = 10, 10

    qkv = torch.empty((batch_size, seq_length, 3, num_head, head_dim), dtype=dtype, device=device, requires_grad=True)
    grad = torch.empty((batch_size, seq_length, num_head, head_dim), dtype=dtype, device=device)

    for _ in range(warm_up):
        qkv.grad = None
        torch.nn.init.uniform_(qkv)
        torch.nn.init.uniform_(grad)

        o = flash_attn_func(qkv, causal=True)
        o.backward(grad)
    
    fwd_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    fwd_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    bwd_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    bwd_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        qkv.grad = None
        torch.nn.init.uniform_(qkv)
        torch.nn.init.uniform_(grad)

        fwd_start_events[i].record()
        o = flash_attn_func(qkv, causal=True)
        fwd_end_events[i].record()

        bwd_start_events[i].record()
        o.backward(grad)
        bwd_end_events[i].record()
    
    torch.cuda.synchronize()
    fwd_elapsed = [s.elapsed_time(e) for s, e in zip(fwd_start_events, fwd_end_events)]
    bwd_elapsed = [s.elapsed_time(e) for s, e in zip(bwd_start_events, bwd_end_events)]
    elapsed = torch.tensor(
        [fwd_elapsed, bwd_elapsed], dtype=torch.double, device=device)
    
    std, avg = torch.std_mean(elapsed, dim=1)
    std, avg = std.tolist(), avg.tolist()
    print(f">>> flash attn:\nbatch size: {batch_size}, seq len: {seq_length}, "
          f"num head: {num_head}, head dim: {head_dim}, "
          f"fwd avg: {avg[0]:.2f} ms, fwd std: {std[0]:.2f} ms, "
          f"bwd avg: {avg[1]:.2f} ms, bwd std: {std[1]:.2f} ms\n")


def benchmark_dist_attn(batch_size, seq_length, num_head, head_dim):
    initialize_distributed()

    seq_rank = get_sequence_parallel_rank()
    seq_world_size = get_sequence_parallel_size()
    seq_group = get_sequence_parallel_group()
    dtype = torch.float16
    device = torch.cuda.current_device()
    warm_up, iters = 10, 10
    
    q = torch.empty((batch_size, num_head, seq_length // seq_world_size, head_dim), dtype=dtype, device="cuda", requires_grad=True)
    k = torch.empty((batch_size, num_head, seq_length // seq_world_size, head_dim), dtype=dtype, device="cuda", requires_grad=True)
    v = torch.empty((batch_size, num_head, seq_length // seq_world_size, head_dim), dtype=dtype, device="cuda", requires_grad=True)
    grad = torch.empty_like(q)

    for _ in range(warm_up):
        q.grad = None
        k.grad = None
        v.grad = None
        torch.nn.init.uniform_(q)
        torch.nn.init.uniform_(k)
        torch.nn.init.uniform_(v)
        torch.nn.init.uniform_(grad)
    
        o = attention(q, k, v, True, 1.3)
        o.backward(grad)

    fwd_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    fwd_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    bwd_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    bwd_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        q.grad = None
        k.grad = None
        v.grad = None
        torch.nn.init.uniform_(q)
        torch.nn.init.uniform_(k)
        torch.nn.init.uniform_(v)
        torch.nn.init.uniform_(grad)

        fwd_start_events[i].record()
        o = attention(q, k, v, True, 1.3)
        fwd_end_events[i].record()

        bwd_start_events[i].record()
        o.backward(grad)
        bwd_end_events[i].record()
    
    torch.cuda.synchronize()
    fwd_elapsed = [s.elapsed_time(e) for s, e in zip(fwd_start_events, fwd_end_events)]
    bwd_elapsed = [s.elapsed_time(e) for s, e in zip(bwd_start_events, bwd_end_events)]
    elapsed = torch.tensor(
        [fwd_elapsed, bwd_elapsed], dtype=torch.double, device=device)
    
    gathered = [torch.empty_like(elapsed) if i !=seq_rank else elapsed \
                for i in range(seq_world_size)]
    dist.all_gather(gathered, elapsed, seq_group)

    gathered = torch.stack(gathered, dim=0)  # world size, 2, iters
    max_val = gathered.max(dim=0, keepdims=False).values # 2, iters
    std, avg = torch.std_mean(max_val, dim=-1)  # 2
    std, avg = std.tolist(), avg.tolist()
    if seq_rank == 0:
        print(f">>> dist attn:\nbatch size: {batch_size}, seq len: {seq_length}, "
            f"num head: {num_head}, head dim: {head_dim}, "
            f"fwd avg: {avg[0]:.2f} ms, fwd std: {std[0]:.2f} ms, "
            f"bwd avg: {avg[1]:.2f} ms, bwd std: {std[1]:.2f} ms\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', type=str, choices=['flash', 'dist'], required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--seq_length', type=int, required=True)
    parser.add_argument('--num_head', type=int, required=True)
    parser.add_argument('--head_dim', type=int, required=True)

    args = parser.parse_args()

    if args.kernel == 'flash':
        benchmark_flash_attn(args.batch_size, args.seq_length, args.num_head, args.head_dim)
    
    elif args.kernel == 'dist':
        benchmark_dist_attn(args.batch_size, args.seq_length, args.num_head, args.head_dim)
