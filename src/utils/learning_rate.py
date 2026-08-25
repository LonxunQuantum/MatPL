import math


def calculate_lr_scale(
        scale_lr, scaling_method, local_batch_size, world_size,
        avg_atom_nums=1.0):
    """Resolve the one-time learning-rate scale used at initialization."""
    if not scale_lr:
        return 1.0
    if local_batch_size <= 0 or world_size <= 0:
        raise ValueError("local_batch_size and world_size must be positive")

    methods = {
        "linear_gpu": float(world_size),
        "sqrt_batch": math.sqrt(local_batch_size),
        "sqrt_gpu": math.sqrt(world_size),
        "sqrt": math.sqrt(local_batch_size * world_size),
        "sqrt_batch_gpu_atom": math.sqrt(
            local_batch_size * world_size * avg_atom_nums),
    }
    if scaling_method not in methods:
        valid = ", ".join(sorted(methods))
        raise ValueError(
            f"Unsupported scaling_method '{scaling_method}'. "
            f"Expected one of: {valid}.")
    return methods[scaling_method]


def resolve_optimizer_peak_lr(learning_rate, lr_scale):
    """Return the actual peak LR that is installed on the optimizer."""
    return learning_rate * lr_scale


def optimizer_update_step(completed_updates, batch_index):
    """Return a zero-based update index from a persisted update count."""
    return completed_updates + batch_index


def calculate_loss_weight_progress(global_update, stop_step):
    """Return linear loss-weight progress in [0, 1] by optimizer update."""
    if stop_step <= 0:
        raise ValueError("stop_step must be positive")
    return min(max(global_update / stop_step, 0.0), 1.0)


def calculate_warmup_lr(
        global_update, warmup_updates, start_lr, optimizer_peak_lr):
    """Linearly warm up by optimizer update, independent of epoch size."""
    if warmup_updates <= 0:
        return optimizer_peak_lr
    progress = min(max(global_update / warmup_updates, 0.0), 1.0)
    return start_lr + progress * (optimizer_peak_lr - start_lr)


def calculate_restart_epoch(T_0, T_mult, max_epochs):
    """
    Calculate the epoch numbers before each restart in CosineAnnealingWarmRestarts.
    Epochs start from 1.
    
    Parameters:
    T_0: Number of epochs for the first period
    T_mult: Period multiplier factor
    max_epochs: Maximum number of epochs
    
    Returns:
    restart_epochs: List of epoch numbers before each restart
    """
    import math
    
    restart_epochs = []
    current_epoch = 0
    current_T = T_0
    while current_epoch < max_epochs:
        # Record the end of this period (before restart)
        current_epoch += current_T
        if current_epoch > max_epochs:
            break
        restart_epochs.append(current_epoch)  # Last epoch before restart, starting from 1
        # Update the length of the next period
        current_T = math.ceil(current_T * T_mult)
    return restart_epochs

def is_epoch_before_restart(T_0, T_mult, current_epoch):
    restart_list = calculate_restart_epoch(T_0, T_mult, current_epoch+1)
    if len(restart_list) == 0 or current_epoch != restart_list[-1]:
        return False
    else:
        return True

if __name__=="__main__":
    for i in range(0, 32):
        print(f"restart {i}: {is_epoch_before_restart(2, 2, i)}")
    # print(f"restart: {restart_last_epoch(2, 2, 1)}")
    # print(f"restart: {restart_last_epoch(2, 2, 2)}")
    # print(f"restart: {restart_last_epoch(2, 2, 6)}")
    # print(f"restart: {restart_last_epoch(2, 2, 14)}")
    # print(f"restart: {restart_last_epoch(2, 2, 30)}")
    # print(f"restart: {restart_last_epoch(2, 2, 62)}")
    # print(f"restart: {restart_last_epoch(2, 2, 126)}")
    # print(f"restart: {restart_last_epoch(2, 2, 254)}")
    # print(f"restart: {restart_last_epoch(2, 2, 510)}")
