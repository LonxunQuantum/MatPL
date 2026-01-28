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