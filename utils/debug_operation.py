import torch
import gc
import psutil

def check_cuda_memory(epoch, num_epochs, types, empty=True):
    if empty:
        torch.cuda.empty_cache()  # Clear the cache to obtain accurate memory usage information.
    allocated_memory = torch.cuda.memory_allocated() / 1024**3  # Allocated memory, converted to GB
    cached_memory = torch.cuda.memory_reserved() / 1024**3  # cached memory, converted to GB
    log = ""
    log += f"{types} cuda memory:"
    log += f"Epoch [{epoch}/{num_epochs}]:"
    log += f"alloc: {allocated_memory:.4f} GB"
    log += f"cached: {cached_memory:.4f} GB"
    print(log)

def find_tensor_memory():
    # 获取所有对象的列表
    objects = gc.get_objects()
    # 过滤出PyTorch张量
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    # 打印张量的大小和内存占用
    for t in tensors:
        print("Size:", t.size(), "Memory:", t.element_size() * t.nelement())

def check_cpu_memory(prefix_info:str=None):
    if prefix_info is not None:
        print(prefix_info)
    # 获取内存信息
    mem = psutil.virtual_memory()
    # 转换为 GB（可选）
    total_gb = mem.total / (1024 ** 3)
    available_gb = mem.available / (1024 ** 3)
    used_gb = mem.used / (1024 ** 3)
    print(f"ALL MEMORY: {total_gb:.5f} GB; AVAIBLE MEMORY: {available_gb:.5f} GB; ALLOCATED MEMORY: {used_gb:.5f} GB; RATE of MEMORY : {mem.percent}%")