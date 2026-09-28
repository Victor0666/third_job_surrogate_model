import psutil
import subprocess

def get_container_cpu_count():
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().strip().split()
            if quota != "max":
                return int(quota) / int(period)
    except:
        pass
    
    return psutil.cpu_count(logical=True)

def get_gpu_memory():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        lines = result.strip().split('\n')
        if lines:
            used, total = lines[0].split(', ')
            return f"{used}MB / {total}MB"
        return "N/A"
    except Exception:
        return "N/A"

def monitor():
    total_cores = get_container_cpu_count()
    while True:
        cpu_percent = psutil.cpu_percent(interval=1)
        used_cores = (cpu_percent / 100.0) * total_cores
        
        mem = psutil.virtual_memory()
        used_mem_mb = int(mem.used / (1024 * 1024))
        total_mem_mb = int(mem.total / (1024 * 1024))
        
        gpu_info = get_gpu_memory()
        
        output = f"CPU Usage: {used_cores:.1f} / {total_cores}, GPU Usage: {gpu_info}, Mem Usage: {used_mem_mb}MB / {total_mem_mb}MB"
        print(output)

if __name__ == "__main__":
    monitor()
