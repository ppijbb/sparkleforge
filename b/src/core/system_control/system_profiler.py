import psutil

class SystemProfiler:
    def get_metrics(self):
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "battery": psutil.sensors_battery().percent if psutil.sensors_battery() else None
        }
