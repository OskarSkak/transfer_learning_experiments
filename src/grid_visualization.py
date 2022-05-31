def transfer_learning_suite():
    import subprocess
    import time
    import psutil

    for i in range(20):
        ps_pid = subprocess.Popen(["python", "visualization.py", str(i)])
        while ps_pid.poll() is None:
            time.sleep(0.01)

if __name__ == '__main__':
    transfer_learning_suite()