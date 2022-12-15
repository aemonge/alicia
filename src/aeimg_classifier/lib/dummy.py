import time
from termcolor import colored
now = time.time

def print_header(model= 'Dummy', gpu=False):
    """
        Pretty print Header
    """
    gpu_msg = f"GPU { colored('enabled', attrs=['bold']) if gpu else 'disabled' }"
    print(f"Model \"{colored(model, 'blue')}\" Starts with {gpu_msg}")

def m_print(epoch=0, run_time=0.0):
    """
        Pretty print
    """
    msg = colored(f"\n\tEpoch: {epoch}", attrs=['bold'])
    msg += "\tLearn: x"
    msg += f"\tTime: {run_time:.5f}"
    msg += colored("\n\tTest Accuracy\n", attrs=['underline'])
    msg += "\tMean: x/1"
    msg += "\tStd: f(x)"
    msg += f"\tLoss: {-epoch}"
    print(msg)

def dummy(gpu = False):
    """
        Simply a dummy for creating the print and charts to check further models performance.
    """
    timer = time.time()
    print_header(gpu=gpu)

    epochs = 5

    for epoch in range(epochs):
        time.sleep(1)

        m_print(epoch=epoch, run_time=(now() - timer))
        timer = time.time()
