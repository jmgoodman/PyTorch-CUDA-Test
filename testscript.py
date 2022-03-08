# converted to convenient script form
# just run python -m testscript
# it should go without saying that you need a CUDA-enabled PyTorch installation before running this will work
# I mean, that's the whole point of this script, to test that very installation

# Imports
import time
import torch

# functions
def timeFun(f, dim, iterations, device='cpu'):
  iterations = iterations
  t_total = 0
  for _ in range(iterations):
    start = time.time()
    f(dim, device)
    end = time.time()
    t_total += end - start

  if device == 'cpu':
    print(f"time taken for {iterations} iterations of {f.__name__}({dim}, {device}): {t_total:.5f}")
  else:
    print(f"time taken for {iterations} iterations of {f.__name__}({dim}, {device}): {t_total:.5f}")
    
def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
    print("GPU is not available")
  else:
    print("GPU is available")

  return device

def simpleFun(dim, device):
  """
  Args:
    dim: integer
    device: "cpu" or "cuda"
  Returns:
    Nothing.
  """

  # 2D tensor filled with uniform random numbers in [0,1), dim x dim
  x = torch.rand(dim,dim,device=device) # don't use "to". just make it a cuda-native array (with device='cuda') and it'll go way faster because you'll be abolishing the transfer overhead!
  # 2D tensor filled with uniform random numbers in [0,1), dim x dim
  y = torch.rand_like(x,device=device)
  # 2D tensor filled with the scalar value 2, dim x dim
  z = 2*torch.ones_like(x,device=device)

  # elementwise multiplication of x and y
  a = x * y
  # matrix multiplication of x and y
  b = x @ y

  del x
  del y
  del z
  del a
  del b
  
# set device
DEVICE = set_device()

# params for timeFun
dim = 10000
iterations = 1

# test your GPU matrix multiplication against a CPU benchmark
timeFun(f=simpleFun, dim=dim, iterations=iterations, device='cpu')
timeFun(f=simpleFun, dim=dim, iterations=iterations, device=DEVICE)