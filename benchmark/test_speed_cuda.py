import compressor
import torch
import sys
import time

torch.cuda.set_device(0)

a = torch.randn(22,100,224,224)
a = a.cuda()

#print('before sending')
#print(a)

compressor = compressor.compressor(using_cuda = True)

# warmups to amortize allocation costs
c,size = compressor.compress(a)
d = compressor.uncompress(c,size)
del c, size, d
c,size = compressor.compress(a)
d = compressor.uncompress(c,size)
del c, size, d

# benchmark
torch.cuda.synchronize()
start = time.time()
c,size = compressor.compress(a)
torch.cuda.synchronize()
end = time.time()
print('Compression time cost')
print(str(end - start))
#print('during sending')
#print(c)
torch.cuda.synchronize()
start = time.time()
d = compressor.uncompress(c,size)
torch.cuda.synchronize()
end = time.time()
print('Uncompression time cost')
print(str(end - start))
#print('after sending')
#print(d)
