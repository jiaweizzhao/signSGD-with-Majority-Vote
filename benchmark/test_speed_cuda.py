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
start = time.time()
c,size = compressor.compress(a)
end = time.time()
print('Compression time cost')
print(str(end - start))
#print('during sending')
#print(c)
start = time.time()
d = compressor.uncompress(c,size)
end = time.time()
print('Uncompression time cost')
print(str(end - start))
#print('after sending')
#print(d)