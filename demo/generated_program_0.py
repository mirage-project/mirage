import triton
import torch
import triton.language as tl

def kernel_launcher(dtensor20014, dtensor20015, dtensor20016, dtensor20017, dtensor20018, dtensor20019, dtensor20020, ):

def main():
	dtensor20014 = torch.randn((16, 256), dtype=torch.float16, device="cuda", requires_grad=False)
	dtensor20015 = torch.randn((256, 4096), dtype=torch.float16, device="cuda", requires_grad=False)
	dtensor20016 = torch.randn((256, 16), dtype=torch.float16, device="cuda", requires_grad=False)
	dtensor20017 = torch.randn((16, 4096), dtype=torch.float16, device="cuda", requires_grad=False)
	dtensor20018 = torch.randn((16, 16), dtype=torch.float16, device="cuda", requires_grad=False)
	dtensor20019 = torch.randn((16, 4096), dtype=torch.float16, device="cuda", requires_grad=False)
	dtensor20020 = torch.randn((16, 4096), dtype=torch.float16, device="cuda", requires_grad=False)
	fn = lambda: kernel_launcher(dtensor20014, dtensor20015, dtensor20016, dtensor20017, dtensor20018, dtensor20019, dtensor20020, )
	quantiles = [0.5, 0.1, 0.9]
	ms, mmin, mmax = triton.testing.do_bench(fn, warmup=1000, rep=1000, quantiles=quantiles)
	print(ms, mmin, mmax)

if __name__ == "__main__":
	main()
