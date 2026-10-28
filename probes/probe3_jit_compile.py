import torch
import torch_npu
import torch.nn.functional as F
torch.npu.set_device(0)
D = torch.device("npu:0")
torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)  # 错误信息建议的开关
print("soc:", torch.npu.get_soc_version())

results=[]
def probe(name, fn):
    try: fn(); results.append((name,"OK",""))
    except Exception as e: results.append((name,"FAIL",str(e).replace("\n"," ")[:100]))

def t(*s, dt=torch.float16): return torch.randn(*s, dtype=dt, device=D)

probe("fft.fft(jit=False)", lambda: torch.fft.fft(torch.randn(256, dtype=torch.float32, device=D)))
probe("fft.rfft(jit=False)", lambda: torch.fft.rfft(torch.randn(256, dtype=torch.float32, device=D)))
probe("fft.irfft(jit=False)", lambda: torch.fft.irfft(torch.fft.rfft(torch.randn(256, dtype=torch.float32, device=D))))
def stft_():
    torch.stft(torch.randn(16000, dtype=torch.float32, device=D), 1024, 256,
               window=torch.hann_window(1024, device=D), return_complex=True)
probe("stft(jit=False)", stft_)
def istft_():
    spec = torch.stft(torch.randn(8000, dtype=torch.float32, device=D), 512, 128,
                      window=torch.hann_window(512, device=D), return_complex=True)
    torch.istft(spec, 512, 128, window=torch.hann_window(512, device=D))
probe("istft(jit=False)", istft_)
probe("complex.abs(jit=False)", lambda: torch.view_as_complex(torch.randn(4,8,2,device=D)).abs().sum())
probe("polar(jit=False)", lambda: torch.polar(torch.ones(8,dtype=torch.float32,device=D), torch.zeros(8,dtype=torch.float32,device=D)))
probe("bincount(int32)", lambda: torch.bincount(torch.randint(0,10,(100,),device=D,dtype=torch.int32)))
def sdpa4d_():
    q,k,v = t(1,8,32,64),t(1,8,32,64),t(1,8,32,64)
    F.scaled_dot_product_attention(q,k,v,is_causal=True)
probe("sdpa(4D,jit=False)", sdpa4d_)
# 回归确认
probe("matmul回归", lambda: (t(64,64)@t(64,64)).sum())
probe("conv1d回归", lambda: F.conv1d(t(1,64,100), t(128,64,5), padding=2))
probe("eager_attn回归", lambda: (F.softmax(((t(1,8,32,64)@t(1,8,64,32))*(1/8.0)).masked_fill(torch.ones(32,32,dtype=torch.bool,device=D).tril()==False, float("-inf")),dim=-1)@t(1,8,32,64)).sum())

ok=[r for r in results if r[1]=="OK"]; fail=[r for r in results if r[1]=="FAIL"]
print(f"===== TOTAL {len(results)} | OK {len(ok)} | FAIL {len(fail)} =====")
for n,s,m in fail: print(f"[FAIL] {n}: {m}")
print("OK:", ", ".join(n for n,_,_ in ok))
