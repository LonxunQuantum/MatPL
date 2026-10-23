import torch,os,sys,json,pathlib,time,importlib.util
spec=importlib.util.spec_from_file_location("aot_ref",str(pathlib.Path(__file__).with_name("test_nep_fused_hip_aot.py")));refmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(refmod)
lib=pathlib.Path(sys.argv[1]);torch.ops.load_library(str(lib/'libCalcOps_cuda.so'));torch.ops.load_library(str(lib/'libCalcOps_bind.so'));ops=torch.ops.CalcOps_cuda
torch.manual_seed(2026);out=[]
shapes=[(32,80,1),(35,40,1),(35,100,1),(82,40,1),(95,60,1),(95,100,1),(103,200,2),(82,40,2),(17,100,1),(1,1,1),(1,200,2),(103,1,1),(103,200,1),(64,17,2),(65,17,2),(96,33,2),(97,33,2),(7,199,1)]
for d,h,q in shapes:
 for counts in [[0,0],[0,19],[17,32],[2135,4552]]:
  n=sum(counts);dtype=torch.float64;kw={'device':'cuda','dtype':dtype}
  x=torch.randn(n,d,**kw)*.2;w=torch.randn(2,d,h,**kw)*.1;b=torch.randn(2,h,**kw)*.1;v=torch.randn(2,h,q,**kw)*.1;c=torch.randn(2,q,**kw)*.1
  ids=torch.randperm(n,device='cuda');offs=torch.tensor([0,counts[0],n],device='cuda');a=torch.randn(q,n,**kw);u=torch.randn(q,n,d,**kw)
  args=(x,w,b,v,c,ids,offs,counts)
  os.environ.update(MATPL_FITTING_GENERIC='1',MATPL_FITTING_LEGACY='1');fref=ops.nep_fitting_forward(*args);bref=ops.nep_fitting_backward(*args,a,u)
  os.environ.pop('MATPL_FITTING_GENERIC');os.environ.pop('MATPL_FITTING_LEGACY');fcand=ops.nep_fitting_forward(*args);bcand=ops.nep_fitting_backward(*args,a,u)
  errors={}
  for name,left,right in zip(['y','g','dx','dw','db','dv','dc'],[*fref,*bref],[*fcand,*bcand]):
   errors[name]=(left-right).abs().max().item() if left.numel() else 0
   torch.testing.assert_close(left,right,rtol=0,atol=1e-10)
  if n == 49:
   values=[z.detach().requires_grad_(True) for z in [x,w,b,v,c]]
   yr,gr=refmod.fitting_reference(*values,ids,offs)
   grads=torch.autograd.grad((yr*a).sum()+(gr*u).sum(),values)
   for name,left,right in zip(['y','g','dx','dw','db','dv','dc'],[yr,gr,*grads],[*fcand,*bcand]):
    errors['pytorch_'+name]=(left-right).abs().max().item() if left.numel() else 0
    torch.testing.assert_close(left,right,rtol=0,atol=1e-10)
  row={'shape':[n,d,h,q],'counts':counts,'max_abs':errors};out.append(row);print(json.dumps(row),flush=True)
pathlib.Path(sys.argv[2]).write_text(json.dumps(out,indent=2)+'\n');print('OPERATOR_CHECK PASS',len(out),flush=True)
