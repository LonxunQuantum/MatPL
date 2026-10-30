"""在已有 onescience311 HIP 构建上增量构建 fitting HIPRTC；不使用复制前的 CMakeCache。"""
from pathlib import Path
import argparse, hashlib, json, shlex, subprocess, shutil
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('output',type=Path)
p.add_argument('--reuse-fitting-object',type=Path)
p.add_argument('--reuse-fitting-sha',type=Path)
a=p.parse_args();source=Path(__file__).resolve().parents[1];base=source/'src/op/build/hip';out=a.output.resolve();build=out/'objects';lib=out/'lib';build.mkdir(parents=True,exist_ok=True);lib.mkdir(exist_ok=True)
entries=json.loads((base/'compile_commands.json').read_text());old=str(Path(next(e['file'] for e in entries if e['file'].endswith('/nep_fitting_launcher.cpp'))).parents[3]);cwd=base/'cmake/hip';objects={};commands=[]
for name in ['nep_fitting.hip','nep_fitting_launcher.cpp','nep_fitting_hiprtc.cpp','CalcOps_bind.cpp']:
 template='nep_fitting_launcher.cpp' if name=='nep_fitting_hiprtc.cpp' else name
 e=next(e for e in entries if e['file'].endswith('/'+template));args=shlex.split(e['command'].replace(old,str(source)));src=source/'src/op'/('kernel_hip' if name.endswith('.hip') else 'src')/name;obj=build/(name+'.o');args[-1]=str(src);args[args.index('-o')+1]=str(obj)
 if name.endswith('.hip'):
  args=[x for x in args if not x.startswith('--offload-arch=')];args.insert(1,'--offload-arch=gfx936')
 args.extend(['-I/opt/dtk-25.04.4/hip/include'])
 headers=[source/'src/op/kernel/nep_fitting_jit_source.h',*sorted((source/'src/op/include').glob('*.h'))]
 key=hashlib.sha256(src.read_bytes()+json.dumps(args).encode()+b''.join(f.read_bytes() for f in headers)).hexdigest();stamp=obj.with_suffix('.sha256')
 if name=='nep_fitting.hip' and a.reuse_fitting_object and a.reuse_fitting_sha:
  assert a.reuse_fitting_sha.read_text().strip()==hashlib.sha256(src.read_bytes()).hexdigest(), '复用对象与当前源码不匹配'
  shutil.copy2(a.reuse_fitting_object,obj);stamp.write_text(key)
 if not (obj.exists() and stamp.exists() and stamp.read_text()==key):
  print('COMPILE',name,flush=True);subprocess.run(args,check=True);stamp.write_text(key)
 else:print('REUSE',name,flush=True)
 objects[name]=str(obj);commands.append(args)
for target in ['CalcOps_cuda','CalcOps_bind']:
 args=shlex.split((cwd/f'CMakeFiles/{target}.dir/link.txt').read_text().replace(old,str(source)))
 for i,x in enumerate(args):
  if x.endswith('.o'):
   filename=Path(x).name.removesuffix('.o');args[i]=objects.get(filename,str((cwd/x).resolve()))
  elif x=='../../lib/libCalcOps_cuda.so':args[i]=str(lib/'libCalcOps_cuda.so')
  elif x.startswith('-Wl,-rpath,'):
   paths=[z for z in x[len('-Wl,-rpath,'):].split(':') if not z.endswith('/src/op/build/hip/lib')];args[i]='-Wl,--enable-new-dtags,-rpath,$ORIGIN:'+':'.join(paths)
 args[args.index('-o')+1]=str(lib/f'lib{target}.so')
 if target=='CalcOps_cuda':args.append(objects['nep_fitting_hiprtc.cpp'])
 print('LINK',target,flush=True);subprocess.run(args,cwd=cwd,check=True);commands.append(args)
files=['src/op/src/nep_fitting_hiprtc.cpp','src/op/src/nep_fitting_launcher.cpp','src/op/src/CalcOps_bind.cpp','src/op/kernel_hip/nep_fitting.hip','src/op/kernel/nep_fitting_jit_source.h','src/model/nep_fused_fitting.py','src/op/cmake/hip/CMakeLists.txt','src/op/include/nep_fitting_hip.h','src/op/include/nep_fitting_jit.h','src/op/include/nep_fitting_limits.h','src/op/include/calculate_nepfitting.h']
manifest={'source_root':str(source),'backend':'HIPRTC','architecture':'gfx936','source_sha256':{f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in files},'library_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in lib.glob('*.so')},'commands':commands,'note':'未改动的其他算子复用同源 OneScience 对象；全新构建使用已更新的 CMake。'}
(out/'build_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n');print('BUILD PASS',lib,flush=True)
