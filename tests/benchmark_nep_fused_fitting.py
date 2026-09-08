#!/usr/bin/env python3
"""Bounded original-vs-fused NEP fitting benchmark on real OMat24 batches."""
import argparse, copy, gc, json, os, platform, random, statistics, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def percentile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs)-1, int(round((len(xs)-1)*q)))]


def summary(xs, atoms):
    return {"p50_ms": 1e3*statistics.median(xs), "p95_ms": 1e3*percentile(xs,.95),
            "atoms_per_s": atoms/statistics.median(xs), "raw_seconds": xs}


def checkpoint_state(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj.get("model_state_dict", obj.get("model", obj)))
    state = {k.removeprefix("module."): v for k,v in state.items()}
    shift = obj.get("energy_shift", obj.get("energy_shift_train", None))
    if torch.is_tensor(shift): shift = shift.tolist()
    return obj, state, shift


def make_model(case, device):
    from src.user.input_param import InputParam
    from src.model.nep_net import NEP
    cfg = json.loads((case/"nep.json").read_text())
    params = InputParam(cfg, "TRAIN")
    ckpt, state, shift = checkpoint_state(case/"model_record/nep_model.ckpt")
    ntypes = len(params.atom_type)
    shift = shift if shift is not None and len(shift) == ntypes else [0.0]*ntypes
    qs = state.get("q_scaler")
    model = NEP(params, shift, q_scaler=None if qs is None else qs.cpu().numpy(),
                dtype=torch.float64, device=torch.device("cpu")).double()
    model.load_state_dict(state, strict=True)
    model.use_analytical_nep_grad = True
    return model.to(device), params, cfg


def cpu_batch(params, indices):
    from src.pre_data.nep_lmdb_dataset import NepLmdbDataset
    from src.pre_data.nep_data_loader import variable_length_collate_fn
    ds = NepLmdbDataset(params.file_paths.train_data_path, params.atom_type,
                        params.nep_param.cutoff[0], params.nep_param.cutoff[1])
    return variable_length_collate_fn([ds[i] for i in indices]), len(ds)


def to_cuda(batch):
    return {k: (v.to("cuda") if hasattr(v, "to") else v) for k,v in batch.items()}


def neighbor_inputs(model, sample):
    from src.utils.op_loader import load_calc_ops
    ops = load_calc_ops()
    nr, na = ops.calculate_maxneigh(sample["num_atom"], sample["box"], sample["box_original"],
        sample["num_cell"], sample["position"], model.cutoff_radial, model.cutoff_angular,
        len(model.atom_type), sample["atom_type_map"], False)
    nr, na = max(10,int(nr.max())), max(10,int(na.max()))
    a,b,c,d,e,f = ops.calculate_neighbor(sample["num_atom"], sample["atom_type_map"],
        model.atom_type_device-1, sample["box"], sample["box_original"], sample["num_cell"],
        sample["position"], model.cutoff_radial, model.cutoff_angular, nr, na, True)
    return a,c,e,b,d,f,sample["num_atom"],sample["atom_type_map"]


def model_outputs(model, batch, fused):
    neigh = neighbor_inputs(model, batch)
    return model(*neigh, charge_label=batch.get("charge"), position=batch["position"],
        box_original=batch["box_original"], volume=batch["volume"], need_force=True,
        need_bec=bool(model.charge_mode), need_charge_virial=True, need_charge_energy=True,
        fitting_groups=batch["fitting_groups"] if fused else None)


def loss_fn(outputs, batch):
    etot, _, force, _, virial = outputs[:5]
    loss = (etot.reshape(-1)-batch["energy"].reshape(-1)).square().mean()
    if force is not None: loss = loss + 100.0*(force.reshape(-1,3)-batch["force"].reshape(-1,3)).square().mean()
    if virial is not None:
        target = batch.get("virial", batch.get("stress"))
        if target is not None:
            pred=virial.reshape(target.shape); mask=target.abs()<1e5
            if mask.any(): loss = loss + .1*(pred[mask]-target[mask]).square().mean()
    return loss


def full_trial(base, cpu, fused, warmup, steps):
    from src.model.nep_fused_fitting import build_fitting_groups
    gc.collect(); torch.cuda.empty_cache()
    model=copy.deepcopy(base).to('cuda'); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    times=[]; prep=[]; atoms=int(cpu["atom_type_map"].numel())
    torch.cuda.reset_peak_memory_stats(); free0,total=torch.cuda.mem_get_info()
    for i in range(warmup+steps):
        torch.cuda.synchronize(); start=time.perf_counter()
        t=time.perf_counter(); prepared={k:v for k,v in cpu.items() if k != "fitting_groups"}
        if fused: prepared["fitting_groups"]=build_fitting_groups(cpu["atom_type_map"])
        group_t=time.perf_counter()-t if fused else 0.0
        batch=to_cuda(prepared); prep_t=time.perf_counter()-t
        opt.zero_grad(set_to_none=True); out=model_outputs(model,batch,fused)
        loss = loss_fn(out,batch)
        if i == 0 and not torch.isfinite(loss): raise RuntimeError("non-finite initial training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); torch.cuda.empty_cache(); opt.step()
        torch.cuda.synchronize(); elapsed=time.perf_counter()-start
        if i>=warmup: times.append(elapsed); prep.append((prep_t,group_t))
    free1,_=torch.cuda.mem_get_info()
    ans=summary(times,atoms); ans.update(cpu_prepare_ms=1e3*statistics.median(x[0] for x in prep),
        cpu_group_ms=1e3*statistics.median(x[1] for x in prep),
        peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
        mem_get_info_delta=free0-free1)
    if not torch.isfinite(loss): raise RuntimeError("non-finite final training loss")
    del model,opt,batch,out,loss; gc.collect(); torch.cuda.empty_cache()
    return ans


def fitting_trial(base, atom_types, fused, D, H, warmup, steps, seed):
    from src.model.nep_fitting import FittingNet
    from src.model.nep_fused_fitting import build_fitting_groups
    from src.model.nep_net import NEP
    gc.collect(); torch.cuda.empty_cache()
    torch.manual_seed(seed); n=atom_types.numel(); nets=[]
    for _ in range(len(base.fitting_net)):
        nets.append(FittingNet([H,1],True,False,"tanh",D,0.0,False,None,True).double().cuda())
    fitting_model = SimpleNamespace(fitting_net=nets, neuron=[H, 1],
                                    charge_mode=0, dtype=torch.float64)
    types_cuda=atom_types.to("cuda"); groups=build_fitting_groups(atom_types.cpu()).to("cuda"); X=torch.randn(n,D,dtype=torch.float64,device="cuda")
    times={"forward":[],"backward":[]}
    torch.cuda.reset_peak_memory_stats(); free0,total=torch.cuda.mem_get_info()
    for i in range(warmup+steps):
        for net in nets: net.zero_grad(set_to_none=True)
        X=X.detach().requires_grad_(True); torch.cuda.synchronize(); t=time.perf_counter()
        y,_,g,_=NEP.calculate_Ei_with_grad(
            fitting_model, types_cuda, X, X.device,
            fitting_groups=groups if fused else None)
        torch.cuda.synchronize(); f=time.perf_counter()-t
        scalar=y.square().mean()+g.square().mean(); torch.cuda.synchronize(); t=time.perf_counter(); scalar.backward(); torch.cuda.synchronize(); b=time.perf_counter()-t
        if i>=warmup: times["forward"].append(f); times["backward"].append(b)
    result={k:summary(v,n) for k,v in times.items()}; free1,_=torch.cuda.mem_get_info()
    result["memory"]={"peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved(),"mem_get_info_delta":free0-free1}
    return result


def compare_real_batch(base, cpu):
    """Check the actual checkpoint/batch before interpreting timing results."""
    sample = to_cuda(cpu)
    models = [copy.deepcopy(base).to('cuda'), copy.deepcopy(base).to('cuda')]
    neighbors = neighbor_inputs(models[0], sample)
    optimizers = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in models]
    outputs = []
    for fused, model in zip((False, True), models):
        out = model(*neighbors, charge_label=sample.get('charge'), position=sample['position'],
                    box_original=sample['box_original'], volume=sample['volume'],
                    need_force=True, need_bec=bool(model.charge_mode),
                    fitting_groups=sample['fitting_groups'] if fused else None)
        loss_fn(out, sample).backward()
        outputs.append(out)
    error = 0.0
    for expected, actual in zip(*outputs):
        if expected is not None:
            torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)
            error = max(error, float((actual-expected).abs().max()))
    for (name, expected), (_, actual) in zip(models[0].named_parameters(), models[1].named_parameters()):
        if expected.grad is None:
            if actual.grad is not None: raise AssertionError('unexpected gradient for ' + name)
        else:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-7, atol=2e-8)
    for model, optimizer in zip(models, optimizers):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    for expected, actual in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)
    del models, optimizers, outputs, sample, neighbors
    gc.collect(); torch.cuda.empty_cache()
    return {'passed': True, 'max_absolute_output_error': error,
            'checks': ['outputs', 'all_parameter_gradients', 'Adam_update']}


def capacity_scan(args, pool_size, save_progress):
    """Each trial gets a fresh CUDA process, including after native allocator OOM."""
    root = args.output.parent / (args.output.stem + '-capacity')
    root.mkdir(parents=True, exist_ok=True)
    trials = []
    bounds = {name: {'largest_success': 0, 'first_oom': None} for name in ('original', 'fused')}

    def trial(name, count):
        output = root / ('{}-{}.json'.format(name, count))
        log = output.with_suffix('.log')
        output.unlink(missing_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), '--case', str(args.case),
                   '--warmup', '2', '--steps', '3', '--batch-size', str(count),
                   '--seed', str(args.seed), '--pool-size', str(pool_size),
                   '--capacity-worker', name, '--output', str(output)]
        with log.open('w') as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if output.exists():
            record = json.loads(output.read_text())
        else:
            message = log.read_text(errors='replace').lower()
            if 'out of memory' not in message:
                raise RuntimeError('capacity trial failed; inspect ' + str(log))
            record = {'status': 'oom', 'process_returncode': completed.returncode}
        record.update(backend=name, requested_structures=count, log=str(log))
        trials.append(record)
        if record['status'] == 'ok': bounds[name]['largest_success'] = count
        elif record['status'] == 'oom': bounds[name]['first_oom'] = count
        else: raise RuntimeError('invalid capacity trial result: ' + str(record))
        save_progress({'trials': trials, 'bounds': bounds, 'cap': args.max_batch})
        return record['status'] == 'ok'

    for name in bounds:
        count = min(args.batch_size, args.max_batch)
        while trial(name, count) and count < args.max_batch:
            count = min(2 * count, args.max_batch)
        low, high = bounds[name]['largest_success'], bounds[name]['first_oom']
        if high is not None:
            while high-low > 1:
                midpoint = (low+high)//2
                if trial(name, midpoint): low = midpoint
                else: high = midpoint
    return {'trials': trials, 'bounds': bounds, 'cap': args.max_batch,
            'scope': 'nested seeded structure prefixes; fresh process, two warmup and three measured steps'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",required=True,type=Path); p.add_argument("--warmup",type=int,default=20)
    p.add_argument("--steps",type=int,default=100); p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--max-batch",type=int); p.add_argument("--output",type=Path,required=True); p.add_argument("--seed",type=int,default=2023)
    p.add_argument('--capacity-worker', choices=('original', 'fused'), help=argparse.SUPPRESS)
    p.add_argument('--capacity-coordinator', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--pool-size', type=int, help=argparse.SUPPRESS)
    a=p.parse_args();
    if a.steps < 1 or a.warmup < 0 or a.batch_size < 1 or (a.max_batch is not None and a.max_batch < 1):
        p.error('steps/batch sizes must be positive and warmup non-negative')
    if a.capacity_coordinator:
        result=json.loads(a.output.read_text())
        def save_progress(scan):
            result['max_batch_scan']=scan
            a.output.write_text(json.dumps(result,indent=2,default=str)+'\n')
        save_progress(capacity_scan(a,a.pool_size,save_progress))
        print('Saved measurements and capacity bounds to',a.output,flush=True)
        return
    if not torch.cuda.is_available(): p.error("a real CUDA GPU allocation is required")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    # Keep the checkpoint template on CPU so each measurement has exactly one
    # live GPU model (the correctness comparison temporarily needs two).
    base,params,cfg=make_model(a.case,torch.device("cpu")); _,size=cpu_batch(params,[0])
    pool_size=min(size,max(a.batch_size,a.max_batch or 0,a.pool_size or 0)); pool=random.Random(a.seed).sample(range(size),pool_size)
    indices=pool[:min(a.batch_size,size)]; cpu,_=cpu_batch(params,indices)
    atom_types=cpu["atom_type_map"]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.capacity_worker:
        try:
            measurement=full_trial(base,cpu,a.capacity_worker=='fused',a.warmup,a.steps)
            record={'status':'ok','structures':len(indices),'atoms':int(atom_types.numel()),'measurement':measurement}
        except torch.cuda.OutOfMemoryError:
            record={'status':'oom','structures':len(indices),'atoms':int(atom_types.numel())}
        a.output.write_text(json.dumps(record,indent=2)+'\n')
        return
    result={"config":vars(a)|{"case":str(a.case),"output":str(a.output)},"seed":a.seed,
      "gpu":{"name":torch.cuda.get_device_name(),"torch":torch.__version__,"cuda":torch.version.cuda,"platform":platform.platform()},
      "dtype":"float64","model":{"D":int(base.feature_nums),"H":int(base.neuron[0]),"types":len(base.atom_type)},
      "batch":{"structures":len(indices),"atoms":int(atom_types.numel())},"fitting":{},"complete_step":{}}
    result['correctness']=compare_real_batch(base,cpu)
    result['complete_step_scope']='preloaded structures; includes grouping/H2D, neighbors, loss/backward, clip, empty_cache and Adam; excludes LMDB I/O'
    result["fitting_input"]="deterministic synthetic descriptors with empirical real-batch type counts"
    for D,H,label in [(int(base.feature_nums),int(base.neuron[0]),"synthetic_model_DH"),(int(base.feature_nums),60,"synthetic_model_D_H60"),(96,100,"synthetic_D96_H100")]:
        result["fitting"][label]={b:fitting_trial(base,atom_types,b=="fused",D,H,a.warmup,a.steps,a.seed) for b in ("original","fused")}
    result["complete_step"]={b:full_trial(base,cpu,b=="fused",a.warmup,a.steps) for b in ("original","fused")}
    a.output.write_text(json.dumps(result,indent=2,default=str)+'\n')
    if a.max_batch:
        # Release this process's entire CUDA context before measuring capacity.
        # A CPU coordinator then waits for one fresh GPU worker at a time.
        command=[sys.executable,str(Path(__file__).resolve()),'--capacity-coordinator',
                 '--case',str(a.case),'--output',str(a.output),'--max-batch',str(a.max_batch),
                 '--batch-size',str(a.batch_size),'--seed',str(a.seed),'--pool-size',str(pool_size)]
        os.execv(sys.executable,command)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,default=str)+"\n")
    print(json.dumps(result,indent=2,default=str))

if __name__=="__main__": main()
