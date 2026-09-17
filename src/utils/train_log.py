from enum import Enum
import torch

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3
    ROOT = 4


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE, device=None, world_size=1):
        self.device = device
        self.world_size = world_size
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0 #rmse
        self.avg = 0 # mse
        self.sum = 0 # mse sum
        self.count = 0
        self.root = 0 # rmse avg
    
    def update(self, val, n=1):
        if n > 0: # for same data, such as virial, some images do not have virial datas, the n will be 0
            if self.summary_type is Summary.AVERAGE:
                self.val = val
                self.sum += val * n
                self.count += n
                self.avg = self.sum / self.count
                self.root = self.avg
            else:
                self.val = val**0.5
                self.sum += val * n
                self.count += n
                self.avg = self.sum / self.count
                self.root = self.avg**0.5

    def all_reduce(self):
        if not torch.distributed.is_initialized():
            return
        total = torch.tensor([self.root, self.val, self.avg],
                             dtype=torch.float32, device=self.device)
        if torch.distributed.get_backend() == "gloo":
            torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
            total /= torch.distributed.get_world_size()
        else:
            torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.AVG)
        self.root, self.val, self.avg = total.tolist()

    def __str__(self):
        if self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        else:
            fmtstr = "{name} {val" + self.fmt + "} ({root" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        elif self.summary_type is Summary.ROOT:
            fmtstr = "{name} {root:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def sync_meters(self):
        for meter in self.meters:
            meter.all_reduce()

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries), flush=True)

    def display_summary(self, entries=[" *"]):
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
