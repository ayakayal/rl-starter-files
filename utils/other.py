import random
import numpy
import torch
import collections


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.cuda.set_device(1) #it was 6
#print('cuda', torch.cuda.current_device())
#torch.cuda.set_device(0) #it was 6
def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d
