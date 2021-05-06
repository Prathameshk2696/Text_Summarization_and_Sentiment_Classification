import linecache
import torch
import torch.utils.data as torch_data
from random import Random

class LabelDataset(torch_data.Dataset):
    def __init__(self, infos, indexes=None):
        self.srcF = infos['source_file']
        self.tgtF = infos['target_file']
        self.labF = infos['label_file']
        self.original_srcF = infos['original_source_file']
        self.original_tgtF = infos['original_target_file']
        self.length = infos['length']
        self.infos = infos
        
        if indexes is None:
            self.indexes = list(range(self.length))
        else:
            self.indexes = indexes

    def __getitem__(self, index):
        index = self.indexes[index]
        src = list(map(int, linecache.getline(self.srcF, index+1).strip().split()))
        tgt = list(map(int, linecache.getline(self.tgtF, index+1).strip().split()))
        label = int(float(linecache.getline(self.labF, index+1).strip()))-1
        
        original_src = linecache.getline(self.original_srcF, index+1).strip().split()
        original_tgt = linecache.getline(self.original_tgtF, index+1).strip().split()

        return src, tgt, label, original_src, original_tgt

    def __len__(self):
        return len(self.indexes)
    
def label_padding(data):
    src, tgt, label, original_src, original_tgt = zip(*data)

    src_len = [len(s) for s in src]
    src_pad = torch.zeros(len(src), max(src_len)).long()
    
    for i, s in enumerate(src):
        end = src_len[i]
        src_pad[i, :end] = torch.LongTensor(s)

    tgt_len = [len(s) for s in tgt]
    tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
    
    for i, s in enumerate(tgt):
        end = tgt_len[i]
        tgt_pad[i, :end] = torch.LongTensor(s)

    return src_pad, tgt_pad, torch.LongTensor(label), \
           torch.LongTensor(src_len), torch.LongTensor(tgt_len), \
           original_src, original_tgt