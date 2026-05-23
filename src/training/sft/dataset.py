import os
import glob
import pyarrow as pa
import torch
from torch.utils.data import Dataset


class SFTArrowDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.arrow_files = self._find_arrow_files()
        self.tables = []
        self.cum_lengths = []
        total = 0
        
        for f in self.arrow_files:
            with pa.ipc.open_stream(f) as reader:
                table = reader.read_all()
                self.tables.append(table)
                length = len(table)
                self.cum_lengths.append(total + length)
                total += length
        
        self.total_len = total
    
    def _find_arrow_files(self):
        pattern = os.path.join(self.data_dir, "*.arrow")
        files = glob.glob(pattern)
        return sorted(files)
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        for i, cum_len in enumerate(self.cum_lengths):
            if idx < cum_len:
                if i == 0:
                    local_idx = idx
                else:
                    local_idx = idx - self.cum_lengths[i - 1]
                table = self.tables[i]
                break
        
        row_table = table.slice(local_idx, 1)
        
        input_ids = torch.tensor(row_table["input_ids"][0].as_py(), dtype=torch.long)
        labels = torch.tensor(row_table["labels"][0].as_py(), dtype=torch.long)
        attention_mask = torch.tensor(row_table["attention_mask"][0].as_py(), dtype=torch.long)
        
        return input_ids, labels, attention_mask


def collate_fn(batch):
    input_ids, labels, attention_masks = zip(*batch)
    
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    attention_mask = torch.stack(attention_masks)
    
    return input_ids, labels, attention_mask
