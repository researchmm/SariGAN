import io
import lmdb
from PIL import Image

import torch 
from torchvision import transforms
import random

_transform = transforms.Compose([
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
)

class MultiResolutionDataset(torch.utils.data.Dataset):
  def __init__(self, path):
    self.env = lmdb.open(path, max_readers=32, readonly=True, lock=False,
      readahead=False, meminit=False,)

    if not self.env:
      raise IOError('Cannot open lmdb dataset', path)

    with self.env.begin(write=False) as txn:
      self.length = int(txn.get('total'.encode('utf-8')).decode('utf-8'))
      self.width = int(txn.get('width'.encode('utf-8')).decode('utf-8')) 
      self.height = int(txn.get('height'.encode('utf-8')).decode('utf-8'))

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    with self.env.begin(write=False) as txn:
      key = '{}-{}-{}'.format(self.width, self.height, str(index).zfill(7)).encode('utf-8')
      img_bytes = txn.get(key)
    buffer = io.BytesIO(img_bytes)
    img = Image.open(buffer)
    img = _transform(img)
    return img#, random.random()
