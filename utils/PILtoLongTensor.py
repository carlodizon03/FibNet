
import torch
import PIL.Image as Image
import numpy as np

class PILToLongTensor(object):
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError("img should be PIL Image. Got {}".format(type(img)))
    
        img_byte = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))

        n_channels = len(img)

        img = img_byte.view(img.size[1], img.size[0], n_channels)

        return img.transpose(0, 1).transpose(0, 2).contiguous().long().squeeze_()
