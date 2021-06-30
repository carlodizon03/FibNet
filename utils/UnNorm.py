
class UnNormalize(object):
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.

        Source: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/2
        """
        for idx, image in enumerate(tensor):
            for t, m, s in zip(image, self.mean, self.std):
                t.mul_(s).add_(m)
            tensor[idx] = image
                # The normalize code -> t.sub_(m).div_(s)
        return tensor