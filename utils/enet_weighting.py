import numpy as np

def calculate(dataloader, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper:
        w_class = 1 / (ln(c + p_class)),
    where c is usually 1.02 and p_class is the propensity score of that
    class:
        propensity_score = freq_class / total_pixels.
    References: https://arxiv.org/abs/1606.02147
    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    - num_classes (``int``): The number of classes.
    - c (``int``, optional): AN additional hyper-parameter which restricts
    the interval of values for the weights. Default: 1.02.
    """
    class_count = 0
    total = 0
    for _, batch in enumerate(dataloader):
        gt_mask = batch
        # gt_mask = gt_mask.to('cpu').numpy()
        print(gt_mask)
        flat_mask = gt_mask.flatten()
        class_count += np.bincount(flat_mask, minlength = num_classes)
        # print(len(flat_mask.shape))
        total += len(flat_mask)
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights



