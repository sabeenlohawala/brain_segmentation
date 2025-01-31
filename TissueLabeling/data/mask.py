import torch
import numpy as np


class Mask(object):
    """Randomly mask out one or more patches from an image and its corresponding label mask.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img, labels):
        """
        Args:
            img (Tensor): Tensor image of size (H, W).
            labels (Tensor): Tensor image of size (H, W).
        Returns:
            Tensor: Image and Label Map with the same n_holes of dimension length x length cut
                    out of them.
        """
        h = img.size(0)
        w = img.size(1)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        labels = labels * mask

        return img, labels
