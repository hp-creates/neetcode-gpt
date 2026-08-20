import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_val = np.finfo(np.float64).min
        max_val = np.max(z)
        z = z-max_val
        z = pow(math.e, z)
        sum_val = np.sum(z)
        res = z/sum_val
        return np.round(res, 4)

