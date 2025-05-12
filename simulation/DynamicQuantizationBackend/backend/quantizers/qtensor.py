import torch

class QTensor:
    def __init__(
        self, tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ):
        self.tensor = tensor
        self.scale = scale
        self.zero_point = zero_point

    def view(self, *shape):
        """Returns a new Qtensor instance with a reshaped tensor."""
        return QTensor(self.tensor.view(*shape), self.scale, self.zero_point)

    def size(self, dim: None = None):
        """Returns the size of the underlying tensor."""
        return self.tensor.size(dim)

    @property
    def shape(self):
        """Returns the shape of the underlying tensor (alternative to size())."""
        return self.tensor.shape

    def clone(self):
        """Returns a deep copy of the Qtensor."""
        return QTensor(
            self.tensor.clone(),
            self.scale.clone(),
            self.zero_point.clone() if self.zero_point is not None else None,
        )

    def to(self, device):
        """Moves the Qtensor to a specified device (CPU/GPU)."""
        return QTensor(
            self.tensor.to(device),
            self.scale.to(device),
            self.zero_point.to(device) if self.zero_point is not None else None,
        )

    def __add__(self, other):
        #TODO
        return NotImplemented

    def isSimmetric(self):
        """Checks if the Qtensor is symmetric (zero_point is None)."""
        return self.zero_point is None

    def __repr__(self):
        return f"Qtensor(tensor={self.tensor}, scale={self.scale}, zero_point={self.zero_point})"
