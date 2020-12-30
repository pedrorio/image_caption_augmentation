import torch


class ToTensorTransform:
    def __call__(self, sample):
        x_inputs, x_attention, y_inputs, y_attention = sample

        x_inputs = torch.from_numpy(x_inputs)
        x_attention = torch.from_numpy(x_attention)

        y_inputs = torch.from_numpy(y_inputs)
        y_attention = torch.from_numpy(y_attention)

        return x_inputs, x_attention, y_inputs, y_attention
