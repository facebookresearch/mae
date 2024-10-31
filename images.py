import torch
from torch.nn.functional import conv2d


class SpreadMasking:
    '''
        Class to spread the mask
    '''
    def __init__(self, mask_ratio, interpolate_ratio=0.5): 
        self.mask_ratio = mask_ratio
        self.interpolate_ratio = interpolate_ratio 
    
    def shuffle(self, tensors):
        shuffled_tensors = []
        for each in tensors[:, 0].unique():
            same_val_rows = tensors[tensors[:, 0] == each]
            shuffled_rows = same_val_rows[torch.randperm(same_val_rows.size(0))]
            shuffled_tensors.append(shuffled_rows)
        tensors = torch.cat(shuffled_tensors)
        return tensors.unsqueeze(0) if tensors.dim() == 1 else tensors
    
    def mask_value(self, tensors, positions, value):
        batch_indices, channel_indices, x_positions, y_positions = positions.t()
        tensors[batch_indices, channel_indices, x_positions, y_positions] = value
        return tensors

    def create_masks_batch(self, mask_tensors):
        mask_tensors = (mask_tensors > 0).float()
        kernel = torch.ones((1, 1, 16, 16), device=mask_tensors.device)

        output_tensors = conv2d(mask_tensors, kernel, stride=16, padding=0)
        binary_tensors = (output_tensors > 0).float()

        if len(binary_tensors.size()) != 4:
            binary_tensors = binary_tensors.unsqueeze(0)
        self.mask_ratio = -1
        if self.mask_ratio == -1:
            return binary_tensors
        else:
            _, channels, height, width = binary_tensors.shape

            check = torch.sum(binary_tensors, dim=(1, 2, 3))
            zero_indices = (check == 0).nonzero(as_tuple=True)[0]
            if len(zero_indices) > 0:
                random_channels = torch.randint(0, channels, (len(zero_indices),))
                random_heights = torch.randint(0, height, (len(zero_indices),))
                random_widths = torch.randint(0, width, (len(zero_indices),))
                binary_tensors[zero_indices, random_channels, random_heights, random_widths] = 1

            mask_positions = torch.nonzero(binary_tensors == 1, as_tuple=False).squeeze()
            unmask_positions = torch.nonzero(binary_tensors == 0, as_tuple=False).squeeze()

            _, counts_one = torch.unique(mask_positions[:, 0], return_counts=True)

            number_in_mask = (counts_one * self.interpolate_ratio).to(torch.int)
            # If number_in_mask > 196 * ratio => 196 * ratio
            max_mask_limit = int(binary_tensors.size(2) * binary_tensors.size(3) * self.mask_ratio)
            number_in_mask = torch.minimum(number_in_mask, torch.tensor(max_mask_limit))
            number_out_mask = (self.mask_ratio * binary_tensors.size(2) * binary_tensors.size(3) - number_in_mask).to(torch.int)

            mask_positions = self.shuffle(mask_positions)
            unmask_positions = self.shuffle(unmask_positions)

            positions = []
            for i in range(counts_one.size(0)):
                in_mask_position = mask_positions[mask_positions[:, 0] == i][:number_in_mask[i]]
                out_mask_position = unmask_positions[unmask_positions[:, 0] == i][:number_out_mask[i]]

                positions.append(in_mask_position)
                positions.append(out_mask_position)

            positions = torch.cat(positions, dim=0)

            output_tensors = torch.zeros_like(binary_tensors, dtype=torch.float32)
            output_tensors = self.mask_value(output_tensors, positions, 1.0)
            return output_tensors