import torch
import numpy as np
import json

from data_loader.pastis.dates_management import (
    get_features_by_id,
    number_dates_by_difference,
)


with open("data_loader/pastis/metadata.geojson") as f:

    metadata = json.load(f)


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """

    def __init__(self, label_type="groups", ground_truths=[]):
        self.label_type = label_type
        self.ground_truths = ground_truths

    def __call__(self, sample):
        tensor_sample = {}
        tensor_sample["inputs"] = torch.tensor(sample["img"]).to(torch.float32)
        tensor_sample["labels"] = (
            torch.tensor(sample["labels"][0].astype(np.float32))
            .to(torch.float32)
            .unsqueeze(-1)
        )
        tensor_sample["doy"] = torch.tensor(np.array(sample["doy"])).to(torch.float32)
        return tensor_sample


class Normalize(object):
    """
    Normalize inputs as in https://arxiv.org/pdf/1802.02080.pdf
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """

    def __init__(self):
        self.mean_fold1 = np.array(
            [
                [
                    [[1165.9398193359375]],
                    [[1375.6534423828125]],
                    [[1429.2191162109375]],
                    [[1764.798828125]],
                    [[2719.273193359375]],
                    [[3063.61181640625]],
                    [[3205.90185546875]],
                    [[3319.109619140625]],
                    [[2422.904296875]],
                    [[1639.370361328125]],
                ]
            ]
        ).astype(np.float32)
        self.std_fold1 = np.array(
            [
                [
                    [[1942.6156005859375]],
                    [[1881.9234619140625]],
                    [[1959.3798828125]],
                    [[1867.2239990234375]],
                    [[1754.5850830078125]],
                    [[1769.4046630859375]],
                    [[1784.860595703125]],
                    [[1767.7100830078125]],
                    [[1458.963623046875]],
                    [[1299.2833251953125]],
                ]
            ]
        ).astype(np.float32)

    def __call__(self, sample):
        # print('mean: ', sample['img'].mean(dim=(0,2,3)))
        # print('std : ', sample['img'].std(dim=(0,2,3)))
        sample["inputs"] = (sample["inputs"] - self.mean_fold1) / self.std_fold1
        # sample['doy'] = sample['doy'] / 365.0001
        return sample


class Cut:
    """
    Randomly selects `seq_len` points from inputs and DOY arrays.
    """

    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __call__(self, sample):
        inputs = sample["inputs"]
        doy = sample["doy"]

        total_len = inputs.shape[0]

        # Ensure the requested seq_len does not exceed available length
        if self.seq_len > total_len:
            raise ValueError(
                f"Requested seq_len ({self.seq_len}) exceeds available length ({total_len})."
            )

        # Randomly select and sort the indices for temporal consistency
        indices = torch.randperm(total_len)[: self.seq_len].sort()[0]

        # Cut the inputs and DOY using the selected indices

        cut_inputs = inputs[indices]
        cut_doy = doy[indices]

        # Return the modified sample with the cut inputs and DOY
        sample["inputs"] = cut_inputs
        sample["doy"] = cut_doy

        return sample


class SimpleTransform:
    def __init__(self, seq_len):
        self.to_tensor = ToTensor()
        self.normalize = Normalize()
        self.crop = Crop(
            img_size=128, crop_size=24, random=True, ground_truths=["labels"]
        )
        self.cut = Cut(seq_len)

    def __call__(self, sample):
        tensor_sample = self.to_tensor(sample)
        normalized_sample = self.normalize(tensor_sample)
        cropped_sample = self.crop(normalized_sample)
        cut_sample = self.cut(cropped_sample)
        return cut_sample


class Crop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, img_size, crop_size, random=False, ground_truths=[]):
        self.img_size = img_size
        self.crop_size = crop_size
        self.random = random
        if not random:
            self.top = int((img_size - crop_size) / 2)
            self.left = int((img_size - crop_size) / 2)
        self.ground_truths = ground_truths

    def __call__(self, sample):
        if self.random:
            top = torch.randint(self.img_size - self.crop_size, (1,))[0]
            left = torch.randint(self.img_size - self.crop_size, (1,))[0]
        else:  # center
            top = self.top
            left = self.left
        sample["inputs"] = sample["inputs"][
            :, :, top : top + self.crop_size, left : left + self.crop_size
        ]
        for gt in self.ground_truths:
            sample[gt] = sample[gt][
                top : top + self.crop_size, left : left + self.crop_size
            ]
        return sample


def process_image(image_id, seq_len=37, root_path="."):
    """
    Loads data, retrieves metadata, applies transformations, and prints results.

    Args:
        image_id (str): ID of the image to process.
        seq_len (int): Number of points to randomly select for inputs and DOY.

    Returns:
        dict: The transformed sample containing 'inputs', 'doy', and 'labels'.
    """
    # Initialize the transform with the desired sequence length
    transform = SimpleTransform(seq_len=seq_len)

    # Load inputs and labels
    inputs = np.load(f"{root_path}/DATA_S2/S2_{image_id}.npy")
    labels = np.load(f"{root_path}/ANNOTATIONS/TARGET_{image_id}.npy")[:1]

    # Get the feature with the given ID
    feature = get_features_by_id(metadata, image_id)

    # Compute DOY based on the difference from the start date
    doy = number_dates_by_difference(
        feature["properties"]["dates-S2"], start_date="20180917"
    )

    # Create the sample input dictionary
    sample_input = {"img": inputs, "doy": doy, "labels": labels}

    # Apply the transformation
    transformed_sample = transform(sample_input)

    return (
        transformed_sample["inputs"],
        transformed_sample["labels"],
        transformed_sample["doy"],
    )
