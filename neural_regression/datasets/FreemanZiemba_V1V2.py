class FreemanZiembaStim(torch.utils.data.Dataset):
    def __init__(
        self,
        stim_directory="/mnt/ceph/users/xzhao/Datasets/FreemanZiemba2013/stim/",
        stim_types=["noise", "tex"],
        im_indices=[99, 38, 327, 71, 393, 13, 48, 336, 402, 18, 52, 23, 56, 60, 30],
        sample_indices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        to_numpy=True,
        transform=None,
    ):
        super().__init__()
        self.stim_directory = stim_directory
        self.stim_types = stim_types
        self.im_indices = im_indices
        self.sample_indices = sample_indices
        self.stim_specs = list(product(stim_types, im_indices, sample_indices))
        self.to_numpy = to_numpy
        self.transform = transform

    def parse_stim_file(self, stim_file):
        name_parts = stim_file.split("-")
        stim_type = name_parts[0]
        im_index = int(name_parts[2][2:])
        sample_index = int(name_parts[3][3:-4])

        return stim_type, im_index, sample_index

    def gen_stim_file(self, stim_type, im_index, sample_index):
        return f"{stim_type}-320x320-im{im_index}-smp{sample_index}.png"

    def __len__(self):
        return len(self.stim_specs)

    def __getitem__(self, idx):
        stim_type, im_index, sample_index = self.stim_specs[idx]
        stim_file = self.gen_stim_file(
            stim_type=stim_type, im_index=im_index, sample_index=sample_index
        )
        stim_path = os.path.join(self.stim_directory, stim_file)
        stim = Image.open(stim_path)
        if self.to_numpy:
            stim = np.array(stim)
        if self.transform:
            stim = self.transform(stim)

        return stim, stim_type, im_index, sample_index

def build_formatted_XY(
    model_responses, neural_responses, stim_types, im_indices, sample_indices
):
    num_cells, num_images, num_stim_types, num_samples = neural_responses.shape
    C, H, W = model_responses.shape[1:]

    X = np.zeros((num_images, num_stim_types, num_samples, C, H, W))
    Y = np.zeros((num_images, num_stim_types, num_samples, num_cells))

    for model_response, stim_type, im_index, sample_index in zip(
        model_responses, stim_types, im_indices, sample_indices
    ):
        stim_type_int = 0 if stim_type == "noise" else 1
        neural_response = neural_responses[
            :,
            NEURAL_DATA_FAMILY_ORDER.index(im_index),
            stim_type_int,
            sample_index - 1,
        ]

        X[NEURAL_DATA_FAMILY_ORDER.index(im_index), stim_type_int, sample_index - 1] = (
            model_response
        )
        Y[NEURAL_DATA_FAMILY_ORDER.index(im_index), stim_type_int, sample_index - 1] = (
            neural_response
        )

    return X, Y