class ApplyCosineAperture:
    def __init__(
        self,
        img_size: int = 320,
        input_degrees: int = 4,
        output_degrees: int = 8,
        aperture_degrees: int = 4,
        gray_c: int = 128,
        input_channels: int = 1,
    ):
        self.gray_c = gray_c
        self.input_degrees = input_degrees
        self.aperture_degrees = aperture_degrees
        self.pos = np.array([0, 0])
        self.output_degrees = output_degrees
        self.size_px = np.array([img_size, img_size])

        # Image size
        px_deg = self.size_px[0] / self.input_degrees

        self.size_px_out = (
            self.size_px * (self.output_degrees / self.input_degrees)
        ).astype(int)
        cnt_px = (self.pos * px_deg).astype(int)

        size_px_disp = ((self.size_px_out - self.size_px) / 2).astype(int)

        self.fill_ind = [
            [
                (size_px_disp[0] + cnt_px[0]),
                (size_px_disp[0] + cnt_px[0] + self.size_px[0]),
            ],
            [
                (size_px_disp[1] + cnt_px[1]),
                (size_px_disp[1] + cnt_px[1] + self.size_px[1]),
            ],
        ]

        # Image aperture
        a = self.aperture_degrees * px_deg / 2
        # Meshgrid with pixel coordinates
        x = np.arange(self.size_px_out[1]) - self.size_px_out[1] / 2
        y = np.arange(self.size_px_out[0]) - self.size_px_out[0] / 2
        xv, yv = np.meshgrid(x, y)
        # Raised cosine aperture
        inner_mask = (xv - cnt_px[1]) ** 2 + (yv - cnt_px[0]) ** 2 < a**2
        cos_mask = (
            1
            / 2
            * (
                1
                + np.cos(
                    np.sqrt((xv - cnt_px[1]) ** 2 + (yv - cnt_px[0]) ** 2) / a * np.pi
                )
            )
        )
        cos_mask[np.logical_not(inner_mask)] = 0

        self.cos_mask = cos_mask

    def __call__(self, im):
        im = im - self.gray_c * np.ones(self.size_px)
        im_template = np.zeros(self.size_px_out)

        im_template[
            self.fill_ind[0][0] : self.fill_ind[0][1],
            self.fill_ind[1][0] : self.fill_ind[1][1],
        ] = im
        im_masked = (im_template * self.cos_mask) + self.gray_c * np.ones(
            self.size_px_out
        )

        return im_masked.astype(np.float32)