import ipywidgets as widgets
import torch
from IPython.display import display

from .degradation import randomly_degrade
from .utils import sample_images, singleton, visualize


@singleton
class GUI:
    def __init__(self, config):
        self.config = config
        # LAYOUT
        self.label_layout = widgets.Layout(width="100px")
        self.slider_layout = widgets.Layout(width="300px")
        self.dropdown_layout = widgets.Layout(width="250px")
        self.checkbox_layout = widgets.Layout(width="180px")
        self.hbox_layout = widgets.Layout(
            display="flex",
            align_items="center",
            justify_content="space-between",
            width="80%",
        )
        # NOISE
        self.label_noise = widgets.Label(value="Noise", layout=self.label_layout)
        self.dropdown_noise = widgets.Dropdown(
            description="Distribution",
            options=["Gaussian", "Uniform"],
            layout=self.dropdown_layout,
        )
        self.slider_noise = widgets.FloatSlider(
            description="Noise Level",
            value=0.5,
            min=0.1,
            max=1,
            step=0.05,
            layout=self.slider_layout,
        )
        self.checkbox_noise = widgets.Checkbox(
            description="Enable", value=True, layout=self.checkbox_layout
        )
        self.hbox_noise = widgets.HBox(
            [
                self.label_noise,
                self.checkbox_noise,
                self.slider_noise,
                self.dropdown_noise,
            ],
            layout=self.hbox_layout,
        )
        # BLUR
        self.label_blur = widgets.Label(value="Blur", layout=self.label_layout)
        self.dropdown_blur = widgets.Dropdown(
            description="Filter",
            options=["Gaussian", "Box"],
            layout=self.dropdown_layout,
        )
        self.slider_blur = widgets.IntSlider(
            description="Kernel size",
            value=5,
            min=3,
            max=13,
            step=2,
            layout=self.slider_layout,
        )
        self.checkbox_blur = widgets.Checkbox(
            description="Enable", value=True, layout=self.checkbox_layout
        )
        self.hbox_blur = widgets.HBox(
            [self.label_blur, self.checkbox_blur, self.slider_blur, self.dropdown_blur],
            layout=self.hbox_layout,
        )
        # LIGHT CONDITIONS
        self.label_light = widgets.Label(
            value="Light conditions", layout=self.label_layout
        )
        self.placeholder_dropdown = widgets.Label(
            value="", layout=self.dropdown_layout
        )  # Placeholder for alignment
        self.slider_light = widgets.FloatSlider(
            description="Brightness",
            value=0,
            min=-1,
            max=1,
            step=0.05,
            layout=self.slider_layout,
        )
        self.checkbox_light = widgets.Checkbox(
            description="Enable", value=True, layout=self.checkbox_layout
        )
        self.hbox_light = widgets.HBox(
            [
                self.label_light,
                self.checkbox_light,
                self.slider_light,
                self.placeholder_dropdown,
            ],
            layout=self.hbox_layout,
        )
        # MISC
        self.button = widgets.Button(
            description="Sample new images", layout=widgets.Layout(width="200px")
        )
        self.button_box = widgets.HBox(
            [self.button],
            layout=widgets.Layout(justify_content="center", margin="15px 0px 5px 0px"),
        )
        self.select_images = widgets.Dropdown(
            description="№ of images",
            options=[3, 6, 9],
            value=9,
            layout=widgets.Layout(width="130px"),
        )
        self.images_box = widgets.HBox(
            [self.select_images], layout=widgets.Layout(justify_content="center")
        )
        self.button.on_click(self.on_button_clicked)

        self.configurable_ui = widgets.VBox(
            [
                self.hbox_noise,
                self.hbox_blur,
                self.hbox_light,
                self.button_box,
                self.images_box,
            ]
        )

        self.out_left = widgets.interactive_output(
            self.plot_left,
            {
                "noise_enabled": self.checkbox_noise,
                "noise_coef": self.slider_noise,
                "distribution": self.dropdown_noise,
                "blur_enabled": self.checkbox_blur,
                "kernel_size": self.slider_blur,
                "blur_type": self.dropdown_blur,
                "light_enabled": self.checkbox_light,
                "brightness": self.slider_light,
                "n_images": self.select_images,
            },
        )
        # Try to use class fields here
        self.out_right = widgets.interactive_output(
            self.plot_right,
            {
                "noise_enabled": self.checkbox_noise,
                "noise_coef": self.slider_noise,
                "distribution": self.dropdown_noise,
                "blur_enabled": self.checkbox_blur,
                "kernel_size": self.slider_blur,
                "blur_type": self.dropdown_blur,
                "light_enabled": self.checkbox_light,
                "brightness": self.slider_light,
                "n_images": self.select_images,
            },
        )
        self.checkbox_noise.observe(self.update_widget_states, names="value")
        self.checkbox_blur.observe(self.update_widget_states, names="value")
        self.checkbox_light.observe(self.update_widget_states, names="value")
        self.vertical_line = widgets.Box(
            layout=widgets.Layout(
                width="3px", margin="30px 0px", border="solid 2px black", height="645px"
            )
        )
        self.plots = widgets.HBox([self.out_left, self.vertical_line, self.out_right])

    def update_widget_states(self, change):
        if change["owner"] == self.checkbox_noise:
            self.slider_noise.disabled = not self.checkbox_noise.value
            self.dropdown_noise.disabled = not self.checkbox_noise.value
        elif change["owner"] == self.checkbox_blur:
            self.slider_blur.disabled = not self.checkbox_blur.value
            self.dropdown_blur.disabled = not self.checkbox_blur.value
        elif change["owner"] == self.checkbox_light:
            self.slider_light.disabled = not self.checkbox_light.value

    def on_button_clicked(self, _):
        with self.out_left:
            self.out_left.clear_output(wait=True)
            self.config.sampled_images = sample_images(self.config.image_dir)
            self.plot_left(n_images=self.select_images.value)

        with self.out_right:
            self.out_right.clear_output(wait=True)
            self.plot_right(
                noise_coef=self.slider_noise.value,
                distribution=self.dropdown_noise.value,
                kernel_size=self.slider_blur.value,
                blur_type=self.dropdown_blur.value,
                brightness=self.slider_light.value,
                noise_enabled=self.checkbox_noise.value,
                blur_enabled=self.checkbox_blur.value,
                light_enabled=self.checkbox_light.value,
                n_images=self.select_images.value,
            )

    def plot_left(self, **kwargs):
        max_rows = kwargs["n_images"] // 3
        visualize(
            self.config.sampled_images,
            n_col=3,
            max_rows=max_rows,
            height=3,
            width=3,
            title="Ground truth",
        )

    def plot_right(
        self,
        noise_coef,
        distribution,
        kernel_size,
        blur_type,
        brightness,
        noise_enabled,
        blur_enabled,
        light_enabled,
        n_images,
    ):
        degraded_images = []
        params = dict(
            noise_coef=noise_coef,
            distribution=distribution,
            kernel_size=kernel_size,
            blur_type=blur_type,
            brightness=brightness,
            noise_enabled=noise_enabled,
            blur_enabled=blur_enabled,
            light_enabled=light_enabled,
        )
        for image in self.config.sampled_images:
            degraded_image = randomly_degrade(image, **params)
            degraded_images.append(degraded_image)

        for param in params:
            self.config.__dict__[param] = params[param]

        degraded_images = torch.stack(degraded_images)
        max_rows = n_images // 3
        visualize(
            degraded_images,
            n_col=3,
            height=3,
            max_rows=max_rows,
            width=3,
            title="Degraded",
        )

    def display(self):
        display(self.configurable_ui)
        display(self.plots)
