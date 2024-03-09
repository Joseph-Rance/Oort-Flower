"""ResNet-34 model for speech dataset."""

from torchvision.models import resnet34

from project.utils.utils import lazy_config_wrapper

get_resnet_34 = lazy_config_wrapper(lambda: resnet34(num_classes=35))