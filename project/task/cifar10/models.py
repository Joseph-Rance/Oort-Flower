"""ResNet-50 model for CIFAR10 dataset."""

from torchvision.models import resnet50

from project.utils.utils import lazy_config_wrapper

get_resnet_50 = lazy_config_wrapper(lambda: resnet50(num_classes=10))