"""ResNet-34 model for speech dataset.

imported from https://github.com/SymbioticLab/Oort/blob/master/training/utils/resnet_speech.py
largely identical to the pytorch implementation, but with only a single input channel
"""

from torchvision.models import resnet50

get_resnet_50 = lazy_config_wrapper(lambda: resnet50(num_classes=10))