# src/baseline_models.py

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Optional

class BaselineModelWrapper(nn.Module):
    """
    Wrapper for TorchVision models to adapt them for our datasets.
    Handles input channel adaptation and output class number adjustment.
    """
    
    def __init__(self, model_name: str, num_classes: int, input_channels: int = 1, pretrained: bool = False):
        super(BaselineModelWrapper, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Get the base model
        self.base_model = self._create_base_model(model_name, pretrained)
        
        # Adapt input layer for different channel numbers
        self._adapt_input_layer()
        
        # Adapt output layer for different class numbers
        self._adapt_output_layer()
        
    def _create_base_model(self, model_name: str, pretrained: bool):
        """Create the base TorchVision model."""
        model_factories = {
            'resnet18': lambda: models.resnet18(pretrained=pretrained),
            'resnet34': lambda: models.resnet34(pretrained=pretrained),
            'resnet50': lambda: models.resnet50(pretrained=pretrained),
            'mobilenet_v2': lambda: models.mobilenet_v2(pretrained=pretrained),
            'mobilenet_v3_small': lambda: models.mobilenet_v3_small(pretrained=pretrained),
            'mobilenet_v3_large': lambda: models.mobilenet_v3_large(pretrained=pretrained),
            'efficientnet_b0': lambda: models.efficientnet_b0(pretrained=pretrained),
            'efficientnet_b1': lambda: models.efficientnet_b1(pretrained=pretrained),
            'vgg16': lambda: models.vgg16(pretrained=pretrained),
            'vgg16_bn': lambda: models.vgg16_bn(pretrained=pretrained),
            'densenet121': lambda: models.densenet121(pretrained=pretrained),
            'squeezenet1_0': lambda: models.squeezenet1_0(pretrained=pretrained),
            'inception_v3': lambda: models.inception_v3(pretrained=pretrained, aux_logits=False),
            'googlenet': lambda: models.googlenet(pretrained=pretrained, aux_logits=False),
        }
        
        if model_name not in model_factories:
            raise ValueError(f"Unsupported model: {model_name}. Available models: {list(model_factories.keys())}")
            
        return model_factories[model_name]()
    
    def _adapt_input_layer(self):
        """Adapt the first convolutional layer for different input channels."""
        if self.input_channels == 3:
            return  # No adaptation needed for RGB
            
        # Find and replace the first conv layer
        if hasattr(self.base_model, 'conv1'):  # ResNet, DenseNet
            old_conv = self.base_model.conv1
            self.base_model.conv1 = nn.Conv2d(
                self.input_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                padding=old_conv.padding, bias=old_conv.bias is not None
            )
        elif hasattr(self.base_model, 'features') and len(self.base_model.features) > 0:  # VGG, MobileNet, EfficientNet
            # Find the first Conv2d layer in features
            for i, layer in enumerate(self.base_model.features):
                if isinstance(layer, nn.Conv2d):
                    old_conv = layer
                    new_conv = nn.Conv2d(
                        self.input_channels, old_conv.out_channels,
                        kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                        padding=old_conv.padding, bias=old_conv.bias is not None
                    )
                    # Copy weights if possible
                    if self.input_channels == 3:
                        new_conv.weight.data = old_conv.weight.data
                        if old_conv.bias is not None:
                            new_conv.bias.data = old_conv.bias.data
                    elif self.input_channels == 1:
                        # For grayscale, average the RGB weights
                        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
                        if old_conv.bias is not None:
                            new_conv.bias.data = old_conv.bias.data
                    
                    self.base_model.features[i] = new_conv
                    break
        elif hasattr(self.base_model, 'Conv2d_1a_3x3'):  # Inception
            old_conv = self.base_model.Conv2d_1a_3x3.conv
            self.base_model.Conv2d_1a_3x3.conv = nn.Conv2d(
                self.input_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                padding=old_conv.padding, bias=old_conv.bias is not None
            )
    
    def _adapt_output_layer(self):
        """Adapt the final layer for different number of classes."""
        # ResNet
        if hasattr(self.base_model, 'fc'):
            old_fc = self.base_model.fc
            self.base_model.fc = nn.Linear(old_fc.in_features, self.num_classes)
        
        # VGG
        elif hasattr(self.base_model, 'classifier') and isinstance(self.base_model.classifier, nn.Sequential):
            old_classifier = self.base_model.classifier[-1]
            if isinstance(old_classifier, nn.Linear):
                self.base_model.classifier[-1] = nn.Linear(old_classifier.in_features, self.num_classes)
        
        # MobileNet
        elif hasattr(self.base_model, 'classifier') and isinstance(self.base_model.classifier, nn.Linear):
            old_classifier = self.base_model.classifier
            self.base_model.classifier = nn.Linear(old_classifier.in_features, self.num_classes)
        
        # DenseNet
        elif hasattr(self.base_model, 'classifier'):
            old_classifier = self.base_model.classifier
            self.base_model.classifier = nn.Linear(old_classifier.in_features, self.num_classes)
        
        # EfficientNet
        elif hasattr(self.base_model, 'classifier') and hasattr(self.base_model.classifier, '1'):
            old_classifier = self.base_model.classifier[1]
            if isinstance(old_classifier, nn.Linear):
                self.base_model.classifier[1] = nn.Linear(old_classifier.in_features, self.num_classes)
        
        # SqueezeNet
        elif hasattr(self.base_model, 'classifier') and hasattr(self.base_model.classifier, '1'):
            old_conv = self.base_model.classifier[1]
            if isinstance(old_conv, nn.Conv2d):
                self.base_model.classifier[1] = nn.Conv2d(
                    old_conv.in_channels, self.num_classes,
                    kernel_size=old_conv.kernel_size, stride=old_conv.stride
                )
        
        # Inception, GoogLeNet
        elif hasattr(self.base_model, 'fc'):
            old_fc = self.base_model.fc
            self.base_model.fc = nn.Linear(old_fc.in_features, self.num_classes)
    
    def forward(self, x):
        return self.base_model(x)
    
    def get_parameter_count(self):
        """Get the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)




def create_baseline_model(model_name: str, num_classes: int, input_channels: int = 1, pretrained: bool = False):
    """
    Factory function to create baseline models from TorchVision.
    
    Args:
        model_name: Name of the TorchVision model to create
        num_classes: Number of output classes
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        pretrained: Whether to use pretrained weights
    
    Returns:
        PyTorch model instance
    """
    return BaselineModelWrapper(model_name, num_classes, input_channels, pretrained)


def get_available_models():
    """Get list of available baseline models from TorchVision."""
    return [
        'resnet18',
        'resnet34', 
        'resnet50',
        'mobilenet_v2',
        'mobilenet_v3_small',
        'mobilenet_v3_large',
        'efficientnet_b0',
        'efficientnet_b1',
        'vgg16',
        'vgg16_bn',
        'densenet121',
        'squeezenet1_0',
        'inception_v3',
        'googlenet'
    ]


def get_model_info(model_name: str, num_classes: int = 10, input_channels: int = 1):
    """Get information about a model without creating it."""
    model_info = {
        'resnet18': {
            'description': 'ResNet-18 with residual connections',
            'typical_params': '~11M',
            'memory_efficient': False
        },
        'resnet34': {
            'description': 'ResNet-34 with residual connections',
            'typical_params': '~21M',
            'memory_efficient': False
        },
        'resnet50': {
            'description': 'ResNet-50 with bottleneck blocks',
            'typical_params': '~25M',
            'memory_efficient': False
        },
        'mobilenet_v2': {
            'description': 'MobileNet-v2 with depthwise separable convolutions',
            'typical_params': '~3M',
            'memory_efficient': True
        },
        'mobilenet_v3_small': {
            'description': 'MobileNet-v3 Small with neural architecture search',
            'typical_params': '~2M',
            'memory_efficient': True
        },
        'mobilenet_v3_large': {
            'description': 'MobileNet-v3 Large with neural architecture search',
            'typical_params': '~5M',
            'memory_efficient': True
        },
        'efficientnet_b0': {
            'description': 'EfficientNet-B0 with compound scaling',
            'typical_params': '~5M',
            'memory_efficient': True
        },
        'efficientnet_b1': {
            'description': 'EfficientNet-B1 with compound scaling',
            'typical_params': '~7M',
            'memory_efficient': True
        },
        'vgg16': {
            'description': 'VGG-16 with deep convolutional layers',
            'typical_params': '~138M',
            'memory_efficient': False
        },
        'vgg16_bn': {
            'description': 'VGG-16 with batch normalization',
            'typical_params': '~138M',
            'memory_efficient': False
        },
        'densenet121': {
            'description': 'DenseNet-121 with dense connections',
            'typical_params': '~7M',
            'memory_efficient': True
        },
        'squeezenet1_0': {
            'description': 'SqueezeNet with fire modules',
            'typical_params': '~1M',
            'memory_efficient': True
        },
        'inception_v3': {
            'description': 'Inception-v3 with factorized convolutions',
            'typical_params': '~27M',
            'memory_efficient': False
        },
        'googlenet': {
            'description': 'GoogLeNet with inception modules',
            'typical_params': '~6M',
            'memory_efficient': True
        }
    }
    
    return model_info.get(model_name, {'description': 'Unknown model', 'typical_params': 'Unknown', 'memory_efficient': False})

