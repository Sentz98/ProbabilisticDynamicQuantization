from backend import QuantMode

# Dynamic Default Parameters

layer_configs = {
    '23': {'skip': True},
}


# # Configuration for specific layers
# layer_configs = {
#     # Skip converting the first linear layer
#     'sub.fc1': {'skip': True},
    
#     # Custom parameters for second convolutional layer
#     'features': {
#         'e_std': 3,
#         'sampling_stride': 4,
#     },

#     'features.1': {
#         'e_std': 1,
#         'sampling_stride': 1,
#     },
    
#     # Special configuration for final classifier
#     'classifier': {
#         'mode' : QuantMode.STATIC
#     }
# }
