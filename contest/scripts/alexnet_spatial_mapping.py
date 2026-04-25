"""
AlexNet Spatial-Semantic Bridge
=========================
Back-project "dead" neuron in FC6 to Conv5 spatial patch via max-pooling masks.

FC6: 4096-dimensional fully connected layer
Conv5: 13x13 x 256 feature maps
MaxPool: 6x6 x 256
"""

import json
import numpy as np


class AlexNetSpatialMapper:
    def __init__(self):
        self.conv5_h, self.conv5_w, self.conv5_c = 13, 13, 256
        self.pool_kernel = 2
        self.pool_h = self.conv5_h // 2
        self.pool_w = self.conv5_w // 2
        self.fc6_size = 4096
    
    def fc6_to_conv5(self, fc6_idx):
        """Back-project FC6 neuron to Conv5 receptive field."""
        pc = fc6_idx % self.conv5_c
        pw = (fc6_idx // self.conv5_c) % self.pool_w
        ph = fc6_idx // (self.conv5_c * self.pool_w)
        
        h_start = ph * self.pool_kernel
        h_end = h_start + self.pool_kernel
        w_start = pw * self.pool_kernel
        w_end = w_start + self.pool_kernel
        
        return {
            "fc6_index": int(fc6_idx),
            "maxpool_coord": [int(ph), int(pw), int(pc)],
            "conv5_receptive_field": {
                "h_range": [int(h_start), int(h_end)],
                "w_range": [int(w_start), int(w_end)],
                "channel": int(pc)
            },
            "spatial_patch": {
                "height": int(h_end - h_start),
                "width": int(w_end - w_start),
                "attribution_weights": [[1.0, 1.0], [1.0, 1.0]]
            }
        }


if __name__ == "__main__":
    mapper = AlexNetSpatialMapper()
    
    test_neurons = [0, 100, 500, 1000, 2000, 3000, 4095]
    
    print("=== AlexNet Spatial-Semantic Bridge ===")
    print(f"Conv5 shape: {mapper.conv5_h}x{mapper.conv5_w}x{mapper.conv5_c}")
    print(f"MaxPool shape: {mapper.pool_h}x{mapper.pool_w}x{mapper.conv5_c}")
    print(f"FC6 size: {mapper.fc6_size}")
    print(f"Flattened size: {mapper.pool_h * mapper.pool_w * mapper.conv5_c}")
    
    print("\n=== Layer Connections ===")
    print("Conv5 → MaxPool: 2x2 max pooling with stride 2")
    print("MaxPool → FC6: flatten (9216) → fully connected (4096)")
    
    print("\n=== Back-Projection Results ===")
    results = []
    for idx in test_neurons:
        result = mapper.fc6_to_conv5(idx)
        results.append(result)
        r = result
        print(f"FC6[{r['fc6_index']}] → MaxPool{r['maxpool_coord']} → Conv5[h={r['conv5_receptive_field']['h_range']}, w={r['conv5_receptive_field']['w_range']}, ch={r['conv5_receptive_field']['channel']}]")
    
    report = {
        "architecture": {
            "conv5_shape": [13, 13, 256],
            "pool4_shape": [6, 6, 256],
            "fc6_size": 4096
        },
        "layer_connections": {
            "conv5_to_maxpool": "2x2 max pooling with stride 2",
            "maxpool_to_fc6": "flatten then fully connected (9216 → 4096)"
        },
        "neuron_backprojection": results
    }
    
    with open("/home/phoenix/robin/alexnet_result.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n✓ Results saved to alexnet_result.json")