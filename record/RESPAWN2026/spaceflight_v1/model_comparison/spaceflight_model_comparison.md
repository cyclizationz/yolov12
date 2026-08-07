# Spaceflight model comparison

The new model is directly comparable on both classes. FC5 and FM6 are one-class, out-of-domain controls: only their output index 0 can be scored against cockpit, and they cannot predict spaceship.

| Model | Box mAP50-95 | Mask mAP50-95 | Inference ms/image |
| --- | ---: | ---: | ---: |
| spaceflight_two_class | 0.8515 | 0.7880 | 1.541 |
| fc5_one_class | 0.0000 | 0.0000 | 0.980 |
| fm6_one_class | 0.0526 | 0.0007 | 0.855 |
