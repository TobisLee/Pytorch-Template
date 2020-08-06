# Pytorch template

Personal pytorch template

## Automatic Mixed Precision (Apex+pytorch)

#### Requirements

- pytorch
- apex

```
1. import amp module
from apex import amp

2. use amp to initialize model and optimizer 
net, optimizer = amp.initialize(net, optimizer, opt_level='O1')

3. use scaled_loss.backward() instead of normal loss.backward()
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```

TODO:
- [x] train template
- [ ] evaluate template
- [ ] use .json as config file
- [ ] tensorboard
- [ ] Distributed Data Parallel
- [x] Automatic Mixed Precision (Apex+pytorch)
- [ ] Automatic Mixed Precision (pytorch1.6 native AMP)
