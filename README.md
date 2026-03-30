# Playground for Hydra

## Prepare environment
Run this in your command line:

```bash
python3 -m venv .venv 
source .venv/bin/activate 
pip install hydra-core torch numpy
```

## Open and see the training file 
Open the `train.py` training dummy file. 
Run it with:
```bash
python3 train.py
```

By default, script is configured with pre-defined configurations in `configs/default.yaml`. Inspect them.
1. Hydra reads-in these configurations.
2. We print these configurations. 
3. Based on these configurations, we configure a dummy Neural Network. 
4. We let hydra to instantiate torch optimizer class. 
5. We perform a dummy training loops.

## Play-around
Now play around with these configurations:
1. Change any value in `configs/default.yaml`
2. Change the optimiser in `configs/default.yaml` to `torch.optim.AdamW`
3. Change the number of epochs through the command line.
4. Launch a multi run with different learning rates.
5. Be creative and experiment!