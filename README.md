# GAL vs SINGLE
This code compares previous defense and attack on GNN (see the sources under "Credits" below)

## Requirements
This project is based on PyTorch 1.6.0 and the PyTorch Geometric library.

First, install PyTorch from the official website: https://pytorch.org/.
Then install PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
(PyTorch Geometric must be installed according to the instructions there).
Eventually, run the following to verify that all dependencies are satisfied:

```setup
conda install --file my_env.txt
```

## Full Evaluation of Different Modes - Example
to run all modes simply run evaluate.py with the following arguments. 
Keep in mind that the -SINGLE flag should be changed to 0 if you do not want to use SINGLE attack
```buildoutcfg
./evaluate.py -SINGLE 1 -num_epochs 50 -finetune_epochs 10
```

## Running a specific mode - Example
```buildoutcfg
./main.py -SINGLE 1 -num_epochs 50 -finetune_epochs 10 -dataset cora -GAL_gnn_type ChebConv```
```

## Credits
For the original GAL paper, see: https://arxiv.org/abs/2009.13504
For the original SINGLE paper, see: https://arxiv.org/abs/2011.03574
