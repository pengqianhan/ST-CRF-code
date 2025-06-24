# ST-CRF: Enhancing Pedestrian Trajectory Prediction through Step-Intention Learning with Conditional Random Fields

This repository contains the implementation of the ST-CRF (Spatio-Temporal Conditional Random Field) model for trajectory prediction.

## Getting Started

### Prerequisites

- Python 3.11
- PyTorch

### Installation

1. Clone this repository:
   ```
   cd ST-CRF
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Model Checkpoints

Trained model checkpoints are available in the `./checkpoint/` directory. 

### Evaluating the Model

For standard evaluation, use:

```
python test_ade_fde.py
```

### Zero-shot Prediction Evaluation

For zero-shot prediction evaluation, use:

```
python test_ade_fde_0shot.py
```

#### Example: Evaluating on ETH-Hotel Dataset

If you want to evaluate the model trained on ETH dataset for zero-shot prediction on the Hotel dataset aka 'E2H':

1. Open `test_ade_fde_0shot.py`
2. Set the following variables:
   ```python
   paths = ['./checkpoint/stcrf_eth']
   test_data = 'hotel'
   ```
3. Run the script:
   ```
   python test_ade_fde_0shot.py
   ```


## Training the Model

If you want to train the ST-CRF model, run:

```
sh train.sh
```



## License

This project is licensed under the [MIT License](LICENSE).

