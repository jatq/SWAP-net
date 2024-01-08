The realization of "SWAP-net: Neural Network for Seismic Wave Azimuth Prediction from Single Three-component Observations"

## dependencies
`pytorch`
`numpy`
`pandas`
`h5py`
`proplot`

## dataset
- download at: https://github.com/smousavi05/STEAD
- filter the data through `data/generate_data.ipynb`，which generate `data/meta.csv`和`data/data.hdf5`
    - the csv file save the meta information of the data
    - the hdf5 file save the 3C waveforms
- read the data through `get_data_label` in `utils` module

## model
- the model is defnined in `models/swapnet.py`
- use `train.ipynb` to train the model and save the pretained parameters in `swap.pt`
- use `predict.ipynb` the make azimuth estimations with the pretrained model.
