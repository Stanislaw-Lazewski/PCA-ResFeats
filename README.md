# PCA-ResFeats

We propose methods to improve the efficiency of image analysis by using local image descriptors in the form of appropriately formed semantic features extracted from the ResNet-50 deep neural network – PCA-ResFeats.
## Instalation
First, download the repository:
```sh
$ git clone git@github.com:Stanislaw-Lazewski/PCA-ResFeats.git
$ cd PCA-ResFeats
```
Second, create conda envirement:
```sh
$ conda env create -n pca-resfeats --file environment.yml
$ conda activate pca-resfeats
```

## Demo

Download Shells and Pebbles dataset available at: https://www.kaggle.com/datasets/vencerlanz09/shells-or-pebbles-an-image-classification-dataset, unzip and then place it in the `data_demo/Shells_and_Pebbles` directory.

Run `src/demo.py`:
```sh
python3 src/demo.py
```
The files `.npy` in the `pca-resfeats` folder contain the extracted PCA-ResFeats. File `pca-resfeats/classification_report.txt` contains the classification results.

## Experiments
### Data preparation
Prepare the datasets from which you want to extract PCA-ResFeats and place them in the `data` directory. Each class should be placed in its own subdirectory. The datasets I have used are:
```
data
├── Caltech101
├── Caltech256
├── Catordog
├── EuroSAT
├── Flowers
└── MLC2008
```

### Running experiments
Run `src/main.py` to repeat my experiments or adapt the function calls to your own datasets:
```sh
python3 src/main.py
```

### Experiments results
The results of the experiments are available here: https://aghedupl-my.sharepoint.com/:f:/g/personal/slazewsk_agh_edu_pl/EpE8LQmXBIpDpVU44eYjuMYBMBCdbs3gO0EtEz9LIfo02w?e=TaumAv

## License
MIT
