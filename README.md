# COPPER

COPPER: a stacking ensemble deep-learning networks for computational recognition exclusive virus-derived small interfering RNAs in plants .

The datasets can be found in `./data/`.  The COPPER models is available in `./model/`.

Before running COPPER_Predict users shuold make sure all the following packages are installed in their Python enviroment:

    numpy == 1.19.5
    pandas == 0.22.0
    sklearn == 0.20.0
    gensim == 4.1.0
    Bio == 1.79
    keras==2.2.4
    keras_self_attention == 0.46.0
    h5py == 2.9.0
    tensorflow == 1.14
    python == 3.6
For advanced users who want to perform prediction by using their own data:

To get the information the user needs to enter for help, run:

    python COPPER_predict.py --help
or

    python COPPER_predict.py -h
Using TensorFlow backend.

usage: COPPER_predict.py [-h] --input inputpath [--output OUTPUTFILE]

# COPPER: a stacking ensemble deep-learning networks for computational recognition exclusive virus-derived small interfering RNAs in plants 
optional arguments:

-h, --help show this help message and exit

--input inputpath query PVsiRNA sequences to be predicted in fasta format.

--output OUTPUTFILE save the prediction results.
