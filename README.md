# LSTM Visualization

This is a repository for the article: https://medium.com/asap-report/visualizing-lstm-networks-part-i-f1d3fa6aace7
We strongly advise to read the article before running the code. The notebook was written in Python 3 and tested on Ubuntu 16.04 (without GPU).

You have to download data from https://archive.ics.uci.edu/ml/machine-learning-databases/auslan2-mld/tctodd.tar.gz and extract it  to `data` directory.

Next, you have to install requirements from `requirements.txt` and graphviz (`sudo apt install graphviz` on Ubuntu). Then you can launch the jupyter notebook.

There is some commented code and saved models, so if you want to experiment and fit models by yourself feel free to uncomment and experiment with code and parameters.

If you want to make a movie, you have to generate frames and then execute ffmpeg:

`ffmpeg -framerate 10 -i lstm2d-%03d.png -f mov output.mov`
