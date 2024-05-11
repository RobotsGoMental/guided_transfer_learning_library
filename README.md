# Guided Transfer Learning 
This repository implements our Guided Transfer Learning (GTL) approach, for more details see [our paper](https://arxiv.org/pdf/2303.16154.pdf).
We propose a new approach called Guided Transfer Learning, which involves assigning guiding parameters to each weight and bias in the network, allowing for a reduction in resources needed to train a network. Guided Transfer Learning enables the network to learn from a small amount of data and potentially has many applications in resource-efficient machine learning.

## Steps to run example Jupyter Notebook
```bash
sudo apt install python3.9 python3.9-distutils python3.9-venv
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install notebook ipykernel
jupyter notebook
```


## Credits

This project was created by Danko Nikolić, Vjekoslav Nikolić and Davor Andrić.

© 2024, Robots Go Mental, UG or its Affiliates. 

## License

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]


[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg