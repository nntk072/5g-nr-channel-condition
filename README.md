# long_nguyen_thesis

<!-- ## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)! -->

<!-- ## Add your filesfind . | sed 's/[^-][^\/]*\//  |/g;s/|\([^|]*\)$/|-- \1/' environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html) -->

---

## Name

Machine Learning based channel classification

## Description

Initial study are the physical random-access channel (PRACH) and the physical uplink shared channel (PUSCH), both of which are readily accessible in any 5G processing chain

The goal of this thesis topic is to research methods for classifying channel conditions, such as line-of-sight (LOS) and non-line-of-sight (NLOS), as well as different cell environments (e.g., urban microcell, rural macrocell)

An initial reference for us to use is Jayawardana, P. A. D. N., Obaid, H., Yesilyurt, T., Tan, B., & Lohan, E. S. (2023). Machine-Learning-Based LOS Detection for 5G Signals with Applications in Airport Environments. Sensors, 23(3), 1470. https://doi.org/10.3390/s23031470

## Directory structure

```
long_nguyen_thesis
  |-- NguyenLong.pdf
  |-- tdl_cdl.ipynb
  |-- cnn_siso.ipynb
  |-- ml_methods.ipynb
  |-- sionna_mimo.ipynb
  |-- visualization.py
  |-- mlp_los_nlos.ipynb
  |-- sionna_siso.ipynb
  |-- README.md
  |-- nr_pusch
  |  |-- configs_new.py
  |  |-- model.py
  |  |-- feature_check_siso.ipynb
  |  |-- configs.py
  |  |-- model_new.py
  |  |-- feature_check_mimo.ipynb
  |-- .gitignore
  |-- los_nlos_final.ipynb
  |-- main.ipynb
  |-- cvnn_model.ipynb
  |-- sionna_mimo copy.ipynb
  |-- evaluation.py
  |-- cnn
  |  |-- model.py
  |  |-- layers.py
  |  |-- module.py
  |-- data_preprocessing.py
  |-- mlp_environment.ipynb
  |-- utils.py
  |-- mlp
  |  |-- __init__.py
  |  |-- mlp_model.py
  |-- ml_algorithms.py
  |-- cnn_mimo.ipynb
  |-- feature_check_tdl_cdl.ipynb
  |-- environment_final.ipynb
```

<!-- ## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge. -->

<!-- ## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method. -->

## Installation

```bash


conda create -n thesis python=3.11

conda activate thesis

# python3 -m pip install tensorflow[and-cuda]==
pip install --upgrade ipykernel jupyterlab
pip install h5py
pip install pandas

conda install pytables
pip install scikit-learn
pip install seaborn
pip install plotly
pip install openpyxl
pip install sionna

# winget install LLVM # Need to ask the IT support to install this since it requires admin rights / necessary if you run in Windows only, Halti didn't need that
conda create -n cnn python=3.11
conda activate cnn
conda install tensorflow-gpu
conda install scikit-image tqdm scikit-learn pydot
conda install scikit-learn-intelex
pip install tensorflow-addons
pip install matplotlib
pip install --upgrade ipykernel jupyterlab
pip install onnx-tf
pip install -U tf2onnx
pip install tensorflow_probability
pip install --upgrade tensorflow
pip install --upgrade tensorflow-gpu
pip install --upgrade tensorflow-probability
pip install tensorflow[and-cuda]


conda create -n cvnn
conda activate cvnn
pip install --upgrade ipykernel jupyterlab
pip install cvnn
pip install scikit-learn
pip install matplotlib
pip install tensorflow[and-cuda]
pip install tensorflow-probability[tf]
pip install onnxruntime
pip install onnxmltools
pip install tf2onnx

conda create -n mlp python=3.10
pip install tensorflow[and-cuda]
pip install scikit-learn
pip install matplotlib
pip install joblib
pip install tf2onnx onnxruntime
```

Reference for LLVM: [Error: LLVM shared library (&#34;LLVM-C.dll&#34;) could not be found](https://github.com/mitsuba-renderer/drjit/issues/196) and [Deadlock when load default scene](https://github.com/NVlabs/sionna/discussions/275)
In case you cannot install LLVM, you can use Google Colab instead, with T4 GPU.

<!-- 
## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.
 -->

## Authors and acknowledgment

Long Nguyen

<!-- ## License
For open source projects, say how it is licensed. -->

## Project status

<!-- If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers. -->


## Plan:

17.01.2025-09.02.2025: Adding SISO and MIMO 5G PUSCH Models + Remove Ray Tracing duplication in the code + Setting up the environment according to the advisor's requirement

27.01.2025-09.02.2025:	Model Feature Extraction + Model Evaluation Metrics Utils

10.02.2025-23.02.2025:	MLP model implementation

24.02.2025-16.03.2025:	MLP model improvement + CNN model implementation

17.03.2025-13.04.2025:	CNN improvement + CCNN implementation

14.04.2025-04.05.2025:	CCNN Improvement
