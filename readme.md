# DocumentDenoising

### Deployment for Windows 10/11

##### 1. Install miniconda
    https://docs.conda.io/en/latest/miniconda.html
    
##### 2. Run commands to install tensorflow for Windows:
    
    https://www.tensorflow.org/install/pip?hl=ru#windows-native

    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    # Anything above 2.10 is not supported on the GPU on Windows Native
    python -m pip install "tensorflow<2.11"
    # Verify install:
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

##### 3. Install tesseract
    
    https://github.com/UB-Mannheim/tesseract/wiki

##### 4. Install python requirements
    
     pip install -r requirements.txt

##### 5. Run ml service from imgServiceProject/main.py

##### 6. Run django app from denoiseim

    

