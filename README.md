# MauiTracker
Maui63 CameraTracking 

### Installation

Download MauiTracker from github and create a virtual environment

half pi tested on python 3.8

``` sh
mkdir repos
cd repos
git clone https://github.com/johnnewto/MauiTracker.git
cd MauiTracker
python -m venv 'venv'
source ./venv/bin/activate
pip install --upgrade pip
pip install -e .
```

 
### Usage

``` sh
python main.py
```


With one of the windows in focus press spacebar to step, g to go continuously, and q to quit


![](images/mainview.png)