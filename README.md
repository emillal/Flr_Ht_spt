# Flr_Ht_spt
Thermal Hotspot Detection using ML


## Step 1 : Creating Pyenv

```
sudo apt-get update
sudo apt-get install libncurses5-dev
sudo apt-get install python-dev
sudo apt-get install python-pip
sudo apt-get install libjpeg8-dev
sudo ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib
pip install pillow
sudo apt-get build-dep python-imaging
sudo apt-get install libjpeg8 libjpeg62-dev libfreetype6 libfreetype6-dev
sudo pip install virtualenv  
virtualenv -p python3 .env                  # Create a virtual environment
source .env/bin/activate         # Activate the virtual environment
```

```pip install -r``` the following dependencies , chance to new version when required

```
Cython==0.23.4
Jinja2==2.8
MarkupSafe==0.23
Pillow==3.0.0
Pygments==2.0.2
appnope==0.1.0
argparse==1.2.1
backports-abc==0.4
backports.ssl-match-hostname==3.5.0.1
certifi==2015.11.20.1
cycler==0.10.0
decorator==4.0.6
future==0.16.0
gnureadline==6.3.3
h5py==2.7.0
ipykernel==4.2.2
ipython==4.0.1
ipython-genutils==0.1.0
ipywidgets==4.1.1
jsonschema==2.5.1
jupyter==1.0.0
jupyter-client==4.1.1
jupyter-console==4.0.3
jupyter-core==4.0.6
matplotlib==2.0.0
mistune==0.8.1
nbconvert==4.1.0
nbformat==4.0.1
notebook==5.7.8
numpy==1.10.4
path.py==8.1.2
pexpect==4.0.1
pickleshare==0.5
ptyprocess==0.5
pyparsing==2.0.7
python-dateutil==2.4.2
pytz==2015.7
pyzmq==15.1.0
qtconsole==4.1.1
scipy==0.16.1
simplegeneric==0.8.1
singledispatch==3.4.0.3
site==0.0.1
six==1.10.0
terminado==0.5
tornado==4.3
traitlets==4.0.0
```
![Screenshot from 2024-07-31 10-47-51](https://github.com/user-attachments/assets/823b8171-aa9e-4424-b8c9-7146be3bc854)
Once a pyenv is created you can proceed.


You can exceute the hotplanner in the hotspot tool using the following command:
```
../hotfloorplan -c hotspot.config -f floorplan_9999.desc -p power_9999.p -o output.flp
```


Use the following command to do the HotSpot tool for thermal analysis :
```
../hotspot -c hotspot.config -f output.flp -p power_9999.ptrace -model_type block -steady_file outputs/output.steady -o outputs/output.ttrace

```



We require the .flp (floorplan) and a .ptrace file aswell.<br />
![Screenshot from 2024-08-07 15-19-22](https://github.com/user-attachments/assets/1ccb4854-2fe4-41f9-88a1-db771b963636)
