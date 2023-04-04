# Installation
## Prerequisites
- Python 3.10 or higher (download from [here](https://www.python.org/downloads/))
- Git (download from [here](https://git-scm.com/downloads))
## Getting code
Clone code wherever you want with command `git clone https://github.com/temchik00/Net-Layer.git`
Go to folder with cloned code
## Creating virtual environment (optional)
Create virtual environment with command `python -m venv %venv_name%`, where `%venv_name%` is a name for virtual environment
Type `%venv_name%/Scripts/activate` to activate virtual environment
## Updating pip (optional)
Type `python -m pip install -U pip`
## Installing dependencies
Type `pip install -r requirements.txt`
# Running
In order to run script type `python main.py %input_path% %output_path%`, where `%input_path%` is a path to source image and `%output_path%` is a path where to save resulting image. Both incluiding file extension.
To process multiple images at once use folders for `%input_path%` and `%output_path%`. Script will use original filename when saving to output.
For additional parameters get help by typing `python main.py -h`
