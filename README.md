Instructions:

git clone https://github.com/mysterefrank/roger_repair_scanline_defects.git
cd roger_repair_scanline_defects


Install requirements file
Run ```pip install -r requirements.txt``` in terminal

Get a replicate token https://replicate.com/

Set up your replicate token:
Run ```export REPLICATE_API_TOKEN="your-token-here"``` in terminal

Put all the images you want to repair in frames_dir
Run ```python repair.py``` in terminal

Wait

Your repaired images will be in output_dir