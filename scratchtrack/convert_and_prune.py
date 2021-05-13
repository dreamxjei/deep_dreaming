import os
from shutil import copyfile
import argparse
from PIL import Image

# takes the Windows lockscreen folder's current assets and converts to jpg.
# then takes wallpapers (width > height) and copies (sorts) them into a separate folder.

# add command line arguments on what folder converted images are stored in
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', default='converted', help='Choose which img folder to save to')  # required=True
args = parser.parse_args()

path = os.getcwd()
if not os.path.exists(os.path.join(path, args.folder)):
    os.makedirs(os.path.join(path, args.folder))
if not os.path.exists(os.path.join(path, args.folder + '_pruned_landscape')):
    os.makedirs(os.path.join(path, args.folder + '_pruned_landscape'))
if not os.path.exists(os.path.join(path, args.folder + '_pruned_portrait')):
    os.makedirs(os.path.join(path, args.folder + '_pruned_portrait'))

# add the directory to the path
# path = "./" + args.folder + "/"

source_dir = 'C:/Users/mrJin/AppData/Local/Packages/Microsoft.Windows.ContentDeliveryManager_cw5n1h2txyewy/LocalState/Assets'
onlyfiles = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
num = len(onlyfiles)

for file in onlyfiles:
    copyfile(os.path.join(source_dir, file), os.path.join(path, args.folder, file + '.jpg'))
    try:
        im = Image.open(os.path.join(path, args.folder, file + '.jpg'))
        width, height = im.size
        if width > height > 50:
            copyfile(os.path.join(path, args.folder, file + '.jpg'), os.path.join(path, args.folder + '_pruned_landscape', file + '.jpg'))
        if height > width > 50:
            copyfile(os.path.join(path, args.folder, file + '.jpg'), os.path.join(path, args.folder + '_pruned_portrait', file + '.jpg'))
    except OSError:
        continue