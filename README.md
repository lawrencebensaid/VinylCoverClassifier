# Vinyl Cover Classifier

Highlight all albums in the wall

## Install

Prerequisite: Python 3.7+

**Install dependencies**

```bash
pip install numpy opencv-python opencv-contrib-python
```

**Run the program**
```bash
python run.py --camera --data data/original/Beatles.pgm
```

*Find 1 album in an image in real time*

## Demo

*Real-time detection and tracking of David Bowie's Hunky Dory album cover (green rectangle).*

<img width="100%" alt="Demo GIF" src="https://user-images.githubusercontent.com/43364935/162777882-38b2d0c5-8f70-442a-a203-e29104f3ae7b.gif">

## Examples

*User real-time classification & tracking. Looking for the Beatles album*

```bash
$ python run.py --data data/original/Beatles.pgm --verbose --camera
```

*Find all albums in an image with adjustible parameters*

```bash
$ python run.py --data data/original --verbose --debug
```

<img width="25%" alt="Result screenshot" src="https://user-images.githubusercontent.com/43364935/163409398-e275adb1-223d-4ff2-b045-45bf1c786ca7.png">

## CLI reference

|Name|Type|Description|
|---|---|---|
|`--camera`|**Boolean**|*Use camera instead of input images to do real-time classification*|
|`--debug`|**Boolean**|*Show debug window*|
|`--verbose`|**Boolean**|*Verbose output*|
|`--size`|**Integer**|*Size to resize images to while in pipeline. (Lower = Faster, Higher = More accurate)*|
|`--confidence`|**Integer**|*Confidence threshold ranging from 0 to 100*|
|`--data`|**String**|*Path to data directory/file used to match potential vinyl covers (Can be a folder or a single image)*|
|`--input`|**String**|*Path to data directory/file used to match potential vinyl covers (Can be a folder or a single image)*|
|`--output`|**String**|*Path to output directory*|

## Known issues

### Windows UI bug

*When on Windows, your app may crash when not applying the `--debug` flag. This depends on your system and environment.*

**How to solve?**

*Apply the `--debug` flag to your run command.*

