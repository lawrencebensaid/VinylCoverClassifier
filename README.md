# Vinyl Cover Classifier

Highlight all albums in the wall

## Run

*Find 1 album in an image in real time*

```bash
$ python run.py --camera --data data/original/Beatles.pgm --max-matches 1
```

### Debug

*Find all albums in an image with adjustible parameters*

```bash
$ python run.py --data data/original --verbose --debug
```

## Flag options

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


## Demo

*Real-time detection and tracking of David Bowie's Hunky Dory album cover (green rectangle).*

![demo](https://user-images.githubusercontent.com/43364935/162777882-38b2d0c5-8f70-442a-a203-e29104f3ae7b.gif)
