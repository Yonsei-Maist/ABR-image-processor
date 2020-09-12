# ABR-image-processor

image proccess using opencv-python and yolo with tensorflow
1. extract x, y values from graph image
2. extract peak from image (if image have peak)

## Environment
```
python 3.7 ~
opencv-python lastest
```

## Install
Need authorization
```
pip install -U git+https://git@github.com/Yonsei-Maist/ABR-image-processor.git
```

## Usage
```
extractor = Extractor()  # initialize

extractor.extract(file_path, is_left_ear)  # read graph from image path

extractor.extract_image(image_object_from_opencv, is_left_ear)  # read graph from image object read by opencv

extractor.extract_with_peak(file_path, is_left_ear)  # read graph and peak from image path

extractor.extract_image_with_peak(image_object_from_opencv, is_left_ear)  # read graph and peak from image object read by opencv
```

## Author
Chanwoo Gwon, Yonsei Univ. Researcher since 2020.05~

## Maintainer
Chanwoo Gwon, arknell@yonsei.ac.kr (2020.09 ~)
