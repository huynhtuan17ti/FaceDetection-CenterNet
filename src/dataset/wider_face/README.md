# Wider faces dataset  
Link to dataset [wider face](http://shuoyang1213.me/WIDERFACE/)  
## Convert to json file
Using **convert.py** to convert `.txt` file to `.json` file  
  
**Format of json file**  
<pre>
[
    {
        "path": folder path of image 1
        "bbox": [[x1, y1, x2, y2], [x1, y1, x2, y2], ..., [x1, y1, x2, y2]] (bounding boxes, following pascal format)
        "id": 1 (numerical order)
    },
    {
        "path": folder path of image 2
        "bbox": [[x1, y1, x2, y2], [x1, y1, x2, y2], ..., [x1, y1, x2, y2]]
        "id": 2 
    },
    {
        ...
    }
    ...
]
</pre>

## Reference  
https://github.com/zisianw/WIDER-to-VOC-annotations
