from bing_image_downloader import downloader

classes=["collapsed building debris",
"earthquake rubble no people",
"broken concrete pile",
"destroyed buildings empty",
"construction debris pile",
"war zone ruins empty",
"dusty abandoned building",
"rocks pile similar to human shape",
"shadows in rubble",
"rescue site no survivors"]

for cls in classes:
    downloader.download(
        cls,
        limit=50,
        output_dir="C:\\Users\\gupta\\OneDrive\\Desktop\\embedded project\\dataset\\train\\no_human",
        force_replace=False,
        adult_filter_off=False,
        timeout=60
    )
 

 