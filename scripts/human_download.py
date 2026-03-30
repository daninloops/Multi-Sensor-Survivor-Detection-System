from bing_image_downloader import downloader
queries=["person trapped in rubble",
             "earthquake survicor debris",
             "person under collapsed building",
             "injured person lying on ground","hand sticking out of debris",
             "rescure team finding survivor",
             "person covered in dust",
             "human silhouette low",
             "person partially visible rubble",
             "burried person rescue"]
for query in queries:
    downloader.download(
             query,
             limit=50,
             adult_filter_off=False,
             output_dir="C:\\Users\\gupta\\OneDrive\\Desktop\\embedded project\\dataset\\train\\human_present",
             timeout=60,
             force_replace=False
)

 
