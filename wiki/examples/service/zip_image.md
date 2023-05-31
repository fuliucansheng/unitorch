# Zip Image Service Tutorial

&nbsp;&nbsp;&nbsp;&nbsp;In this tutorial, we will explore the usage of an image service for training and inference with image-related models. This service is specifically designed to handle large-scale image datasets, containing over 300 million images. The image service operates in the backend and offers an HTTP API that allows Users & GPU processes to access image bytes by using an image ID.

<hr/>

![Zip Image Service](zip_image.jpg)

<hr/>

### Configuration files

#### [Single zip folder](https://github.com/fuliucansheng/unitorch/blob/master/examples/configs/services/zip_image/config.ini)
```ini
[core/cli]
zip_folder = zip_folder/
service_name = core/service/zip_image

[core/service/zip_image]
zip_folder = ${core/cli:zip_folder}
zip_extension = .zip
```

#### [Multiple zip folders](https://github.com/fuliucansheng/unitorch/blob/master/examples/configs/services/zip_image/config_v2.ini)
```ini
[core/cli]
zip_folder1 = zip_folder1/
zip_folder2 = zip_folder2/
zip_folder3 = zip_folder3/
service_name = core/service/zip_image

[core/service/zip_image]
zip_folder = [
    "${core/cli:zip_folder1}",
    "${core/cli:zip_folder2}",
    "${core/cli:zip_folder3}"
  ]
zip_extension = .zip
```

### Start Service

```bash
unitorch-service start path/to/zip/image/service.ini \
    --zip_folder path/to/zip/folder
```

### Stop Service

```bash
unitorch-service stop path/to/zip/image/service.ini \
    --zip_folder path/to/zip/folder
```

### Restart Service

```bash
unitorch-service restart path/to/zip/image/service.ini \
    --zip_folder path/to/zip/folder
```