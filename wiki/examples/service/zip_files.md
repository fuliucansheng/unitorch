# Zip Files Service Tutorial

&nbsp;&nbsp;&nbsp;&nbsp;In this tutorial, we will explore the usage of a file service for training and inference with file-related models. This service is specifically designed to handle large-scale files datasets, containing over 300 million filess. The files service operates in the backend and offers an HTTP API that allows Users & GPU processes to access file bytes by using a file ID.

<hr/>

![Zip Files Service](zip_files.jpg)

<hr/>

### Configuration files

#### [Single zip folder](https://github.com/fuliucansheng/unitorch/blob/master/examples/configs/services/zip_files/config.ini)
```ini
[core/cli]
zip_folder = zip_folder/
service_name = core/service/zip_files

[core/service/zip_files]
zip_folder = ${core/cli:zip_folder}
zip_extension = .zip
```

#### [Multiple zip folders](https://github.com/fuliucansheng/unitorch/blob/master/examples/configs/services/zip_files/config_v2.ini)
```ini
[core/cli]
zip_folder1 = zip_folder1/
zip_folder2 = zip_folder2/
zip_folder3 = zip_folder3/
service_name = core/service/zip_files

[core/service/zip_files]
zip_folder = [
    "${core/cli:zip_folder1}",
    "${core/cli:zip_folder2}",
    "${core/cli:zip_folder3}"
  ]
zip_extension = .zip
```

### Start Service

```bash
unitorch-service start path/to/zip/files/service.ini \
    --zip_folder path/to/zip/folder
```

### Stop Service

```bash
unitorch-service stop path/to/zip/files/service.ini \
    --zip_folder path/to/zip/folder
```

### Restart Service

```bash
unitorch-service restart path/to/zip/files/service.ini \
    --zip_folder path/to/zip/folder
```