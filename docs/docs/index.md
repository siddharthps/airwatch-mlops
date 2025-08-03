# Air Quality ML Project documentation!

## Description

MLOps project on Air Quality data

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `aws s3 sync` to recursively sync files in `data/` up to `s3://air-quality-mlops-data-chicago-2025/data/`.
* `make sync_data_down` will use `aws s3 sync` to recursively sync files from `s3://air-quality-mlops-data-chicago-2025/data/` to `data/`.


