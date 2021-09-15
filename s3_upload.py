"""
Upload local .csv dataset as .parquet in S3
Plucked and refactored from https://github.com/jacopotagliabue/you-dont-need-a-bigger-boat/blob/main/local_flow/rec/local_dataset_upload.py
"""
import os
import pandas as pd
import click
import os 

def get_filename(path):
    return os.path.splitext(os.path.basename(path))[0]



def upload_file_as_parquet(file_path, target_s3_folder, chunksize=None, partition_cols=None):
    print('Begin reading file {}'.format(file_path))

    s3_file_name = os.path.join(target_s3_folder, get_filename(file_path) + '.parquet')
    if not chunksize is None:
        df_content = next(pd.read_csv(file_path, chunksize=chunksize))
    else:
        df_content = pd.read_csv(file_path)

    print('Begin upload to S3')
    df_content.to_parquet(path=s3_file_name, engine='pyarrow', partition_cols=partition_cols)

    print('Parquet files for {} stored at : {}'.format(file_path, s3_file_name))


@click.command()
@click.option('--sku-to-content-path',\
            default='./SIGIR-ecom-data-challenge/train/sku_to_content.csv',
            envvar="SKU_TO_CONTENT_PATH",\
            help="Path to the sku_to_content.csv path. You can also use environment variable SKU_TO_CONTENT_PATH to set this property",)
@click.option('--browsing-train-path',\
            default='./SIGIR-ecom-data-challenge/train/browsing_train.csv',
            envvar="BROWSING_TRAIN_PATH",\
            help="Path to the browsing_train.csv path. You can also use environment variable BROWSING_TRAIN_PATH to set this property",)
@click.option('--search-train-path',\
            envvar="SEARCH_TRAIN_PATH",\
            default='./SIGIR-ecom-data-challenge/train/search_train.csv',
            help="Path to the search_train.csv path. You can also use environment variable SEARCH_TRAIN_PATH to set this property",)
@click.option('--target-save-path',\
            envvar="TARGET_SAVE_PATH",\
            help="Path to the parquet files. You can also use environment variable TARGET_SAVE_PATH to set this property",)
def upload_files(sku_to_content_path=None,\
                 browsing_train_path=None,\
                 search_train_path=None,\
                 target_save_path=None):
    assert sku_to_content_path is not None
    assert browsing_train_path is not None
    assert search_train_path is not None
    assert target_save_path is not None and 's3://' in target_save_path
    # upload to S3 at some know path under the CartFlow directory
    # for now, upload some rows
    # there is no versioning whatsoever at this stage
    upload_file_as_parquet(sku_to_content_path, target_save_path)
    click.secho(f'Uploaded {sku_to_content_path} to Path : {target_save_path}',fg='green')
    upload_file_as_parquet(browsing_train_path, target_save_path)
    click.secho(f'Uploaded {browsing_train_path} to Path : {target_save_path}',fg='green')
    upload_file_as_parquet(search_train_path, target_save_path)
    click.secho(f'Uploaded {search_train_path} to Path : {target_save_path}',fg='green')
    


if __name__ == '__main__':
    upload_files()