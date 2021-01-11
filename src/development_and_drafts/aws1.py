import boto3
import pandas as pd

boto3_connection = boto3.resource('s3')
s3 = boto3.client('s3')
s3.download_file('jeopardy-storage', 'data/kids_teen.tsv', 'kids_teen_from_bucket.tsv')
kids_teen = pd.read_csv('kids_teen.tsv', sep='t')
