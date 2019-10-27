"""
    I just create Google Cloud Project
    and connected to BigQuery. Now I have
    whole access to lots of the public datasets.
    Cool!
"""

from google.cloud import bigquery


def get_dataset():
    client = bigquery.Client()
    query = (
        "SELECT name FROM `bigquery-public-data.usa_names.usa_1910_2013`"
        'WHERE state = "TX"'
        'LIMIT 10'
    )
    safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)
    query_job = client.query(
        query,
        job_config=safe_config
    )
    return query_job.to_dataframe()


if __name__ == "__main__":
    dataset = get_dataset()
    print(dataset.head())
