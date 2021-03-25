from ..logger import get_logger

logger = get_logger(__name__.split(".", 1)[-1])

from ..monitoring.time_it import timing

#@timing
def dask_repartition(df):
        logger.debug('Partitions ' + str(df.npartitions)+ ' ... repartitioning ...')
        df = df.repartition(partition_size="100MB")
        logger.debug('Partitions: ' + str(df.npartitions))
        return df