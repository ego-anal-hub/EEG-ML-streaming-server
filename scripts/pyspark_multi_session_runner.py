from pyspark.sql import SparkSession
from multiprocessing import Process

# Define a function to create and run a SparkSession
def create_and_run_spark_session(app_name):
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .getOrCreate()

    # Perform your Spark operations here

    # Stop the SparkSession
    spark.stop()

# Create and run multiple SparkSession instances in separate processes
if __name__ == "__main__":
    processes = []

    # Create and start the first SparkSession
    p1 = Process(target=create_and_run_spark_session, args=("SparkSession 1",))
    p1.start()
    processes.append(p1)

    # Create and start the second SparkSession
    p2 = Process(target=create_and_run_spark_session, args=("SparkSession 2",))
    p2.start()
    processes.append(p2)

    # Wait for all processes to finish
    for process in processes:
        process.join()
