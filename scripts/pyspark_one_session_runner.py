from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder \
    .appName("MyApp") \
    .master("local[*]") \
    .getOrCreate()

# Task 1: Read data from a file and perform transformations
data1 = spark.read.csv("file1.csv", header=True)
# Apply transformations and perform computations on data1

# Task 2: Read data from a database and perform operations
data2 = spark.read.format("jdbc").options(...)\
    .load()
# Apply operations and computations on data2

# Task 3: Read data from a Kafka topic and process streaming data
streaming_data = spark.readStream.format("kafka").options(...)\
    .load()
# Apply streaming transformations and computations on streaming_data

# Stop the SparkSession
spark.stop()
