from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark import SparkConf

# Create Spark Session
# spark = SparkSession.builder.appName("SparkKafka2DBStreaming").getOrCreate()

def save_to_db(df, epoch_id):
    df.write \
        .format('jdbc') \
        .option('url', 'jdbc:mysql://localhost/eeg_state') \
        .option('driver', 'com.mysql.cj.jdbc.Driver') \
        .option('dbtable', 'eeg_data') \
        .option('user', 'root') \
        .option('password', '0000') \
        .mode('append') \
        .save()

conf = SparkConf()
conf.set("spark.app.name", "EEG-Store-SparkStreaming")
conf.set("spark.master", "local[*]")
conf.set("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0")
conf.set("spark.jars", "file:///C:/Users/cshiz/Desktop/2023_1/Capstone/workspace/DataAnalysisServer/resources/mysql-connector-j-8.0.32.jar")
conf.set("spark.driver.extraJavaOptions", "-Dspark.driver.extraJavaOptions=-Dsun.io.serialization.extendedDebugInfo=true")


spark = SparkSession.builder.config(conf=conf).getOrCreate()

# Define schema for our data
eeg_schema = StructType([
    StructField("time", StringType(), False),
    StructField("space", StructType([
        StructField("id", StringType(), False)
    ])),
    StructField("person", StructType([
        StructField("id", StringType(), False),
        StructField("eeg", StructType([
            StructField("ch1", StructType([
                StructField("pos", StringType(), False),
                StructField("value", FloatType(), False)
            ])),
            StructField("ch2", StructType([
                StructField("pos", StringType(), False),
                StructField("value", FloatType(), False)
            ])),
            StructField("ch3", StructType([
                StructField("pos", StringType(), False),
                StructField("value", FloatType(), False)
            ])),
            StructField("ch4", StructType([
                StructField("pos", StringType(), False),
                StructField("value", FloatType(), False)
            ])),
        ])),
        StructField("state", StructType([
            StructField("reasonal", StructType([
                StructField("concentration", IntegerType(), False),
                StructField("activity", IntegerType(), False)
            ])),
            StructField("emotional", StructType([
                StructField("state", IntegerType(), False),
                StructField("preference", IntegerType(), False)
            ])),
        ]))
    ]))
])

# Create DataFrame representing the stream of input lines from connection to Kafka
kafka_df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka_ip:9092") \
    .option("subscribe", "raw-eeg-marked") \
    .option("group.id", "da-group") \
    .option('client_id', 'eeg-da-server-consumer-4') \
    .option("startingOffsets", "latest") \
    .load() \
    .selectExpr("CAST(value AS STRING)")
    # .select(from_json(col("value").cast("string"), schema).alias("data")) \
    # .select("data.*")

eeg_data = kafka_df.select(from_json(regexp_replace(col("value"), '\\\\|(^")|("$)', ''), eeg_schema).alias("eeg_data")).select( \
    col("eeg_data.time"),
    col("eeg_data.space.id").alias("space_id"),
    col("eeg_data.person.id").alias("person_id"),
    col("eeg_data.person.eeg.ch1.pos").alias("ch1_pos"),
    col("eeg_data.person.eeg.ch1.value").alias("ch1_value"),
    col("eeg_data.person.eeg.ch2.pos").alias("ch2_pos"),
    col("eeg_data.person.eeg.ch2.value").alias("ch2_value"),
    col("eeg_data.person.eeg.ch3.pos").alias("ch3_pos"),
    col("eeg_data.person.eeg.ch3.value").alias("ch3_value"),
    col("eeg_data.person.eeg.ch4.pos").alias("ch4_pos"),
    col("eeg_data.person.eeg.ch4.value").alias("ch4_value"),
    col("eeg_data.person.state.reasonal.concentration").alias("concentration"),
    col("eeg_data.person.state.reasonal.activity").alias("activity"),
    col("eeg_data.person.state.emotional.state").alias("state"),
    col("eeg_data.person.state.emotional.preference").alias("preference"),
)


# Start running the query that prints the running counts to the console
query = eeg_data \
    .writeStream \
    .foreachBatch(save_to_db) \
    .start()

# query = eeg_data \
#     .writeStream \
#     .outputMode("append") \
#     .format("console") \
#     .option("truncate", "false") \
#     .start()

query.awaitTermination()
