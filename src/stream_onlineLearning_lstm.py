import findspark
findspark.init()

from pyspark import SparkConf
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
import json
import numpy as np
import operator
from tensorflow.keras.models import load_model

import os
from glob import glob
# import pandas as pd

conf = SparkConf()
conf.set("spark.app.name", "EEG-Analysis-State-SparkStreaming")
conf.set("spark.master", "local[*]")
conf.set("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0")
conf.set("spark.sql.shuffle.partitions", "1")
conf.set("spark.sql.streaming.statefulOperator.checkCorrectness.enabled", "false")
conf.set("spark.sql.execution.arrow.enabled", "true")
# conf.set("spark.driver.memory", "2g")  # Increase this as needed
# conf.set("spark.executor.memory", "2g")  # Increase this as needed
# conf.set("spark.sql.shuffle.partitions", "4")

spark = SparkSession.builder.config(conf=conf).getOrCreate()

# Create DataFrame representing the stream of input lines from connection to Kafka
kafka_df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka_ip:9092") \
    .option("subscribe", "raw-eeg-marked") \
    .option('client_id', 'eeg-da-server-consumer-onlineLearning') \
    .option('group_id', 'da-group') \
    .load() \
    .selectExpr("CAST(value AS STRING)")

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

eeg_data = kafka_df.select(from_json(regexp_replace(col("value"), '\\\\|(^")|("$)', ''), eeg_schema).alias("eeg_data")).select( \
    col("eeg_data.time"),
    col("eeg_data.space.id").alias("spaceId"),
    col("eeg_data.person.id").alias("personalId"),
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

# 타임스탬프를 Spark의 'timestamp' 형식으로 변환
eeg_data = eeg_data.withColumn("time", 
                   expr("concat(substr(time, 1, 4), '-', substr(time, 5, 2), '-', substr(time, 7, 2), ' ', substr(time, 9, 2), ':', substr(time, 11, 2), ':', substr(time, 13, 2), '.', substr(time, 15, 3))"))
eeg_data = eeg_data.withColumn("time", to_timestamp(col("time"), "yyyy-MM-dd HH:mm:ss.SSS"))

# Define the udf for LSTM model prediction
@udf(returnType=StringType())
def predict_udf(spaceid, personalid, ch1_values, ch2_values, ch3_values, ch4_values):
    # Create an array of shape (n, 4) where n is the length of the input arrays
    if len(ch1_values) > 256:
        ch1_values = ch1_values[:len(ch1_values) - (len(ch1_values) % 256)]
        ch2_values = ch2_values[:len(ch1_values) - (len(ch1_values) % 256)]
        ch3_values = ch3_values[:len(ch1_values) - (len(ch1_values) % 256)]
        ch4_values = ch4_values[:len(ch1_values) - (len(ch1_values) % 256)]
        features_array = np.column_stack((ch1_values, ch2_values, ch3_values, ch4_values))
        features_array = features_array.reshape(-1, 256, 4)

        folder_path = "../models/LSTM/online_learn_models/"
        model_files = glob(os.path.join(folder_path, "eeg2emotionstate_LSTM_online_" + spaceid + "_" + personalid))
        sorted_model_files = sorted(model_files, key=os.path.getmtime, reverse=True)
        personal_model = load_model(sorted_model_files[0])

        # Fit the model to the data chunk
        personal_model.train_on_batch(features_array, labels_chunk)
        
        # Save the model after each batch if needed
        personal_model.save("../models/LSTM/online_learn_models/eeg2emotionstate_LSTM_online_" + spaceid + "_" + personalid)

        return 'new learned'

# 사용자 정의 함수로 변환
# predict_udf = udf(predict_udf, ArrayType(FloatType()))
spark.udf.register("predict_udf", predict_udf)

# 윈도우 슬라이딩 처리를 위한 설정
window_duration = "5 seconds"
slide_duration = "4 seconds"
# .withWatermark("time", "1 minutes")
eeg_data_windowed = eeg_data.withWatermark("time", "1 microseconds").withColumn(
    "window",
    window(col("time"), window_duration, slide_duration)
).groupBy(
    "window", "spaceId", "personalId"
).agg(
    predict_udf(col("spaceId"), col("personalId"), collect_list(col("ch1_value")), collect_list(col("ch2_value")), collect_list(col("ch3_value")), collect_list(col("ch4_value"))).alias("result")
)

query = eeg_data_windowed \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .start()

query.awaitTermination()
