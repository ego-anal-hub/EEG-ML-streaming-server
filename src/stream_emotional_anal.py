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

# Kafka에서 스트림 데이터 읽기
kafka_df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka_ip:9092") \
    .option("subscribe", "raw-eeg") \
    .option('client_id', 'eeg-da-server-consumer-3') \
    .option('group_id', 'da-group') \
    .load() \
    .selectExpr("CAST(value AS STRING)")

eeg_schema = StructType([
    StructField("time", StringType()),
    StructField("space", StructType([
        StructField("id", StringType())
    ])),
    StructField("person", StructType([
        StructField("id", StringType()),
        StructField("eeg", StructType([
            StructField("ch1", StructType([
                StructField("pos", StringType()),
                StructField("value", FloatType())
            ])),
            StructField("ch2", StructType([
                StructField("pos", StringType()),
                StructField("value", FloatType())
            ])),
            StructField("ch3", StructType([
                StructField("pos", StringType()),
                StructField("value", FloatType())
            ])),
            StructField("ch4", StructType([
                StructField("pos", StringType()),
                StructField("value", FloatType())
            ]))
        ]))
    ]))
])

eeg_data = kafka_df.select(from_json(regexp_replace(col("value"), '\\\\|(^")|("$)', ''), eeg_schema).alias("eeg_data")) \
    .select(
        col("eeg_data.time").alias("time"),
        col("eeg_data.space.id").alias("spaceId"),
        col("eeg_data.person.id").alias("personalId"),
        col("eeg_data.person.eeg.ch1.value").alias("ch1_value"),
        col("eeg_data.person.eeg.ch2.value").alias("ch2_value"),
        col("eeg_data.person.eeg.ch3.value").alias("ch3_value"),
        col("eeg_data.person.eeg.ch4.value").alias("ch4_value")
    )

# 타임스탬프를 Spark의 'timestamp' 형식으로 변환
eeg_data = eeg_data.withColumn("time", 
                   expr("concat(substr(time, 1, 4), '-', substr(time, 5, 2), '-', substr(time, 7, 2), ' ', substr(time, 9, 2), ':', substr(time, 11, 2), ':', substr(time, 13, 2), '.', substr(time, 15, 3))"))
eeg_data = eeg_data.withColumn("time", to_timestamp(col("time"), "yyyy-MM-dd HH:mm:ss.SSS"))

# Define the udf for LSTM model prediction
@udf(returnType=ArrayType(IntegerType()))
def predict_udf(ch1_values, ch2_values, ch3_values, ch4_values):
    # Create an array of shape (n, 4) where n is the length of the input arrays
    result = []
    if len(ch1_values) > 256:
        # ch1_values = ch1_values[-256:]
        # ch2_values = ch2_values[-256:]
        # ch3_values = ch3_values[-256:]
        # ch4_values = ch4_values[-256:]
        ch1_values = ch1_values[:len(ch1_values) - (len(ch1_values) % 256)]
        ch2_values = ch2_values[:len(ch2_values) - (len(ch2_values) % 256)]
        ch3_values = ch3_values[:len(ch3_values) - (len(ch3_values) % 256)]
        ch4_values = ch4_values[:len(ch4_values) - (len(ch4_values) % 256)]
        features_array = np.column_stack((ch1_values, ch2_values, ch3_values, ch4_values))
        features_array = features_array.reshape(-1, 256, 4)

        folder_path = "../models/LSTM/"
        model_files = glob(os.path.join(folder_path, "eeg2emotionstate_LSTM*"))
        sorted_model_files = sorted(model_files, key=os.path.getmtime, reverse=True)
        most_recent_model = load_model(sorted_model_files[0])

        # Extract the file title (filename without extension)
        # first_file = sorted_model_files[0]
        # most_recent_model = os.path.splitext(os.path.basename(first_file))[0]
        
        # Perform the LSTM prediction using the loaded model
        prediction = most_recent_model.predict(features_array)

        state = int(np.argmax(prediction, axis=1).astype(int)[0])

        if state == 1 : preference = -1
        elif state == 2 : preference = 0
        elif (state > 2) or (state < 1) : preference = 1
        
        # state{
        # 0: '공포', '분노', '슬픔', '혼란'
        # 1: '불안', '후회', '짜증', '실망'
        # 2: '놀람', '호기심', '관심'
        # 3: '만족', '희망', '기대', '안심'
        # 4: '기쁨', '행복', '열정', '사랑'}
        # preference{긍정: 1, 보통: 0, 부정: -1}
        result = [state, preference]
    return result

# 사용자 정의 함수로 변환
# predict_udf = udf(predict_udf, ArrayType(FloatType()))
spark.udf.register("predict_udf", predict_udf)

# 윈도우 슬라이딩 처리를 위한 설정
window_duration = "10 seconds"
slide_duration = "9 seconds"
# .withWatermark("time", "1 minutes")
eeg_data_windowed = eeg_data.withWatermark("time", "1 microseconds").withColumn(
    "window",
    window(col("time"), window_duration, slide_duration)
).groupBy(
    "window", "spaceId", "personalId"
).agg(
    predict_udf(collect_list(col("ch1_value")), collect_list(col("ch2_value")), collect_list(col("ch3_value")), collect_list(col("ch4_value"))).alias("result")
)

result_data = eeg_data_windowed.select(
    date_format(col("window.end"), "yyyyMMddHHmmssSSS").alias("time"),
    struct(
        col("spaceId").alias("id")
    ).alias("space"),
    struct(
        col("personalId").alias("id"),
        struct(
            struct(col("result")[0].alias("state"), col("result")[1].alias("preference")).alias("emotional")
        ).alias("state")
    ).alias("person")
)

# JSON 변환
result_data_json = result_data.select(to_json(struct(col("*"))).alias("value"))

# Kafka로 내보내기
query = result_data_json \
    .writeStream \
    .outputMode("append") \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka_ip:9092") \
    .option("topic", "state-emotional") \
    .option("client_id", 'eeg-da-server-producer-3') \
    .option("checkpointLocation", "C:/Users/cshiz/spark/spark-3.4.0-bin-hadoop3/checkpoint_3") \
    .start()

# query = eeg_data_windowed \
#     .writeStream \
#     .outputMode("append") \
#     .format("console") \
#     .option("truncate", "false") \
#     .start()

query.awaitTermination()
