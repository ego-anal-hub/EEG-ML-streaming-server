from pyspark import SparkConf
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
import json
import numpy as np
from scipy.fft import fft
import operator
# import pandas as pd

import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta

import findspark
findspark.init()


# 사용자 정의 함수 (FFT 처리)
def apply_fft(values):
    ############### Pre-Processing #####################
    eeg_data = values  
    sampling_rate = 256

    # Generate spectrograms for each EEG channel
    spectrograms = []

    for channel in eeg_data:
        frequencies, times, Sxx = signal.spectrogram(channel, fs=sampling_rate)
        spectrograms.append(Sxx)

    spectrograms = np.array(spectrograms)

    ############### Training #####################

    # 데이터 정규화
    scaler = StandardScaler()
    X = scaler.fit_transform(spectrograms)

    # 이미 학습된 모델 불러오기
    model = load_model('trained_model.h5')

    # 모델 컴파일:
    model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])

    # 예측 결과 분석:
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)

    return [int(v) for v in y_pred_classes]
   

# 사용자 정의 함수로 변환
udf_apply_fft = udf(apply_fft, ArrayType(IntegerType()))

# SparkSession 생성
conf = SparkConf()
conf.set("spark.app.name", "EEG-Analysis-SparkStreaming")
conf.set("spark.master", "local[*]")
conf.set("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0")
conf.set("spark.sql.shuffle.partitions", "4")
conf.set("spark.sql.streaming.statefulOperator.checkCorrectness.enabled", "false")
conf.set("spark.sql.execution.arrow.enabled", "true")
# conf.set("spark.driver.memory", "2g")  # Increase this as needed
# conf.set("spark.executor.memory", "2g")  # Increase this as needed

spark = SparkSession.builder.config(conf=conf).getOrCreate()

# Kafka에서 스트림 데이터 읽기
kafka_df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "raw-eeg") \
    .option('client_id', 'eeg-da-server-consumer-1') \
    .option('group_id', 'da-group') \
    .load() \
    .selectExpr("CAST(value AS STRING)")

eeg_schema = StructType([
    StructField("time", StringType(), nullable=False),
    StructField("space", StructType([
        StructField("id", StringType(), nullable=False)
    ]), nullable=False),
    StructField("person", StructType([
        StructField("id", StringType(), nullable=False),
        StructField("eeg", StructType([
            StructField("ch1", StructType([
                StructField("pos", StringType(), nullable=False),
                StructField("value", FloatType(), nullable=False)
            ]), nullable=False),
            StructField("ch2", StructType([
                StructField("pos", StringType(), nullable=False),
                StructField("value", FloatType(), nullable=False)
            ]), nullable=False),
            StructField("ch3", StructType([
                StructField("pos", StringType(), nullable=False),
                StructField("value", FloatType(), nullable=False)
            ]), nullable=False),
            StructField("ch4", StructType([
                StructField("pos", StringType(), nullable=False),
                StructField("value", FloatType(), nullable=False)
            ]), nullable=False)
        ]), nullable=False)
    ]), nullable=False)
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

# 윈도우 슬라이딩 처리를 위한 설정
window_duration = "10 milliseconds"
slide_duration = "10 milliseconds"

eeg_data_windowed = eeg_data.withWatermark("time", "1 microseconds").withColumn(
    "window",
    window(col("time"), window_duration, slide_duration)
).groupBy(
    "window", "spaceId", "personalId"
).agg(
    udf_apply_fft(collect_list(col("ch1_value"))).alias("ch1_color"),
    udf_apply_fft(collect_list(col("ch2_value"))).alias("ch2_color"),
    udf_apply_fft(collect_list(col("ch3_value"))).alias("ch3_color"),
    udf_apply_fft(collect_list(col("ch4_value"))).alias("ch4_color"),
)

result_data = eeg_data_windowed.select(
    date_format(col("window.end"), "yyyyMMddHHmmssSSS").alias("time"),
    struct(
        col("spaceId").alias("id")
    ).alias("space"),
    struct(
        col("personalId").alias("id"),
        struct(
            struct(lit("TP9").alias("pos"), col("ch1_color")[0].alias("r"), col("ch1_color")[1].alias("g"), col("ch1_color")[2].alias("b")).alias("ch1"),
            struct(lit("AF7").alias("pos"), col("ch2_color")[0].alias("r"), col("ch2_color")[1].alias("g"), col("ch2_color")[2].alias("b")).alias("ch2"),
            struct(lit("AF8").alias("pos"), col("ch3_color")[0].alias("r"), col("ch3_color")[1].alias("g"), col("ch3_color")[2].alias("b")).alias("ch3"),
            struct(lit("TP10").alias("pos"), col("ch4_color")[0].alias("r"), col("ch4_color")[1].alias("g"), col("ch4_color")[2].alias("b")).alias("ch4")
        ).alias("color")
    ).alias("person")
)


# JSON 변환
result_data_json = result_data.select(to_json(struct(col("*"))).alias("value"))

# Kafka로 내보내기
query = result_data_json \
    .writeStream \
    .outputMode("append") \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "unordered-color-spectrum") \
    .option("client_id", 'eeg-da-server-producer-1') \
    .option("checkpointLocation", "C:/Users/cshiz/spark/spark-3.4.0-bin-hadoop3/checkpoint") \
    .start()

# query = result_data_json \
#     .writeStream \
#     .outputMode("append") \
#     .format("console") \
#     .option("truncate", "false") \
#     .start()

query.awaitTermination()