CREATE SCHEMA IF NOT EXISTS main.fitness;

CREATE TABLE IF NOT EXISTS main.fitness.bronze_peloton_workouts_raw (
  workout_id STRING,
  workout_created_at TIMESTAMP,
  source_payload STRING,
  ingested_at TIMESTAMP
) USING DELTA;

CREATE TABLE IF NOT EXISTS main.fitness.bronze_peloton_metrics_raw (
  workout_id STRING,
  source_payload STRING,
  ingested_at TIMESTAMP
) USING DELTA;

CREATE TABLE IF NOT EXISTS main.fitness.peloton_workouts (
  workout_id STRING,
  status STRING,
  fitness_discipline STRING,
  created_at TIMESTAMP,
  start_time TIMESTAMP,
  end_time TIMESTAMP,
  total_work DOUBLE,
  distance DOUBLE,
  calories DOUBLE,
  device_type STRING,
  ride_id STRING,
  ride_title STRING,
  ride_duration DOUBLE,
  instructor_name STRING
) USING DELTA;

CREATE TABLE IF NOT EXISTS main.fitness.peloton_metrics (
  workout_id STRING,
  metric_name STRING,
  average_value DOUBLE,
  max_value DOUBLE,
  sample_count BIGINT
) USING DELTA;

CREATE OR REPLACE VIEW main.fitness.silver_peloton_workouts AS
SELECT * FROM main.fitness.peloton_workouts;

CREATE OR REPLACE VIEW main.fitness.silver_peloton_metrics AS
SELECT * FROM main.fitness.peloton_metrics;

CREATE OR REPLACE VIEW main.fitness.gold_peloton_daily_summary AS
SELECT
  DATE(start_time) AS workout_date,
  COUNT(*) AS workout_count,
  SUM(calories) AS total_calories,
  AVG(total_work) AS avg_total_work,
  AVG(ride_duration) AS avg_duration
FROM main.fitness.peloton_workouts
GROUP BY DATE(start_time);

CREATE OR REPLACE VIEW main.fitness.gold_peloton_discipline_summary AS
SELECT
  fitness_discipline,
  COUNT(*) AS workouts,
  AVG(calories) AS avg_calories,
  AVG(total_work) AS avg_total_work,
  AVG(ride_duration) AS avg_duration
FROM main.fitness.peloton_workouts
GROUP BY fitness_discipline;

CREATE OR REPLACE VIEW main.fitness.gold_peloton_instructor_summary AS
SELECT
  instructor_name,
  COUNT(*) AS workouts,
  AVG(total_work) AS avg_total_work,
  AVG(calories) AS avg_calories
FROM main.fitness.peloton_workouts
WHERE instructor_name IS NOT NULL
GROUP BY instructor_name;

CREATE OR REPLACE VIEW main.fitness.secure_peloton_workouts AS
SELECT
  sha2(workout_id, 256) AS workout_key,
  status,
  fitness_discipline,
  date(start_time) AS workout_date,
  total_work,
  distance,
  calories,
  ride_title,
  ride_duration,
  instructor_name
FROM main.fitness.peloton_workouts;
