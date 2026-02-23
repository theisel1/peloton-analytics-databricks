CREATE CATALOG IF NOT EXISTS main;
CREATE SCHEMA IF NOT EXISTS main.fitness;

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
);

CREATE TABLE IF NOT EXISTS main.fitness.peloton_metrics (
  workout_id STRING,
  metric_name STRING,
  average_value DOUBLE,
  max_value DOUBLE,
  sample_count BIGINT
);
