CREATE OR REPLACE VIEW main.fitness.v_peloton_daily_summary AS
SELECT
  DATE(start_time) AS workout_date,
  COUNT(*) AS workout_count,
  SUM(calories) AS total_calories,
  AVG(total_work) AS avg_total_work,
  AVG(ride_duration) AS avg_duration
FROM main.fitness.peloton_workouts
GROUP BY DATE(start_time)
ORDER BY workout_date DESC;

CREATE OR REPLACE VIEW main.fitness.v_peloton_discipline_summary AS
SELECT
  fitness_discipline,
  COUNT(*) AS workouts,
  AVG(calories) AS avg_calories,
  AVG(total_work) AS avg_total_work,
  AVG(ride_duration) AS avg_duration
FROM main.fitness.peloton_workouts
GROUP BY fitness_discipline
ORDER BY workouts DESC;
