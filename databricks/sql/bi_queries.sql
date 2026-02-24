-- Daily performance trend
SELECT *
FROM main.fitness.gold_peloton_daily_summary
ORDER BY workout_date DESC;

-- Discipline split
SELECT *
FROM main.fitness.gold_peloton_discipline_summary
ORDER BY workouts DESC;

-- Top instructors by average output
SELECT *
FROM main.fitness.gold_peloton_instructor_summary
ORDER BY avg_total_work DESC;

-- Secure shareable dataset
SELECT *
FROM main.fitness.secure_peloton_workouts
ORDER BY workout_date DESC;
