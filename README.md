# Sumedh 2025 Internship
Summary of work/discoveries/research May 2025 - Sept 2025


# Workout duration–based scoring Algorithms
Examines deficits (ie how scores short but intense workouts as zero), gives a failed logarithmic adjustment that still ignores workout intensity, & how of duration scoring remains effort-agnostic. Defines attributes/reqs for a "good" scoring function (mixture of WHO-based thresholds + partial credit for short sessions when intensity is high).

[Scoring Formula Variants](./Brainstorming%20Improvements%20to%20the%20Physical%20Activity%20Score%20Computation.pdf)


# Stress Testing Celery/Reddis for Incoming/Streaming Data
Tests varying size batches of DailyLog database entries form small batches like 10-50 entries up to hundreds &  submitting each entry as a Celery background task and measuring how long the entire process took. Caompared submission speed, task completion time, and overall throughput across different to see where the system performs most efficiently.

[Compute Tests](./Reddis%20%26%20Celery%20Execution%20Time.pdf)

Reddis & Celery Execution Time
Using Signalys.py as a scxoring engine


PA score--Kafka code 


PA score status
