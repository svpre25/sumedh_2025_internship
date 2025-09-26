# Sumedh 2025 Internship
Summary of work/discoveries/research May 2025 - Sept 2025


**Analysis of a workout duration–based scoring Algorithm:**
Examines deficits (ie how scores short but intense workouts as zero), gives a failed logarithmic adjustment that still ignores workout intensity, & how of duration scoring remains effort-agnostic. Defines attributes/reqs for a "good" scoring function (mixture of WHO-based thresholds + partial credit for short sessions when intensity is high).


**Stress Testing Celery/Reddis for Incoming/Streaming Data**
The experiment tested batch processing performance by creating different sized batches of DailyLog database entries (likely ranging from small batches like 10-50 entries up to larger batches of hundreds or thousands), then submitting each entry as a Celery background task and measuring how long the entire process took. The goal was to determine the optimal batch size by comparing submission speed, task completion time, and overall throughput across different batch sizes to see where the system performs most efficiently.





