# Sumedh 2025 Internship


# Workout duration–based scoring Algorithms
Examines deficits (ie how scores short but intense workouts as zero), gives a failed logarithmic adjustment that still ignores workout intensity, & how of duration scoring remains effort-agnostic. Defines attributes/reqs for a "good" scoring function (mixture of WHO-based thresholds + partial credit for short sessions when intensity is high).

[Scoring Formula Variants](./Brainstorming%20Improvements%20to%20the%20Physical%20Activity%20Score%20Computation.pdf)


# Stress Testing Celery/Reddis for Incoming/Streaming Data
Tests varying size batches of DailyLog database entries form small batches like 10-50 entries up to hundreds &  submitting each entry as a Celery background task and measuring how long the entire process took. Caompared submission speed, task completion time, and overall throughput across different to see where the system performs most efficiently.

[Compute Tests](./Reddis%20%26%20Celery%20Execution%20Time.pdf)


# Using Signals as a proxy for On insert

# Client Logging
Frontend and backend logs aren't connected, making it impossible to trace frontend errors back to their root causes in the backend system. Can't correlate frontend errors with backend issues due to disconnected logging systems.

[Client Logging](https://github.com/Preffect-Inc/Preffect-HealthEngine/blob/main/app/views/client_log_view.py)

# Kafka 
The goal was to build a robust, scalable scoring system—one that is invariant to scale, easily modifiable, and supports both real-time database score updates and downstream ML tasks. To achieve this, a resilient publisher-listener architecture was implemented, featuring thread-local producers and consumers that automatically write offsets for reliable processing.

[Kafka Backend](https://github.com/Preffect-Inc/Preffect-HealthEngine/pull/374/files#diff-f0b36047804fc1a021d20667d8da0073a215761639235064f52630a03d570e10)

# Fan Out


# Design 1


# To maximize task parallelism )Graph with Indegree of nodes sempahores)

# Manual Fn roiting 

# Fast/Ninja APIS

# PA pipleine--status


