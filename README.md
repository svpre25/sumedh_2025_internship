# Sumedh 2025 Internship


# Workout duration–based scoring Algorithms
Examines deficits (ie how scores short but intense workouts as zero), gives a failed logarithmic adjustment that still ignores workout intensity, & how of duration scoring remains effort-agnostic. Defines attributes/reqs for a "good" scoring function (mixture of WHO-based thresholds + partial credit for short sessions when intensity is high).

[Scoring Formula Variants](./Brainstorming%20Improvements%20to%20the%20Physical%20Activity%20Score%20Computation.pdf)


# Stress Testing Celery/Reddis for Incoming/Streaming Data
Tests varying size batches of DailyLog database entries form small batches like 10-50 entries up to hundreds &  submitting each entry as a Celery background task and measuring how long the entire process took. Caompared submission speed, task completion time, and overall throughput across different to see where the system performs most efficiently.

[Compute Tests](./Reddis%20%26%20Celery%20Execution%20Time.pdf)


# Client Logging
Frontend and backend logs aren't connected, making it impossible to trace frontend errors back to their root causes in the backend system. Can't correlate frontend errors with backend issues due to disconnected logging systems.

[Client Logging](https://github.com/Preffect-Inc/Preffect-HealthEngine/blob/main/app/views/client_log_view.py)

# Kafka Pub/Sub + Fanout
The goal was to build a robust, scalable scoring system—one that is invariant to scale, easily modifiable, and supports both real-time database score updates and downstream ML tasks. To achieve this, a resilient publisher-listener architecture was implemented, featuring thread-local producers and consumers that automatically write offsets for reliable processing.

[Kafka Backend](https://github.com/Preffect-Inc/Preffect-HealthEngine/pull/374/files#diff-f0b36047804fc1a021d20667d8da0073a215761639235064f52630a03d570e10)

# Failed Designs
- **Using Django Signal**  
  Tempting for loose coupling, but problematic with Celery. Signal handlers can conflict with async task execution and complicate debugging.

- **Manual Function Routing**  
  `exec(fn + param_name)` — technically worked, but it’s an unsafe and unreadable reinvention of dispatch mechanisms. Better to rely on native routing tools. Killed.

- **Dict Packing/Unpacking**  
  Trying to avoid “slinging” by deeply nesting/unpacking dicts. Resulted in unreadable code — killed.

- **Graph-Based Execution Engine**  
  - Tasks = nodes  
  - Edges into node = required vars (used to calculate semaphores)  
  - Used topological sort for serializability  
  - Used level-order traversal to maximize parallelism  
  worked--killed

- **Kafka-Based Event Bus**  
  Provided strong decoupling and flexibility. After implementatio

# Physical Activity Pipeline: Working
<pre lang="markdown"><code>```
When a "user" sends a physical activity log-entry to our server:
User
  → Ingress
    → Django View
      → Manual Log Service
        → Daily Log Entry
          → Enqueue Celery Task to invoke pipeline
            → Orchestrator receives Log ID to process (FastAPI)
              → Invokes DB Reader Microservice (FastAPI)
              → Invokes Scoring Microservice (FastAPI)
              → Invokes DB Writer Microservice (FastAPI)
```</code></pre>

orchestrator-7c7f7fbf84-qt7jb orchestrator SCORE TRACE: 1. Received request for log c392d69a-8d0c-420b-8166-ebf3222e3bd5
db-reader-5867bbff8c-648sz db-reader SCORE TRACE (DB): Fetching log c392d69a-8d0c-420b-8166-ebf3222e3bd5
orchestrator-7c7f7fbf84-qt7jb orchestrator SCORE TRACE: GET /get_log_by_id/c392d69a-8d0c-420b-8166-ebf3222e3bd5 -> {'user_id': '7c9b01d3-9f89-401c-af37-3c17d6dc355f', 'log_id': 'c392d69a-8d0c-420b-8166-ebf3222e3bd5', 'created_at': '2025-09-26T15:44:01.351528Z', 'datetime_filter': '2024-09-26T14:30:00Z', 'timezone': 'UTC', 'log_type': 'exercise', 'log_access_origin': 'quest_id: 74042272-ca60-410b-9cce-1b597643329f', 'source': 'manual', 'entry_method': ['manual', 'audio'], 'priority': 'first', 'data': {'notes': 'optional_string', 'new_log': True, 'duration': 6666.0, 'end_time': '2024-09-26T14:30:00+00:00', 'intensity': 1.0, 'start_time': '2024-09-26T14:30:00+00:00', 'specific_activity': 'screaming', 'distance_or_amount': 'optional string'}, 'meta_data': {}}
db-reader-5867bbff8c-648sz db-reader SCORE TRACE (DB): Fetching user profile 7c9b01d3-9f89-401c-af37-3c17d6dc355f
orchestrator-7c7f7fbf84-qt7jb orchestrator SCORE TRACE: GET /get_user_profile/7c9b01d3-9f89-401c-af37-3c17d6dc355f -> {'user_id': '7c9b01d3-9f89-401c-af37-3c17d6dc355f', 'date_of_birth': '1995-01-01', 'gender': 'male', 'height': '175', 'zip_code': '12345', 'additional_notes': 'Created for scoring pipeline testing', 'TDEE': 2000.0, 'BMR': 1500.0, 'BMI': 22.5}
orchestrator-7c7f7fbf84-qt7jb orchestrator SCORE TRACE: GET /same_day_entries/c392d69a-8d0c-420b-8166-ebf3222e3bd5 -> [{'id': 1, 'daily_log_id': '491d6da7-65bd-4497-b5eb-586b88ff5c10', 'duration_score': 100, 'intensity_score': 20, 'consistency_score': 80, 'calories_score': 100, 'overall_score': 75}, {'id': 3, 'daily_log_id': '45e1a02f-bf58-496e-ba20-8576e09e519f', 'duration_score': 100, 'intensity_score': 20, 'consistency_score': 40, 'calories_score': 0, 'overall_score': 53}, {'id': 4, 'daily_log_id': 'be001428-d8bd-45c9-b29e-7e90723b25fc', 'duration_score': 100, 'intensity_score': 20, 'consistency_score': 60, 'calories_score': 0, 'overall_score': 60}]
orchestrator-7c7f7fbf84-qt7jb orchestrator SCORE TRACE: GET /live_entries_last_calendar_days/7c9b01d3-9f89-401c-af37-3c17d6dc355f -> []
orchestrator-7c7f7fbf84-qt7jb orchestrator SCORE TRACE: GET /live_entries_last_available/7c9b01d3-9f89-401c-af37-3c17d6dc355f -> [{'id': 1, 'user_id': '7c9b01d3-9f89-401c-af37-3c17d6dc355f', 'date': '2024-09-26', 'duration_score': 100, 'consistency_score': 20, 'calories_score': 100, 'overall_score': 75, 'seven_day_score': 75}]
activity-scorer-5fb57d9fc8-nf2zf activity-scorer SCORE TRACE (Scorer): Computing entry score for workout
orchestrator-7c7f7fbf84-qt7jb orchestrator SCORE TRACE: POST /compute_entry -> {'duration_score': 100, 'intensity_score': 20, 'consistency_score': 80, 'calories_score': None, 'overall_score': 67}
activity-scorer-5fb57d9fc8-nf2zf activity-scorer SCORE TRACE (Scorer): Entry scores: duration_score=100 intensity_score=20 consistency_score=80 calories_score=None overall_score=67
activity-scorer-5fb57d9fc8-nf2zf activity-scorer SCORE TRACE (Scorer): Computing live score with 0 calendar days, 1 available days
activity-scorer-5fb57d9fc8-nf2zf activity-scorer SCORE TRACE (Scorer): Live scores (freq from 1 calendar days, avg from 1 available): duration_score=100 consistency_score=20 calories_score=None overall_score=67 seven_day_score=75
orchestrator-7c7f7fbf84-qt7jb orchestrator SCORE TRACE: POST /compute_live -> {'duration_score': 100, 'consistency_score': 20, 'calories_score': None, 'overall_score': 67, 'seven_day_score': 75}
db-writer-6495d9c9fc-qkbpw db-writer SCORE TRACE (DB Writer): Writing entry score for log c392d69a-8d0c-420b-8166-ebf3222e3bd5
db-writer-6495d9c9fc-qkbpw db-writer SCORE TRACE (DB Writer): Created ActivityEntry 7
orchestrator-7c7f7fbf84-qt7jb orchestrator SCORE TRACE: POST /write_entry -> {'id': 7, 'daily_log_id': 'c392d69a-8d0c-420b-8166-ebf3222e3bd5', 'duration_score': 100, 'intensity_score': 20, 'consistency_score': 80, 'calories_score': 0, 'overall_score': 67}
db-writer-6495d9c9fc-qkbpw db-writer SCORE TRACE (DB Writer): Writing live score for user 7c9b01d3-9f89-401c-af37-3c17d6dc355f on 2024-09-26
db-writer-6495d9c9fc-qkbpw db-writer SCORE TRACE (DB Writer): Updated ActivityLive 1
orchestrator-7c7f7fbf84-qt7jb orchestrator SCORE TRACE: POST /write_live -> {'id': 1, 'user_id': '7c9b01d3-9f89-401c-af37-3c17d6dc355f', 'date': '2024-09-26', 'duration_score': 100, 'consistency_score': 20, 'calories_score': 0, 'overall_score': 67, 'seven_day_score': 75}
orchestrator-7c7f7fbf84-qt7jb orchestrator SCORE TRACE: 11. Scoring pipeline complete for log c392d69a-8d0c-420b-8166-ebf3222e3bd5

