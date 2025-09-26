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

# Failed Design/Directions
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
  worked--killed due to uninterparbility and existance of airflow dag

- **Kafka-Based Event Bus**  
  Provided strong decoupling and flexibility. After implementing, descsion was made that honrizontal scaling in pods could handle flow.

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


# Pipeline Sequence
This diagram is clearest desciption of my pipeline. (Note: I did not make this by hand: credits to Greptile AI reviewer)
```mermaid
sequenceDiagram
    participant C as Client/Trigger
    participant O as Orchestrator
    participant DBR as DB Reader
    participant AS as Activity Scorer
    participant DBW as DB Writer
    participant DB as Database
    
    C->>+O: POST /api/score_log (log_id)
    Note over O: TRACE: 1. Received request
    
    O->>+DBR: GET /get_log_by_id/{log_id}
    DBR->>+DB: Query DailyLog
    DB-->>-DBR: Log data
    DBR-->>-O: Log data + user_id
    Note over O: TRACE: 2. Retrieved log data
    
    O->>+DBR: GET /get_user_profile/{user_id}
    DBR->>+DB: Query UserProfile  
    DB-->>-DBR: Profile data
    DBR-->>-O: Profile data
    Note over O: TRACE: 3. Retrieved user profile
    
    O->>+DBR: GET /same_day_entries/{log_id}
    DBR->>+DB: Query same day ActivityEntries
    DB-->>-DBR: Same day entries
    DBR-->>-O: Same day entries
    Note over O: TRACE: 4. Retrieved same-day entries
    
    O->>+DBR: GET /live_entries_last_calendar_days/{user_id}
    DBR->>+DB: Query 7-day calendar history
    DB-->>-DBR: Calendar history
    DBR-->>-O: Calendar history
    Note over O: TRACE: 5. Retrieved 7-day history
    
    O->>+DBR: GET /live_entries_last_available/{user_id}
    DBR->>+DB: Query available history
    DB-->>-DBR: Available history  
    DBR-->>-O: Available history
    Note over O: TRACE: 6. Retrieved available history
    
    O->>+AS: POST /compute_entry (EntryScoreRequest)
    Note over AS: Calculate duration, intensity,<br/>consistency, calories scores
    AS-->>-O: Entry scores
    Note over O: TRACE: 7. Computed entry scores
    
    O->>+AS: POST /compute_live (LiveScoreRequest)
    Note over AS: Calculate live scores +<br/>7-day rolling average
    AS-->>-O: Live scores
    Note over O: TRACE: 8. Computed live scores
    
    O->>+DBW: POST /write_entry (WriteEntryRequest)
    DBW->>+DB: Update/Create ActivityEntry
    DB-->>-DBW: Entry saved
    DBW-->>-O: Success
    Note over O: TRACE: 9. Saved entry score
    
    O->>+DBW: POST /write_live (WriteLiveRequest) 
    DBW->>+DB: Update/Create ActivityLive
    DB-->>-DBW: Live score saved
    DBW-->>-O: Success
    Note over O: TRACE: 10. Saved live score
    
    O-->>-C: SuccessResponse with scores
    Note over O: TRACE: 11. Pipeline complete


