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


# Score Pipeline Architecture

```mermaid
flowchart LR
    %% Input
    A["🏃 User Logs<br/>Exercise"] 
    B["📝 ManualLogService"]
    C["⚡ Celery Task<br/>invoke_scoring_pipeline"]
    
    %% Core Services
    D["🎯 Orchestrator<br/>:8000"]
    E["📊 DB-Reader<br/>:8001"] 
    F["🧮 Activity-Scorer<br/>:8002"]
    G["💾 DB-Writer<br/>:8003"]
    
    %% Database
    H["🗄️ PostgreSQL<br/>Database"]
    
    %% Flow with arrows
    A --> B
    B --> C
    C --> D
    
    %% Data retrieval
    D -->|"GET /get_log_by_id/{id}"| E
    D -->|"GET /get_user_profile/{id}"| E  
    D -->|"GET /same_day_entries/{id}"| E
    D -->|"GET /live_entries_history/{id}"| E
    
    %% Scoring computation  
    D -->|"POST /compute_entry<br/>{workout, profile, history}"| F
    D -->|"POST /compute_live<br/>{workout, profile, history}"| F
    
    %% Data persistence
    D -->|"POST /write_entry<br/>{log_id, scores}"| G
    D -->|"POST /write_live<br/>{user_id, date, scores}"| G
    
    %% Database operations
    E -.->|"Read"| H
    G -.->|"Write"| H
    
    %% Styling
    classDef service fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    classDef external fill:#fff8e1,stroke:#f57c00,stroke-width:2px
    classDef database fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    
    class D,E,F,G service
    class A,B,C external  
    class H database
```

## 📋 **Scoring Flow Details**

### 1️⃣ **Trigger Phase**
```
User Exercise Log → ManualLogService → Celery Task
```

### 2️⃣ **Data Retrieval Phase** 
```
Orchestrator → DB-Reader:
├── Log data (exercise details)
├── User profile (TDEE, BMI, age, gender)
├── Same-day entries (frequency calculation)
├── Calendar history (7-day pattern)
└── Available history (rolling average)
```

### 3️⃣ **Scoring Computation Phase**
```
Orchestrator → Activity-Scorer:
├── Entry Score: duration, intensity, consistency, calories
└── Live Score: daily aggregate + 7-day average
```

### 4️⃣ **Persistence Phase**
```
Orchestrator → DB-Writer:
├── ActivityEntry table (individual workout scores)
└── ActivityLive table (daily aggregate scores)
```

## 🎯 **Score Components**

| Score Type | Components | Purpose |
|------------|------------|---------|
| **Entry** | Duration, Intensity, Consistency, Calories | Individual workout rating |
| **Live** | Duration, Consistency, Calories, Overall, 7-Day | Daily fitness summary |

## Service Details

### 🎯 **Orchestrator** (Port 8000)
- **Role**: Pipeline coordinator
- **Endpoints**: `/api/score_log`, `/health`
- **Function**: Orchestrates the entire scoring workflow

### 📊 **DB-Reader** (Port 8001)  
- **Role**: Data retrieval service
- **Endpoints**: `/get_log_by_id/{id}`, `/get_user_profile/{id}`, `/same_day_entries/{id}`, etc.
- **Function**: Fetches log data, user profiles, and historical entries

### 🧮 **Activity-Scorer** (Port 8002)
- **Role**: Scoring computation engine  
- **Endpoints**: `/compute_entry`, `/compute_live`
- **Function**: Calculates duration, intensity, consistency, and calorie scores

### 💾 **DB-Writer** (Port 8003)
- **Role**: Data persistence service
- **Endpoints**: `/write_entry`, `/write_live` 
- **Function**: Saves computed scores to ActivityEntry and ActivityLive tables

## Data Flow

1. **Trigger**: User logs exercise → ManualLogService queues scoring task
2. **Orchestration**: Orchestrator receives log_id and coordinates pipeline
3. **Data Gathering**: DB-Reader fetches all required data (log, profile, history)
4. **Score Computation**: Activity-Scorer calculates entry and live scores
5. **Persistence**: DB-Writer saves scores to database tables
6. **Response**: Success response with computed scores returned

## Score Types

- **Entry Score**: Individual workout scoring (duration, intensity, consistency, calories)
- **Live Score**: Aggregated daily scoring with 7-day average
- **Overall Score**: Weighted average of all valid component scores


