from prefect import task, get_run_logger

# ---- Prefect Orchestration ----

# Prefect Q1
# What is the difference between a @task and a @flow in Prefect?
# A @flow controls the whole pipeline, decides the order of steps, and brings everything together.
# A @task is a single, small step inside the flow that does one specific job 
# (like loading data from an API or saving to a database).
#
# Would you decorate a simple Celsius-to-Fahrenheit function with @task?
# No, I wouldn't. A pure, in-memory Celsius-to-Fahrenheit calculation does not need to be decorated with @task
# because it is simple, has no I/O, and does not benefit from Prefect's task features such as
# retries or monitoring. It can remain a regular Python helper function.

# Prefect Q2
@task(name="call_api", retries=3, retry_delay_seconds=30)


# Prefect Q3
# Where in the UI do you look to understand what went wrong?
# I would open the failed flow run in the Prefect UI and select the transform task.
# In the task-run details, I would check the logs and error/state information.
#
# What specific information would you expect to find there?
# I would expect to see the Python error message or traceback, any input, any retry
# attempts, and the logs leading up to the failure.


# ---- Production Patterns ----

# Production Q1
# What raise_for_status() does? and 
# raise_for_status() checks whether the API request was successful. If the server returns
# an error like an HTTP 500 error, it raises an exception and causes the task to fail.
#
# Why it is better than writing if response.status_code != 200: print("error") in a pipeline task?
# This is better than just printing an error because print() does not stop the pipeline.
# The next tasks could still run with bad or missing data.
# With raise_for_status(), Prefect knows the task failed, can retry it, and will stop dependent
# downstream tasks from running. With the print() approach, the pipeline keeps going 
# even though the API failed, which could cause problems or bad data in later tasks.


# Production Q2
# What does upsert protect you from in this scenario?
# Upsert protects the pipeline from creating duplicate records when we re-run it from the
# beginning. If a record with the same date already exists, it updates that record
# instead of trying to create another one.
#
# what would happen if you had used plain insert instead?
# If we used plain insert, the records that were already loaded before the crash
# would still be in the database. Re-running the pipeline would try to insert
# those same records again, which could cause duplicate data or a database error
# because the date already exists.


# Production Q3
@task
def load_enriched(enrichment_records: list):
    get_run_logger().info(f"Successfully upserted {len(enrichment_records)} enrichment records.")


# Production Q4
# How the incremental processing check in the transform task contributes to idempotency?
# The incremental processing check helps make the pipeline idempotent because it only
# processes records that have not already been enriched. This mean running the pipeline
# again does not repeatedly process the same records.
#
# What would be the practical consequences if you remove it?
# If we removed the check, the ML and LLM steps would run on all 365 records every time.
# This would increase API/compute costs and make the pipeline take much longer to finish.
# It could also cause data correctness problems because existing enrichment results.