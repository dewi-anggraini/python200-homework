# --- Supabase Connection ----
# Q1
# supabase-py needs to connect to project's URL and the anon/public API key.
# Both are stored in the Supabase dashboard under Project Settings -> API. 
# They should be stored in an ignored .env file instead of being hardcoded because committed credentials
# can be copied from a public repository and used by someone else.

# Q2
import os 
from dotenv import load_dotenv
from supabase import create_client


def get_client():
    load_dotenv()
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url:
        raise ValueError('SUPABASE_URL env is missing')
    if not key: 
        raise ValueError('SUPABASE_KEY env is missing')
    
    return create_client(url, key)

supabase = get_client()

# Q3
# Row Level Security (RLS) is a PostgreSQL secutiry feature that controls which rows in a table a user can access or modify. 
# It is disabled for this course to simplify development and allow the Python program to insert and
# access the weather data without creating additional RLS policies. I would keep RLS enabled in a real multi-user application, such as a
# medical record tracker where users must only be able to access their own records.



# --- supabase-py CRUD ---
# Q1
record = {
    "date": "2026-08-24",
    "temperature_2m_max": 22.0,
    "temperature_2m_min": 16.0,
    "precipitation_sum": 0.0,
    "wind_speed_10m_max": 44.0,
}
def insert_test_record(supabase):
    response = supabase.table("weather_raw").insert(record).execute()
    return response.data
    
insert_test_record(supabase)

# Comment:
# Running this function twice would cause a duplicate key error, the second insert will fail because "date"
# is the primary key and the record for 2026-08-24 already exists.
# Replacing insert() with upsert(record, on_conflict="date") makes the write safe to repeat.


# Q2
def get_records_by_date_range(supabase, start, end):
    response = supabase.table("weather_raw").select('*').gte('date', start).lte('date', end).execute()
    return response.data

print(get_records_by_date_range(supabase, "2026-08-23", "2026-08-25"))


# Q3
# Plain 'insert' adds new rows and can result in an error if the record already exists. 
# 'upsert', inserts a new row if the key is new, or updates the existing row if there's a conflict.
# Example:
# we use insert when we know the records are new. And we use upsert when
# loading weather data that may already exist for the same date.

def safe_upsert(supabase, records):
    response = supabase.table("weather_raw").upsert(records, on_conflict="date").execute()
    print(f"Rows affected: {len(response.data)}")
    return response.data

safe_upsert(supabase, [record])

# --- Idempotency ---
# Q1
# idempotence is important because it makes the pipeline stable, 
# saves resources, prevents errors, and makes everything faster and easier to manage.
# For example: when we run processes 8,000 of 10,000 records successfully, then fails. 
# The retry starts from the beginning and reprocesses the 8,000 records that already loaded. 
# Without idempotency, these 8,000 records now exist twice.
