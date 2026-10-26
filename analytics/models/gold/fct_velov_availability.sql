with observations as (
    select *
    from {{ ref('stg_velov_history') }}
),

with_next_hour as (
    select
        station_id,
        timestamp,
        capacity,
        bikes_available,
        temperature,
        is_raining,
        lead(timestamp) over (partition by station_id order by timestamp) as target_timestamp,
        lead(bikes_available) over (partition by station_id order by timestamp) as target_bikes_available
    from observations
)

select
    station_id,
    timestamp,
    capacity,
    bikes_available,
    temperature,
    is_raining,
    target_timestamp,
    target_bikes_available
from with_next_hour
where target_timestamp = timestamp + interval '1 hour'
