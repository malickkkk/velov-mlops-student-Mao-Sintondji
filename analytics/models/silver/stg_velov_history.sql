with source as (
    select
        station_id,
        timestamp,
        capacity,
        bikes_available,
        temperature,
        is_raining
    from {{ source('bronze', 'velov_history') }}
),

cleaned as (
    select
        cast(station_id as integer) as station_id,
        cast(timestamp as timestamptz) as timestamp,
        cast(capacity as integer) as capacity,
        cast(bikes_available as integer) as bikes_available,
        cast(temperature as double precision) as temperature,
        cast(is_raining as boolean) as is_raining
    from source
    where station_id > 0
      and capacity > 0
      and bikes_available between 0 and capacity
      and timestamp is not null
)

select * from cleaned
