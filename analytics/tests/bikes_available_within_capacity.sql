select *
from {{ ref('stg_velov_history') }}
where bikes_available < 0
   or bikes_available > capacity
