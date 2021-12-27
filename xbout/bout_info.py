_BOUT_PER_PROC_VARIABLES = [
    "wall_time",
    "wtime",
    "wtime_rhs",
    "wtime_invert",
    "wtime_comms",
    "wtime_io",
    "wtime_per_rhs",
    "wtime_per_rhs_e",
    "wtime_per_rhs_i",
    "PE_XIND",
    "PE_YIND",
    "MYPE",
]
_BOUT_PER_PROC_VARIABLES_REQUIRED_FROM_RESTARTS = ["hist_hi", "tt"]
_BOUT_TIME_DEPENDENT_META_VARS = ["iteration"]
