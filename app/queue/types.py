from enum import Enum


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class JobType(StringEnum):
    QUICK_DISSECT = "quick_dissect"
    BULK_DISSECT = "bulk_dissect"


class JobStage(StringEnum):
    DOWNLOAD = "download"
    ANALYZE = "analyze"
    PROCESS = "process"


class JobStatus(StringEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobEventType(StringEnum):
    ENQUEUED = "enqueued"
    CLAIMED = "claimed"
    STAGE_COMPLETED = "stage_completed"
    RETRY_SCHEDULED = "retry_scheduled"
    FAILED = "failed"
    MANUAL_RETRY = "manual_retry"
    LEASE_RECOVERED = "lease_recovered"
    CONTINUATION_ENQUEUED = "continuation_enqueued"
